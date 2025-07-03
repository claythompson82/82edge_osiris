# dgm_kernel/meta_loop.py
"""
Darwin Gödel Machine — self-improving meta-loop (async capable)
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, cast

import redis

from dgm_kernel import metrics
from dgm_kernel.llm_client import draft_patch
from dgm_kernel.mutation_strategies import MutationStrategy
from dgm_kernel.prover import _get_pylint_score as _prover_pylint_score
from dgm_kernel.prover import prove_patch
from dgm_kernel.sandbox import run_patch_in_sandbox
from llm_sidecar.reward import proofable_reward

# ─────────────────────────────── globals & config ────────────────────────────
log = logging.getLogger(__name__)

REDIS = redis.Redis(host="redis", port=6379, decode_responses=True)

PATCH_QUEUE = "dgm:patch_queue"
TRACE_QUEUE = "dgm:recent_traces"
APPLIED_LOG = "dgm:applied_patches"
ROLLED_BACK_LOG = "dgm:rolled_back_traces"

PATCH_RATE_LIMIT_SECONDS = int(os.getenv("DGM_PATCH_RATE_LIMIT_SECONDS", "3600"))
LOOP_WAIT_S = 1.0
MAX_MUTATIONS_PER_LOOP = 3
ROLLBACK_SLEEP_S = int(os.getenv("DGM_ROLLBACK_SLEEP_SECONDS", "300"))

PATCH_HISTORY_FILE = Path(__file__).resolve().parent.parent / "patch_history.json"
_last_patch_time: float = 0.0
if PATCH_HISTORY_FILE.exists():  # pragma: no cover – startup rewind
    try:
        hist: list[dict[str, Any]] = json.loads(PATCH_HISTORY_FILE.read_text())
        if hist:
            _last_patch_time = hist[-1].get("timestamp", 0.0)
    except Exception as exc:  # pragma: no cover
        log.error("Failed to read patch history: %s", exc)

_MUTATION_NAME = os.getenv("DGM_MUTATION", "ASTInsertComment")

# ───────────────────────────── mutation helpers ──────────────────────────────
def _load_mutation() -> MutationStrategy:
    mod = importlib.import_module("dgm_kernel.mutation_strategies")
    cls = cast(type[MutationStrategy], getattr(mod, _MUTATION_NAME))
    return cls()


_mutation_strategy = _load_mutation()


def _mutate_code(code: str) -> str:
    """Synchronous helper for tests and internal use."""
    return _mutation_strategy.mutate(code)


async def _generate_patch_async(
    traces: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Async wrapper so property-tests can await patch generation."""
    return await generate_patch(traces)


# Back-compat alias used by a few tests
_generate_patch = _mutate_code

# ──────────────────────────── trace acquisition ─────────────────────────────
async def fetch_recent_traces(n: int = 100) -> list[dict[str, Any]]:
    """Pop the newest *n* traces, logging decode errors and returning list."""
    traces: list[dict[str, Any]] = []
    try:
        raw: list[str] = []
        for _ in range(n):
            item = REDIS.rpop(TRACE_QUEUE)
            if item is None:
                break
            raw.append(item)

        if not raw:
            return []

        for idx, txt in enumerate(raw):
            try:
                traces.append(json.loads(txt))
            except json.JSONDecodeError as exc:
                log.error(
                    "Failed to decode JSON for trace: %s. Error: %s. Trace index: %s",
                    txt,
                    exc,
                    idx,
                )

        if traces:
            log.info(
                "Fetched %s traces successfully, encountered %s decoding errors.",
                len(traces),
                len(raw) - len(traces),
            )
        elif raw:
            log.warning(
                "Fetched %s raw traces, but all failed to decode.",
                len(raw),
            )
    except redis.exceptions.RedisError as exc:
        log.error("Redis error while fetching traces: %s", exc)
        return []
    except Exception as exc:  # pragma: no cover - unexpected edge cases
        log.error("Unexpected error in fetch_recent_traces: %s", exc)
        return []

    return traces


# ───────────────────────────── patch generation ─────────────────────────────
async def generate_patch(
    traces: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Produce a JSON patch dict with keys ``target``, ``before``, ``after``.
    Returns *None* on failure so the caller can try another mutation.
    """
    patch = draft_patch(traces)
    if patch is None:
        metrics.increment_patch_generation(mutation=_MUTATION_NAME, result="failure")
        return None

    patch["after"] = _mutate_code(patch.get("after", ""))
    metrics.increment_patch_generation(mutation=_MUTATION_NAME, result="success")
    return patch


# ───────────────────────────── verification helpers ────────────────────────
def _get_pylint_score(code: str) -> float:
    return _prover_pylint_score(code)


async def _lint_with_ruff(code: str) -> bool:
    """Return *True* if `ruff check` passes on *code*."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    try:
        tmp.write(code)
        tmp.close()
        proc = await asyncio.create_subprocess_exec(
            "ruff", "check", tmp.name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
    finally:
        Path(tmp.name).unlink(missing_ok=True)


async def _run_unit_tests(target: str, code: str) -> bool:
    """Write *code* to *target*, run the ‘sanity_only’ shard, restore file."""
    tgt = Path(target)
    if not tgt.exists():
        return False

    original = tgt.read_text()
    tgt.write_text(code)
    try:
        proc = await asyncio.create_subprocess_exec(
            "pytest",
            "-q",
            "tests/dgm_kernel_tests/test_meta_loop.py::sanity_only",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
    finally:
        tgt.write_text(original)


async def _verify_patch(traces: list[dict[str, Any]], patch: dict[str, Any]) -> bool:
    """Danger-string, ruff, shard-tests; returns *True* if patch passes."""
    code = patch.get("after", "")
    if not code:
        return False

    banned = ("os.system", "eval(", "exec(", "subprocess.Popen", "subprocess.call")
    if any(tok in code for tok in banned):
        metrics.unsafe_token_found_total.inc()
        return False

    if not await _lint_with_ruff(code):
        return False

    tgt = patch.get("target")
    return tgt is not None and await _run_unit_tests(tgt, code)


# ───────────────────────────── apply / rollback ────────────────────────────
def _apply_patch(patch: dict[str, Any]) -> bool:
    tgt = Path(patch["target"])
    success = True
    try:
        tgt.parent.mkdir(parents=True, exist_ok=True)
        tgt.write_text(patch["after"])
        importlib.invalidate_caches()
        module_name = str(tgt.with_suffix("")).replace("/", ".").lstrip(".")
        try:
            module = importlib.import_module(module_name)
            importlib.reload(module)
        except ModuleNotFoundError:
            log.warning(
                "Module %s not found directly, attempting to load spec.", module_name
            )
            spec = importlib.util.spec_from_file_location(module_name, str(tgt))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                raise ImportError(f"Could not load spec for {module_name} from {tgt}")
    except Exception as exc:
        log.error("apply_patch failed: %s", exc)
        success = False

    metrics.increment_patch_apply(mutation=_MUTATION_NAME, result="success" if success else "failure")
    return success


def _rollback(patch: dict[str, Any]) -> None:
    tgt = Path(patch["target"])
    tgt.write_text(patch["before"])
    importlib.invalidate_caches()
    try:
        mod = importlib.import_module(str(tgt.with_suffix("")).replace("/", "."))
        importlib.reload(mod)
    except Exception as exc:
        log.error("rollback reload error: %s", exc)


def _record_patch_history(entry: dict[str, Any]) -> None:
    try:
        history: list[dict[str, Any]] = []
        if PATCH_HISTORY_FILE.exists():
            history = json.loads(PATCH_HISTORY_FILE.read_text())
        history.append(entry)
        PATCH_HISTORY_FILE.write_text(json.dumps(history, indent=2))
    except Exception as exc:  # pragma: no cover
        log.error("record history failed: %s", exc)


# ───────────────────────────── one-shot & forever loops ─────────────────────
async def loop_once() -> None:
    traces = await fetch_recent_traces()
    if not traces:
        return

    patch = await generate_patch(traces)
    if not patch or not await _verify_patch(traces, patch):
        return

    diff = "\n".join(
        difflib.unified_diff(
            patch["before"].splitlines(),
            patch["after"].splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )
    if prove_patch(diff) < 0.8:
        return

    ok, _, exit_code = run_patch_in_sandbox(patch)
    if not ok:
        return

    if time.time() - _last_patch_time < PATCH_RATE_LIMIT_SECONDS:
        return

    _apply_patch(patch)
    _record_patch_history(
        {
            "patch_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "diff": diff,
            "reward": 0.0,
            "sandbox_exit_code": exit_code,
        }
    )


async def meta_loop() -> None:
    while True:
        await loop_once()
        await asyncio.sleep(LOOP_WAIT_S)


def loop_forever() -> None:
    pending: dict[str, Any] | None = None
    none_streak = 0
    rollback_streak = 0

    while True:
        traces = asyncio.run(fetch_recent_traces())

        if pending:
            if not asyncio.run(_verify_patch(traces, pending)):
                _rollback(pending)
                pending = None
                rollback_streak += 1
            else:
                rollback_streak = 0
        else:
            for _ in range(MAX_MUTATIONS_PER_LOOP):
                pending = asyncio.run(_generate_patch_async(traces))
                if pending is not None:
                    none_streak = 0
                    break
                none_streak += 1

            if none_streak >= 3:
                os.environ["DGM_MUTATION"] = "ASTInsertComment"

        if rollback_streak >= 4:
            metrics.rollback_backoff_total.inc()
            os.environ["DGM_MUTATION"] = "ASTInsertComment"
            rollback_streak = 0
            time.sleep(ROLLBACK_SLEEP_S)
        else:
            time.sleep(LOOP_WAIT_S)


# ───────────────────────────── CLI entry-point ──────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="DGM Kernel Meta-Loop")
    parser.add_argument("--once", action="store_true", help="run exactly one iteration")
    args = parser.parse_args()

    if args.once:
        asyncio.run(loop_once())
    else:
        asyncio.run(meta_loop())
