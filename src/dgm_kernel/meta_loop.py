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
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import redis

from dgm_kernel import metrics
from dgm_kernel.crypto_sign import sign_patch, verify_patch
from dgm_kernel.llm_client import PatchDict, draft_patch
from dgm_kernel.mutation_scheduler import MutationScheduler
from dgm_kernel.mutation_strategies import (
    ASTInsertComment,
    ASTRenameIdentifier,
    MutationStrategy,
)
from dgm_kernel.otel import tracer
from dgm_kernel.prover import _get_pylint_score as _prover_pylint_score
from dgm_kernel.prover import prove_patch
from dgm_kernel.sandbox import run_patch_in_sandbox
from dgm_kernel.trace_schema import HistoryEntry, validate_traces
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
        hist: list[HistoryEntry] = json.loads(PATCH_HISTORY_FILE.read_text())
        if hist:
            _last_patch_time = hist[-1].get("timestamp", 0.0)
    except Exception as exc:  # pragma: no cover
        log.error("Failed to read patch history: %s", exc)

_MUTATION_NAME = os.getenv("DGM_MUTATION", "ASTInsertComment")
_scheduler = MutationScheduler()
_recent_rewards: deque[float] = deque(maxlen=5)


# ───────────────────────────── mutation helpers ──────────────────────────────
def _load_mutation() -> MutationStrategy:
    mod = importlib.import_module("dgm_kernel.mutation_strategies")
    env_name = os.getenv("DGM_MUTATION", "ASTInsertComment")

    if env_name == "auto":
        candidates: list[MutationStrategy] = []
        for attr in dir(mod):
            cls = getattr(mod, attr)
            if isinstance(cls, type) and hasattr(cls, "mutate") and hasattr(cls, "name"):
                try:
                    candidates.append(cast(MutationStrategy, cls()))
                except Exception:
                    pass
        chosen = mod.weighted_choice(candidates)
        globals()["_MUTATION_NAME"] = chosen.name
        return chosen

    if not hasattr(mod, env_name) and env_name == "ASTInsertCommentAndRename":

        class ASTInsertCommentAndRename(MutationStrategy):
            name = "ASTInsertCommentAndRename"

            def mutate(self, code: str) -> str:
                code = ASTInsertComment().mutate(code)
                return ASTRenameIdentifier().mutate(code)

        setattr(mod, "ASTInsertCommentAndRename", ASTInsertCommentAndRename)

    cls = cast(type[MutationStrategy], getattr(mod, env_name))
    globals()["_MUTATION_NAME"] = env_name
    return cls()


_mutation_strategy: MutationStrategy = _load_mutation()


def _set_mutation_strategy(name: str) -> None:
    """Update global mutation strategy and mirror env var."""
    global _MUTATION_NAME, _mutation_strategy
    _MUTATION_NAME = name
    os.environ["DGM_MUTATION"] = name
    _mutation_strategy = _load_mutation()
    # Safely re-initialize the scheduler
    if TYPE_CHECKING:
        assert isinstance(_scheduler, MutationScheduler)
    _scheduler.__init__()


def _mutate_code(code: str) -> str:
    """Synchronous helper for tests and internal use."""
    return _mutation_strategy.mutate(code)


async def _generate_patch_async(traces: list[dict[str, Any]]) -> PatchDict | None:
    """Async wrapper so property-tests can await patch generation."""
    return await generate_patch(traces)


# Back-compat alias used by a few tests
_generate_patch = _mutate_code


# ──────────────────────────── trace acquisition ─────────────────────────────
async def fetch_recent_traces(n: int = 100) -> list[dict[str, Any]]:
    """Pop the newest *n* traces, logging decode errors and returning list of dicts."""
    traces_raw: list[dict[str, Any]] = []
    try:
        raw_items: list[str] = []
        for _ in range(n):
            item = REDIS.rpop(TRACE_QUEUE)
            if item is None:
                break
            raw_items.append(item)

        if not raw_items:
            return []

        for idx, txt in enumerate(raw_items):
            try:
                traces_raw.append(json.loads(txt))
            except json.JSONDecodeError as exc:
                log.error(
                    "Failed to decode JSON for trace: %s. Error: %s. Trace index: %s",
                    txt,
                    exc,
                    idx,
                )
        decoded_count = len(traces_raw)
        traces = validate_traces(traces_raw)
        trace_dicts = [t.to_dict() for t in traces]

        if trace_dicts:
            log.info(
                "Fetched %s traces successfully as dicts, encountered %s decoding errors.",
                len(trace_dicts),
                len(raw_items) - decoded_count,
            )
        elif raw_items:
            log.warning(
                "Fetched %s raw traces, but all failed to decode.",
                len(raw_items),
            )
        return trace_dicts
    except redis.exceptions.RedisError as exc:
        log.error("Redis error while fetching traces: %s", exc)
        return []
    except Exception as exc:  # pragma: no cover - unexpected edge cases
        log.error("Unexpected error in fetch_recent_traces: %s", exc)
        return []


# ───────────────────────────── patch generation ─────────────────────────────
async def generate_patch(traces: list[dict[str, Any]]) -> PatchDict | None:
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
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(code)

    try:
        proc = await asyncio.create_subprocess_exec(
            "ruff", "check", str(tmp_path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        return proc.returncode == 0
    finally:
        tmp_path.unlink(missing_ok=True)


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


async def _verify_patch(traces: list[dict[str, Any]], patch: PatchDict) -> bool:
    """Danger-string, ruff, shard-tests; returns *True* if patch passes."""
    code = patch.get("after", "")
    if not code:
        return False

    banned = ("os.system", "eval(", "exec(", "subprocess.Popen", "subprocess.call")
    if any(tok in code for tok in banned):
        metrics.unsafe_token_found_total.inc()
        return False

    if "sig" in patch:
        diff = "\n".join(
            difflib.unified_diff(
                patch.get("before", "").splitlines(),
                patch.get("after", "").splitlines(),
                fromfile="before",
                tofile="after",
                lineterm="",
            )
        )
        if not verify_patch(diff, patch["sig"]):
            metrics.patch_sig_invalid_total.inc()
            return False

    if not await _lint_with_ruff(code):
        return False

    tgt = patch.get("target")
    return tgt is not None and await _run_unit_tests(tgt, code)


# ───────────────────────────── apply / rollback ────────────────────────────
def _apply_patch(patch: PatchDict) -> bool:
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
            log.warning("Module %s not found directly, attempting to load spec.", module_name)
            spec = importlib.util.spec_from_file_location(module_name, str(tgt))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                raise ImportError(f"Could not load spec for {module_name} from {tgt}")
    except Exception as exc:
        log.error("apply_patch failed: %s", exc)
        success = False

    metrics.increment_patch_apply(
        mutation=_MUTATION_NAME, result="success" if success else "failure"
    )
    if success:
        metrics.increment_mutation_success(strategy=_MUTATION_NAME)
    else:
        metrics.increment_mutation_failure(strategy=_MUTATION_NAME)
    return success


def _rollback(patch: PatchDict) -> None:
    tgt = Path(patch["target"])
    tgt.write_text(patch["before"])
    importlib.invalidate_caches()
    try:
        mod = importlib.import_module(str(tgt.with_suffix("")).replace("/", "."))
        importlib.reload(mod)
    except Exception as exc:
        log.error("rollback reload error: %s", exc)


def _record_patch_history(entry: HistoryEntry) -> None:
    try:
        history: list[HistoryEntry] = []
        if PATCH_HISTORY_FILE.exists():
            history = json.loads(PATCH_HISTORY_FILE.read_text())
        if "diff" in entry and "sig" not in entry:
            try:
                entry["sig"] = sign_patch(entry["diff"])
            except Exception as exc:  # pragma: no cover - signing issues
                log.error("sign diff failed: %s", exc)
        history.append(entry)
        PATCH_HISTORY_FILE.write_text(json.dumps(history, indent=2))
        # Update patch application rate metric
        try:
            timestamps = [h.get("timestamp", 0.0) for h in history]
            if len(timestamps) >= 11:
                delta = timestamps[-1] - timestamps[-11]
                metrics.patch_apply_minutes_average.set(delta / 10 / 60)
        except Exception as exc:  # pragma: no cover - metric update issues
            log.error("patch rate metric update failed: %s", exc)
    except Exception as exc:  # pragma: no cover
        log.error("record history failed: %s", exc)


# ───────────────────────────── one-shot & forever loops ─────────────────────
async def loop_once() -> None:
    with tracer.start_as_current_span("meta_loop.iteration"):
        with tracer.start_as_current_span("fetch_traces"):
            traces = await fetch_recent_traces()
        if not traces:
            return

        with tracer.start_as_current_span("generate_patch"):
            patch = await generate_patch(traces)
        with tracer.start_as_current_span("verify_patch"):
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

        with tracer.start_as_current_span("sandbox"):
            # Unpack only the values you need
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
    pending: PatchDict | None = None
    none_streak = 0
    rollback_streak = 0

    while True:
        traces = asyncio.run(fetch_recent_traces())

        avg_reward = (
            sum(_recent_rewards) / len(_recent_rewards) if _recent_rewards else 0.0
        )
        chosen = _scheduler.next_strategy(avg_reward)
        if chosen != _MUTATION_NAME:
            _set_mutation_strategy(chosen)

        if pending:
            if not asyncio.run(_verify_patch(traces, pending)):
                _rollback(pending)
                pending = None
                _recent_rewards.append(-1.0)
                rollback_streak += 1
            else:
                _recent_rewards.append(1.0)
                rollback_streak = 0
        else:
            for _ in range(MAX_MUTATIONS_PER_LOOP):
                pending = asyncio.run(_generate_patch_async(traces))
                if pending is not None:
                    none_streak = 0
                    break
                none_streak += 1

            if none_streak >= 3:
                _set_mutation_strategy("ASTInsertComment")

        if rollback_streak >= 4:
            metrics.rollback_backoff_total.inc()
            _set_mutation_strategy("ASTInsertComment")
            rollback_streak = 0
            time.sleep(ROLLBACK_SLEEP_S)
        else:
            time.sleep(LOOP_WAIT_S)


# ───────────────────────────── CLI entry-point ──────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="DGM Kernel Meta-Loop")
    parser.add_argument(
        "--once", action="store_true", help="run exactly one iteration"
    )
    args = parser.parse_args()

    if args.once:
        asyncio.run(loop_once())
    else:
        loop_forever()
