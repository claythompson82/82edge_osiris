# dgm_kernel/meta_loop.py
"""
Darwin Gödel Machine – self-improvement meta-loop (async capable).
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
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import redis  # runtime Redis client

from dgm_kernel.llm_client import draft_patch
from dgm_kernel.mutation_strategies import MutationStrategy
from dgm_kernel.prover import prove_patch, _get_pylint_score as _prover_pylint_score
from dgm_kernel.sandbox import run_patch_in_sandbox
from llm_sidecar.reward import proofable_reward

log = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Redis keys
# ────────────────────────────────────────────────────────────────────────────
REDIS = redis.Redis(host="redis", port=6379, decode_responses=True)
PATCH_QUEUE = "dgm:patch_queue"
TRACE_QUEUE = "dgm:recent_traces"
APPLIED_LOG = "dgm:applied_patches"
ROLLED_BACK_LOG = "dgm:rolled_back_traces"

# ────────────────────────────────────────────────────────────────────────────
# Loop / rate-limit knobs
# ────────────────────────────────────────────────────────────────────────────
PATCH_RATE_LIMIT_SECONDS = int(os.getenv("DGM_PATCH_RATE_LIMIT_SECONDS", 3600))
LOOP_WAIT_S: float = 1.0                       # sleep between loop_forever() ticks
MAX_MUTATIONS_PER_LOOP: int = 3               # when queue is empty

# ────────────────────────────────────────────────────────────────────────────
# Patch-history bookkeeping (used for rate-limit reset)
# ────────────────────────────────────────────────────────────────────────────
PATCH_HISTORY_FILE = (
    Path(__file__).resolve().parent.parent / "patch_history.json"
)
_last_patch_time: float = 0.0
if PATCH_HISTORY_FILE.exists():  # pragma: no cover (startup-only)
    try:
        history = json.loads(PATCH_HISTORY_FILE.read_text())
        if isinstance(history, list) and history:
            _last_patch_time = float(history[-1].get("timestamp", 0.0))
    except Exception as exc:  # pragma: no cover
        log.error("Failed to read patch history: %s", exc)

# ────────────────────────────────────────────────────────────────────────────
# Mutation strategy plug-in
# ────────────────────────────────────────────────────────────────────────────
_MUTATION_NAME = os.getenv("DGM_MUTATION", "ASTInsertComment")


def _load_mutation() -> MutationStrategy:
    mod = importlib.import_module("dgm_kernel.mutation_strategies")
    cls = cast(type[MutationStrategy], getattr(mod, _MUTATION_NAME))
    return cls()


_mutation_strategy: MutationStrategy = _load_mutation()


# ---------------------------------------------------------------------------#
#                               Helper routines
# ---------------------------------------------------------------------------#
def _get_pylint_score(patch_code: str) -> float:
    """Proxy to prover internals (easier to monkey-patch in tests)."""
    return _prover_pylint_score(patch_code)


async def _lint_with_ruff(code: str) -> bool:
    """Run ruff on *code* and return True iff lint passes."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    try:
        tmp.write(code)
        tmp.close()
        proc = await asyncio.create_subprocess_exec(
            "ruff",
            "check",
            tmp.name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
    except FileNotFoundError:
        log.error("ruff not installed")
        return False
    finally:
        Path(tmp.name).unlink(missing_ok=True)


async def _run_unit_tests(target: str, code: str) -> bool:
    """Temporarily patch *target* on disk and run `pytest -q`."""
    tgt = Path(target)
    if not tgt.exists():
        log.error("Target %s does not exist", target)
        return False

    original = tgt.read_text()
    tgt.write_text(code)
    try:
        proc = await asyncio.create_subprocess_exec(
            "pytest",
            "-q",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
    finally:
        tgt.write_text(original)


# ---------------------------------------------------------------------------#
#                           Core meta-loop helpers
# ---------------------------------------------------------------------------#
async def fetch_recent_traces(n: int = 100) -> List[Dict[str, Any]]:
    """Pop *n* recent traces from Redis; tolerate JSON decode errors."""
    out: List[Dict[str, Any]] = []
    try:
        raw: List[str] = []
        for _ in range(n):
            j = REDIS.rpop(TRACE_QUEUE)
            if j is None:
                break
            raw.append(j)

        for idx, j in enumerate(raw):
            try:
                out.append(json.loads(j))
            except json.JSONDecodeError as exc:
                log.error("Bad trace[%d]: %s (%s)", idx, j[:120], exc)

    except redis.exceptions.RedisError as exc:
        log.error("Redis error: %s", exc)

    return out


async def generate_patch(
    traces: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Call the LLM / heuristic search to emit a JSON patch.  Thin wrapper around
    `dgm_kernel.llm_client.draft_patch`, but also runs the mutation strategy on
    `after` so tests can stub easily.
    """
    patch = draft_patch(traces)
    if patch is None:
        return None
    patch["after"] = _mutation_strategy.mutate(patch.get("after", ""))
    return patch


# Compat wrapper used by unit-tests to monkey-patch generation easily.
async def _generate_patch(
    traces: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    return await generate_patch(traces)


async def _verify_patch(
    traces: List[Dict[str, Any]], patch: Dict[str, Any]
) -> bool:
    """Static safety checks, ruff lint, unit-tests, pylint, prover score."""
    patch_code = patch.get("after", "")
    if not patch_code:
        return False

    # quick string blacklist
    for tok in ("os.system", "eval(", "exec(", "subprocess.Popen", "subprocess.call"):
        if tok in patch_code:
            log.warning("Dangerous token %s in patch", tok)
            return False

    if not await _lint_with_ruff(patch_code):
        return False

    tgt = patch.get("target")
    if not tgt:
        log.error("Patch missing 'target'")
        return False

    if not await _run_unit_tests(tgt, patch_code):
        return False

    # pylint heuristic gate
    if _get_pylint_score(patch_code) < 6.0:
        return False

    # prover semantic check
    diff_text = "\n".join(
        difflib.unified_diff(
            patch.get("before", "").splitlines(),
            patch_code.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )
    if prove_patch(diff_text) < 0.8:
        return False

    return True


def _apply_patch(patch: Dict[str, Any]) -> bool:
    """Write *after* into *target* and hot-reload the module."""
    tgt = Path(patch["target"])
    tgt.parent.mkdir(parents=True, exist_ok=True)
    tgt.write_text(patch["after"])

    importlib.invalidate_caches()
    mod_name = str(tgt.with_suffix("")).replace("/", ".").lstrip(".")
    try:
        module = importlib.import_module(mod_name)
        importlib.reload(module)
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location(mod_name, tgt)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            log.error("Cannot load module %s", mod_name)
            return False
    return True


def _record_patch_history(entry: Dict[str, Any]) -> None:
    """Append *entry* (dict) to PATCH_HISTORY_FILE (JSON list)."""
    history: List[Dict[str, Any]] = []
    if PATCH_HISTORY_FILE.exists():
        try:
            history = json.loads(PATCH_HISTORY_FILE.read_text())
        except json.JSONDecodeError:
            log.error("Corrupted patch history; resetting.")
    history.append(entry)
    PATCH_HISTORY_FILE.write_text(json.dumps(history, indent=2))


# ---------------------------------------------------------------------------#
#                       Single-shot and continuous loops
# ---------------------------------------------------------------------------#
async def loop_once() -> None:
    """Run one iteration – fetch traces → propose patch → verify → sandbox → apply."""
    global _last_patch_time

    traces = await fetch_recent_traces()
    if not traces:
        return

    patch = await generate_patch(traces)
    if not patch or not await _verify_patch(traces, patch):
        return

    ok, logs, exit_code = run_patch_in_sandbox(patch)
    if not ok:
        log.warning("Sandbox failed (exit %s)\n%s", exit_code, logs)
        return

    now = time.time()
    if now - _last_patch_time < PATCH_RATE_LIMIT_SECONDS:
        return

    if _apply_patch(patch):
        reward = sum(proofable_reward(t, patch["after"]) for t in traces)
        diff = "\n".join(
            difflib.unified_diff(
                patch["before"].splitlines(),
                patch["after"].splitlines(),
                fromfile="before",
                tofile="after",
                lineterm="",
            )
        )
        patch_id = str(uuid.uuid4())
        REDIS.lpush(APPLIED_LOG, json.dumps(patch | {"patch_id": patch_id, "reward": reward}))
        _record_patch_history(
            {
                "patch_id": patch_id,
                "timestamp": now,
                "diff": diff,
                "reward": reward,
                "sandbox_exit_code": exit_code,
            }
        )
        _last_patch_time = now
    else:
        log.error("Failed to apply patch to %s", patch["target"])


async def meta_loop() -> None:
    """Continuous supervisor loop (runs forever)."""
    while True:
        await loop_once()
        await asyncio.sleep(5)


# ---------------------------------------------------------------------------#
#                         Legacy synchronous test-loop
# ---------------------------------------------------------------------------#
def _rollback(patch: Dict[str, Any]) -> None:
    """Simple rollback helper for old unit-tests."""
    tgt = Path(patch["target"])
    tgt.write_text(patch["before"])
    importlib.invalidate_caches()
    try:
        module = importlib.import_module(str(tgt.with_suffix("")).replace("/", ".").lstrip("."))
        importlib.reload(module)
    except Exception as exc:
        log.error("Rollback reload failed: %s", exc)


def loop_forever() -> None:  # kept for existing tests
    pending: Optional[Dict[str, Any]] = None
    while True:
        traces = asyncio.run(fetch_recent_traces())
        if pending:
            if not asyncio.run(_verify_patch(traces, pending)):
                _rollback(pending)
                pending = None
        else:
            for _ in range(MAX_MUTATIONS_PER_LOOP):
                pending = asyncio.run(_generate_patch(traces))
        time.sleep(LOOP_WAIT_S)


# ---------------------------------------------------------------------------#
#                              CLI entry-point
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser("DGM Kernel Meta-Loop")
    parser.add_argument("--once", action="store_true", help="Run exactly one iteration.")
    args = parser.parse_args()
    if args.once:
        asyncio.run(loop_once())
    else:
        asyncio.run(meta_loop())

