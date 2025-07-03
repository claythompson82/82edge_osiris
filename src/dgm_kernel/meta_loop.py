# dgm_kernel/meta_loop.py
"""
Darwin Gödel Machine – self-improvement meta-loop (async capable)
"""

from __future__ import annotations
import asyncio
import importlib
import importlib.util
import logging
import time
import json
import argparse
import subprocess
import tempfile
import os
import uuid
import difflib
import redis  # Changed from 'from redis import Redis'
from pathlib import Path
from typing import Any, Dict, List, cast

# from redis import Redis # Original import
from llm_sidecar.reward import proofable_reward
from dgm_kernel.llm_client import draft_patch  # Added dgm_kernel.llm_client import
from dgm_kernel.prover import (
    prove_patch,
    _get_pylint_score as _prover_pylint_score,
)
from dgm_kernel.sandbox import run_patch_in_sandbox
from dgm_kernel.mutation_strategies import MutationStrategy

log = logging.getLogger(__name__)

REDIS = redis.Redis(
    host="redis", port=6379, decode_responses=True
)  # Updated instantiation
PATCH_QUEUE = "dgm:patch_queue"  # ← incoming JSON patches
TRACE_QUEUE = "dgm:recent_traces"  # ← recent trading traces (jsonl)
APPLIED_LOG = "dgm:applied_patches"  # ← audit trail
ROLLED_BACK_LOG = "dgm:rolled_back_traces"  # ← traces that triggered rollback

# Rate limiting configuration (seconds between successful patches)
PATCH_RATE_LIMIT_SECONDS = int(os.environ.get("DGM_PATCH_RATE_LIMIT_SECONDS", 3600))

# How long to sleep between iterations of loop_forever()
LOOP_WAIT_S = 1.0

# Maximum attempts to mutate when no patch is pending
MAX_MUTATIONS_PER_LOOP = 3

# History file tracking applied patches
PATCH_HISTORY_FILE = Path(__file__).resolve().parent.parent / "patch_history.json"

# Initialize last patch time from history if available
_last_patch_time = 0.0
if PATCH_HISTORY_FILE.exists():  # pragma: no cover - startup state
    try:  # pragma: no cover
        history = json.loads(PATCH_HISTORY_FILE.read_text())  # pragma: no cover
        if isinstance(history, list) and history:  # pragma: no cover
            _last_patch_time = history[-1].get("timestamp", 0.0)  # pragma: no cover
    except Exception as e:  # pragma: no cover - history shouldn't crash startup
        log.error(f"Failed to read patch history: {e}")  # pragma: no cover

_MUTATION_NAME = os.environ.get("DGM_MUTATION", "ASTInsertComment")


def _load_mutation() -> MutationStrategy:
    mod = importlib.import_module("dgm_kernel.mutation_strategies")
    cls = cast(type[MutationStrategy], getattr(mod, _MUTATION_NAME))
    return cls()


_mutation_strategy = _load_mutation()


def _mutate_code(code: str) -> str:
    return _mutation_strategy.mutate(code)


async def _generate_patch_async(traces: List[Dict[str, Any]]) -> Dict[str, Any] | None:  # pragma: no cover - thin wrapper
    """Wrapper for generate_patch so tests can monkey-patch easier."""
    return await generate_patch(traces)


def _generate_patch(arg: Any) -> Any:
    if isinstance(arg, str):
        return _mutate_code(arg)
    res = _generate_patch_async(arg)
    if asyncio.iscoroutine(res):
        return asyncio.run(res)
    return res

# ────────────────────────────────────────────────────────────────────────────
# ▼ 1.  fetch_recent_traces()  (pull N traces from Redis)
# ▼ 2.  generate_patch()       (call LLM / α-search to propose code diff)
# ▼ 3.  apply_patch()          (load diff, patch in-place, hot-reload module)
# ▼ 4.  rollback_if_bad()      (revert if reward ↓ or errors ↑)
# ────────────────────────────────────────────────────────────────────────────


async def fetch_recent_traces(n: int = 100) -> List[Dict[str, Any]]:  # pragma: no cover - network access
    """Pop the newest N traces for evaluation. Handles JSON decoding errors."""
    traces = []
    try:
        # Efficiently get N items using pipeline or Lua script if supported for async,
        # or loop rpop if not. For simplicity with sync Redis client in async context,
        # direct rpop loop is used here. Consider redis-py's async client for true async.
        raw_traces = []
        for _ in range(n):
            j = REDIS.rpop(TRACE_QUEUE)
            if j is None:  # rpop returns None if the list is empty
                break
            raw_traces.append(j)

        if not raw_traces:
            return []  # Return empty list if no traces were fetched

        for i, trace_str in enumerate(raw_traces):
            try:
                traces.append(json.loads(trace_str))
            except json.JSONDecodeError as e:
                log.error(
                    f"Failed to decode JSON for trace: {trace_str}. Error: {e}. Trace index: {i}"
                )
                # Optionally, could add malformed trace to a separate Redis list for inspection
                # REDIS.lpush("dgm:malformed_traces", trace_str)

        if traces:  # Log only if some traces were successfully parsed
            log.info(
                f"Fetched {len(traces)} traces successfully, encountered {len(raw_traces) - len(traces)} decoding errors."
            )
        elif raw_traces:  # Log if raw traces were fetched but none could be parsed
            log.warning(
                f"Fetched {len(raw_traces)} raw traces, but all failed to decode."
            )
        # If raw_traces is empty, nothing is logged by these conditions, which is fine.

    except redis.exceptions.RedisError as e:
        log.error(f"Redis error while fetching traces: {e}")
        # Depending on the severity, could raise or return empty list
        return []  # Or handle more gracefully
    except Exception as e:  # Catch any other unexpected errors
        log.error(f"Unexpected error in fetch_recent_traces: {e}")
        return []

    return traces


async def generate_patch(  # pragma: no cover - external LLM
    traces: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """
    Emit a JSON patch using an LLM or search routine.
    TODO(https://github.com/82edge/osiris/issues/99):
        Replace with full patch generation logic.
    """
    patch = draft_patch(traces)
    if patch is None:
        return None
    patch["after"] = _mutate_code(patch.get("after", ""))
    return patch




def _get_pylint_score(patch_code: str) -> float:  # pragma: no cover - thin shim
    """Proxy to prover._get_pylint_score for easier patching in tests."""
    return _prover_pylint_score(patch_code)


async def _lint_with_ruff(code: str) -> bool:  # pragma: no cover - integration
    """Run ruff on the provided code string and return True if it passes."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
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
        log.error("ruff command not found")
        return False
    finally:
        Path(tmp.name).unlink(missing_ok=True)


async def _run_unit_tests(target: str, code: str) -> bool:  # pragma: no cover - slow path
    """Temporarily apply the patch and run pytest to ensure tests pass."""
    tgt = Path(target)
    if not tgt.exists():
        log.error("Target file %s does not exist", target)
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
    except FileNotFoundError:
        log.error("pytest command not found")
        return False
    finally:
        tgt.write_text(original)


async def _verify_patch(traces: List[Dict[str, Any]], patch: Dict[str, Any]) -> bool:  # pragma: no cover - heavy validation
    """Validate the patch for dangerous code, lint errors, and failing tests."""
    patch_code = patch.get("after", "")
    if not patch_code:
        log.error("_verify_patch called with patch containing no 'after' code.")
        return False

    dangerous_tokens = [
        "os.system",
        "eval(",
        "exec(",
        "subprocess.Popen",
        "subprocess.call",
    ]
    for tok in dangerous_tokens:
        if tok in patch_code:
            log.warning("Patch contains disallowed pattern: %s", tok)
            return False

    if not await _lint_with_ruff(patch_code):
        return False

    target = patch.get("target")
    if not target:
        log.error("Patch missing target path")
        return False

    if not await _run_unit_tests(target, patch_code):
        return False

    return True


def _apply_patch(patch: Dict[str, Any]) -> bool:  # pragma: no cover - IO heavy
    """
    Atomically write patch['after'] into patch['target'] on disk,
    then `importlib.reload()` the module in-memory.
    Return True on success.
    """
    tgt = Path(patch["target"])
    # Create parent directories if they don't exist
    tgt.parent.mkdir(parents=True, exist_ok=True)
    tgt.write_text(patch["after"])
    importlib.invalidate_caches()
    # Ensure the module path is in a format importlib can use
    # e.g., osiris_policy.strategy if target is osiris_policy/strategy.py
    module_name = str(tgt.with_suffix("")).replace("/", ".").lstrip(".")
    try:
        module = importlib.import_module(module_name)
        importlib.reload(module)
    except ModuleNotFoundError:
        # This can happen if the module is being created for the first time
        # or if it's not in a package.
        # Attempt to load it based on its path directly if it's a new top-level module
        # This part can be tricky and might need adjustment based on project structure
        log.warning(
            f"Module {module_name} not found directly, attempting to load spec."
        )
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(tgt))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                # This doesn't reload it into sys.modules in a standard way for subsequent reloads
                # but makes its code run. For DGM, this might be enough for a first patch.
            else:
                raise ImportError(f"Could not load spec for {module_name} from {tgt}")
        except Exception as e:
            log.error(f"Error loading module {module_name} after writing patch: {e}")
            return False

    return True


def _record_patch_history(entry: Dict[str, Any]) -> None:  # pragma: no cover - disk log
    """Append a patch entry to PATCH_HISTORY_FILE in a JSON list."""
    history = []
    if PATCH_HISTORY_FILE.exists():
        try:
            history = json.loads(PATCH_HISTORY_FILE.read_text())
        except json.JSONDecodeError:
            log.error("Corrupted patch history; resetting file")
            history = []
    history.append(entry)
    PATCH_HISTORY_FILE.write_text(json.dumps(history, indent=2))


async def loop_once() -> None:  # pragma: no cover - CLI helper
    """Run a single iteration of the meta-loop."""
    global _last_patch_time
    traces = await fetch_recent_traces()
    if not traces:
        log.info("No traces found, exiting.")
        return

    patch = await generate_patch(traces)
    if not patch:
        log.info("No patch generated, exiting.")
        return

    accepted = await _verify_patch(traces, patch)
    if not accepted:
        log.warning(
            f"Patch for {patch.get('target')} was not approved, exiting."
        )
        return

    diff_text = "".join(
        difflib.unified_diff(
            patch.get("before", "").splitlines(),
            patch.get("after", "").splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )
    score = prove_patch(diff_text)
    if score < 0.8:
        log.warning("Prover score %.2f below threshold", score)
        return

    sandbox_ok, sandbox_logs, exit_code = run_patch_in_sandbox(patch)
    if not sandbox_ok:
        log.warning("Patch failed sandbox test (exit code %s).", exit_code)
        log.debug("Sandbox output:\n%s", sandbox_logs)
        return

    now = time.time()
    if now - _last_patch_time < PATCH_RATE_LIMIT_SECONDS:
        log.info("Rate limit active, skipping patch application.")
        return

    if _apply_patch(patch):
        new_r = sum(
            proofable_reward(t, patch.get("after")) for t in traces
        )
        if new_r >= 0:
            patch_id = str(uuid.uuid4())
            diff = "".join(
                difflib.unified_diff(
                    patch.get("before", "").splitlines(),
                    patch.get("after", "").splitlines(),
                    fromfile="before",
                    tofile="after",
                    lineterm="",
                )
            )
            REDIS.lpush(
                APPLIED_LOG,
                json.dumps(patch | {"reward": new_r, "patch_id": patch_id}),
            )
            _record_patch_history(
                {
                    "patch_id": patch_id,
                    "timestamp": now,
                    "diff": diff,
                    "reward": new_r,
                    "sandbox_exit_code": exit_code,
                }
            )
            log.info("Patch applied ✔️  reward=%.4f", new_r)
            _last_patch_time = now
        else:
            log.warning("Patch rolled back, reward=%.4f", new_r)
            _rollback(patch)
            for t in traces:
                t["rolled_back"] = True
                REDIS.lpush(ROLLED_BACK_LOG, json.dumps(t))
    else:
        log.error(f"Failed to apply patch for target: {patch.get('target')}")


async def meta_loop() -> None:  # pragma: no cover - production loop
    """Run the main async supervisor loop (runs forever)."""
    global _last_patch_time
    while True:
        traces = await fetch_recent_traces()
        if not traces:
            await asyncio.sleep(5)
            continue

        patch = await generate_patch(traces)
        if not patch:
            log.info("No patch generated, continuing.")
            continue

        accepted = await _verify_patch(traces, patch)
        if not accepted:
            log.warning(
                f"Patch for {patch.get('target')} was not approved, skipping application."
            )
            # Optionally, add to a different Redis log for rejected patches
            # REDIS.lpush("dgm:rejected_patches", json.dumps(patch))
            continue

        diff_text = "".join(
            difflib.unified_diff(
                patch.get("before", "").splitlines(),
                patch.get("after", "").splitlines(),
                fromfile="before",
                tofile="after",
                lineterm="",
            )
        )
        score = prove_patch(diff_text)
        if score < 0.8:
            log.warning("Prover score %.2f below threshold", score)
            continue

        sandbox_ok, sandbox_logs, exit_code = run_patch_in_sandbox(patch)
        if not sandbox_ok:
            log.warning(
                "Patch failed sandbox test (exit code %s).", exit_code
            )
            log.debug("Sandbox output:\n%s", sandbox_logs)
            continue

        now = time.time()
        if now - _last_patch_time < PATCH_RATE_LIMIT_SECONDS:
            log.info("Rate limit active, skipping patch application.")
            continue

        if _apply_patch(patch):
            # Evaluate reward on the same trace batch
            new_r = sum(
                proofable_reward(t, patch.get("after")) for t in traces
            )  # Pass patch content to reward
            if new_r >= 0:  # simple non-regression gate for now
                patch_id = str(uuid.uuid4())
                diff = "".join(
                    difflib.unified_diff(
                        patch.get("before", "").splitlines(),
                        patch.get("after", "").splitlines(),
                        fromfile="before",
                        tofile="after",
                        lineterm="",
                    )
                )
                REDIS.lpush(
                    APPLIED_LOG,
                    json.dumps(patch | {"reward": new_r, "patch_id": patch_id}),
                )
                _record_patch_history(
                    {
                        "patch_id": patch_id,
                        "timestamp": now,
                        "diff": diff,
                        "reward": new_r,
                        "sandbox_exit_code": exit_code,
                    }
                )
                log.info(
                    "Patch applied ✔️  reward=%.4f",
                    new_r,
                )
                _last_patch_time = now
            else:
                log.warning("Patch rolled back, reward=%.4f", new_r)
                _rollback(patch)
                for t in traces:
                    t["rolled_back"] = True
                    REDIS.lpush(ROLLED_BACK_LOG, json.dumps(t))
        else:
            log.error(f"Failed to apply patch for target: {patch.get('target')}")
            # _rollback(patch) # Consider if rollback is safe if apply itself failed.


def _rollback(patch: Dict[str, Any]) -> None:  # pragma: no cover - simple file revert
    """Roll back the patch by writing the 'before' content to the target file and reloading the module."""
    tgt = Path(patch["target"])
    tgt.write_text(patch["before"])
    importlib.invalidate_caches()
    module_name = str(tgt.with_suffix("")).replace("/", ".").lstrip(".")
    try:
        module = importlib.import_module(module_name)
        importlib.reload(module)
        log.info(f"Rolled back {patch['target']} successfully.")
    except Exception as e:
        # Log error during rollback, but proceed as the file is reverted.
        log.error(f"Error reloading module {module_name} during rollback: {e}")


def loop_forever() -> None:  # pragma: no cover - background service
    """Self-healing loop running indefinitely."""
    pending_patch: Dict[str, Any] | None = None
    while True:
        traces = asyncio.run(fetch_recent_traces())

        if pending_patch:
            log.info("Verifying pending patch")
            if not asyncio.run(_verify_patch(traces, pending_patch)):
                log.info("Patch failed verification, rolling back")
                _rollback(pending_patch)
                pending_patch = None
        else:
            log.info("No patch pending, generating mutation")
            for _ in range(MAX_MUTATIONS_PER_LOOP):
                pending_patch_candidate = _generate_patch(traces)
                if asyncio.iscoroutine(pending_patch_candidate):
                    pending_patch_candidate = asyncio.run(pending_patch_candidate)
                pending_patch = pending_patch_candidate

        time.sleep(LOOP_WAIT_S)


# Entrypoint for standalone container / CLI
if __name__ == "__main__":  # pragma: no cover - manual execution only
    logging.basicConfig(level=logging.INFO)  # pragma: no cover
    parser = argparse.ArgumentParser(description="DGM Kernel Meta-Loop")  # pragma: no cover
    parser.add_argument(  # pragma: no cover
        "--once", action="store_true", help="Run the meta-loop only once."  # pragma: no cover
    )  # pragma: no cover
    args = parser.parse_args()  # pragma: no cover

    if args.once:  # pragma: no cover
        log.info("Running DGM meta-loop once.")  # pragma: no cover
        asyncio.run(loop_once())  # pragma: no cover
        log.info("DGM meta-loop (once) finished.")  # pragma: no cover
    else:  # pragma: no cover
        log.info("Starting DGM meta-loop to run continuously.")  # pragma: no cover
        asyncio.run(meta_loop())  # pragma: no cover
