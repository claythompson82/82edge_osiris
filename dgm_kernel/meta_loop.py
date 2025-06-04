# dgm_kernel/meta_loop.py
"""
Darwin Gödel Machine – self-improvement meta-loop (async capable)

TODOs are marked ▼
"""

from __future__ import annotations
import asyncio, importlib, logging, time, json, argparse  # Added json import, Added argparse
import redis  # Changed from 'from redis import Redis'
from pathlib import Path

# from redis import Redis # Original import
from llm_sidecar.reward import proofable_reward
from dgm_kernel.llm_client import draft_patch  # Added dgm_kernel.llm_client import
from dgm_kernel.prover import prove_patch  # Added import

log = logging.getLogger(__name__)

REDIS = redis.Redis(
    host="redis", port=6379, decode_responses=True
)  # Updated instantiation
PATCH_QUEUE = "dgm:patch_queue"  # ← incoming JSON patches
TRACE_QUEUE = "dgm:recent_traces"  # ← recent trading traces (jsonl)
APPLIED_LOG = "dgm:applied_patches"  # ← audit trail
ROLLED_BACK_LOG = "dgm:rolled_back_traces"  # ← traces that triggered rollback

# ────────────────────────────────────────────────────────────────────────────
# ▼ 1.  fetch_recent_traces()  (pull N traces from Redis)
# ▼ 2.  generate_patch()       (call LLM / α-search to propose code diff)
# ▼ 3.  apply_patch()          (load diff, patch in-place, hot-reload module)
# ▼ 4.  rollback_if_bad()      (revert if reward ↓ or errors ↑)
# ────────────────────────────────────────────────────────────────────────────


async def fetch_recent_traces(n: int = 100) -> list[dict]:
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


async def generate_patch(
    traces: list[dict],
) -> dict | None:  # Return type updated to include None
    """
    Placeholder for LLM / search routine to emit a JSON patch.
    ▼ TODO: Replace with actual patch generation logic.
    """
    return draft_patch(traces)


async def _verify_patch(traces: list[dict], patch: dict) -> tuple[bool, float]:
    """
    Verifies the patch using the prover.
    Returns (is_accepted, pylint_score).
    """
    # patch_id is not strictly necessary for verification itself by prove_patch,
    # but good to have for context if prove_patch starts logging or using it.
    # Using a default if not present in the patch dict.
    patch_id = patch.get("id", "unknown_patch_id")
    patch_diff = patch.get(
        "diff", ""
    )  # Diff might not always be present or used by prove_patch
    patch_code = patch.get("after", "")

    if not patch_code:
        log.error("_verify_patch called with patch containing no 'after' code.")
        return False, 0.0  # Cannot verify empty code, score 0

    # Call the imported prove_patch function
    verification_result = prove_patch(
        id=patch_id, diff=patch_diff, patch_code=patch_code
    )

    accepted = verification_result.status == "APPROVED"
    pylint_score = verification_result.score  # This is the Pylint score from the prover

    log.info(
        f"Patch verification result: {'Accepted' if accepted else 'Rejected'}. Pylint Score: {pylint_score}"
    )

    return accepted, pylint_score


def _apply_patch(patch: dict) -> bool:
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
    module_name = str(tgt.with_suffix("")).replace("/", ".")
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


async def meta_loop():
    """Main async supervisor loop (runs forever)."""
    while True:
        traces = await fetch_recent_traces()
        if not traces:
            await asyncio.sleep(5)
            continue

        patch = await generate_patch(traces)
        if not patch:
            log.info("No patch generated, continuing.")
            continue

        accepted, verification_score = await _verify_patch(traces, patch)
        if not accepted:
            log.warning(
                f"Patch for {patch.get('target')} was not approved by verifier (score: {verification_score}), skipping application."
            )
            # Optionally, add to a different Redis log for rejected patches
            # REDIS.lpush("dgm:rejected_patches", json.dumps(patch))
            continue

        if _apply_patch(patch):
            # Evaluate reward on the same trace batch
            new_r = sum(
                proofable_reward(t, patch.get("after")) for t in traces
            )  # Pass patch content to reward
            if new_r >= 0:  # simple non-regression gate for now
                REDIS.lpush(
                    APPLIED_LOG,
                    json.dumps(
                        patch
                        | {"reward": new_r, "verification_score": verification_score}
                    ),
                )
                log.info(
                    "Patch applied ✔️  reward=%.4f, verification_score=%.2f",
                    new_r,
                    verification_score,
                )
            else:
                log.warning("Patch rolled back, reward=%.4f", new_r)
                _rollback(patch)
                for t in traces:
                    t["rolled_back"] = True
                    REDIS.lpush(ROLLED_BACK_LOG, json.dumps(t))
        else:
            log.error(f"Failed to apply patch for target: {patch.get('target')}")
            # _rollback(patch) # Consider if rollback is safe if apply itself failed.


def _rollback(patch: dict):
    """Rolls back the patch by writing the 'before' content to the target file and reloading the module."""
    tgt = Path(patch["target"])
    tgt.write_text(patch["before"])
    importlib.invalidate_caches()
    module_name = str(tgt.with_suffix("")).replace("/", ".")
    try:
        module = importlib.import_module(module_name)
        importlib.reload(module)
        log.info(f"Rolled back {patch['target']} successfully.")
    except Exception as e:
        # Log error during rollback, but proceed as the file is reverted.
        log.error(f"Error reloading module {module_name} during rollback: {e}")


# Entrypoint for standalone container / CLI
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Ensure json is imported if this script is run directly (it's already there)
    # import json

    parser = argparse.ArgumentParser(description="DGM Kernel Meta-Loop")
    parser.add_argument(
        "--once", action="store_true", help="Run the meta-loop only once."
    )
    args = parser.parse_args()

    if args.once:
        log.info("Running DGM meta-loop once.")

        # Define a new async function to run the loop once
        async def run_once():
            traces = await fetch_recent_traces()
            if not traces:
                log.info("No traces found, exiting.")
                return

            patch = await generate_patch(traces)
            if not patch:
                log.info("No patch generated, exiting.")
                return

            accepted, verification_score = await _verify_patch(traces, patch)
            if not accepted:
                log.warning(
                    f"Patch for {patch.get('target')} was not approved (score: {verification_score}), exiting."
                )
                return

            if _apply_patch(patch):
                new_r = sum(
                    proofable_reward(t, patch.get("after")) for t in traces
                )  # Pass patch content to reward
                if new_r >= 0:
                    REDIS.lpush(
                        APPLIED_LOG,
                        json.dumps(
                            patch
                            | {
                                "reward": new_r,
                                "verification_score": verification_score,
                            }
                        ),
                    )
                    log.info(
                        "Patch applied (once) ✔️  reward=%.4f, verification_score=%.2f",
                        new_r,
                        verification_score,
                    )
                else:
                    log.warning("Patch rolled back (once), reward=%.4f", new_r)
                    _rollback(patch)
                    for t in traces:
                        t["rolled_back"] = True
                        REDIS.lpush(ROLLED_BACK_LOG, json.dumps(t))
            else:
                log.error(
                    f"Failed to apply patch (once) for target: {patch.get('target')}"
                )

        asyncio.run(run_once())
        log.info("DGM meta-loop (once) finished.")
    else:
        log.info("Starting DGM meta-loop to run continuously.")
        asyncio.run(meta_loop())
