# dgm_kernel/meta_loop.py
"""
Darwin Gödel Machine – self-improvement meta-loop (async capable)

TODOs are marked ▼
"""

from __future__ import annotations
import asyncio, importlib, logging, time, json # Added json import
import redis # Changed from 'from redis import Redis'
from pathlib import Path
# from redis import Redis # Original import
from llm_sidecar.reward import proofable_reward

log = logging.getLogger(__name__)

REDIS = redis.Redis(host="redis", port=6379, decode_responses=True) # Updated instantiation
PATCH_QUEUE = "dgm:patch_queue"        # ← incoming JSON patches
TRACE_QUEUE = "dgm:recent_traces"      # ← recent trading traces (jsonl)
APPLIED_LOG  = "dgm:applied_patches"   # ← audit trail

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
            return [] # Return empty list if no traces were fetched

        for i, trace_str in enumerate(raw_traces):
            try:
                traces.append(json.loads(trace_str))
            except json.JSONDecodeError as e:
                log.error(f"Failed to decode JSON for trace: {trace_str}. Error: {e}. Trace index: {i}")
                # Optionally, could add malformed trace to a separate Redis list for inspection
                # REDIS.lpush("dgm:malformed_traces", trace_str)

        if traces: # Log only if some traces were successfully parsed
            log.info(f"Fetched {len(traces)} traces successfully, encountered {len(raw_traces) - len(traces)} decoding errors.")
        elif raw_traces: # Log if raw traces were fetched but none could be parsed
            log.warning(f"Fetched {len(raw_traces)} raw traces, but all failed to decode.")
        # If raw_traces is empty, nothing is logged by these conditions, which is fine.

    except redis.exceptions.RedisError as e:
        log.error(f"Redis error while fetching traces: {e}")
        # Depending on the severity, could raise or return empty list
        return [] # Or handle more gracefully
    except Exception as e: # Catch any other unexpected errors
        log.error(f"Unexpected error in fetch_recent_traces: {e}")
        return []

    return traces

async def generate_patch(traces: list[dict]) -> dict | None: # Return type updated to include None
    """
    Placeholder for LLM / search routine to emit a JSON patch.
    ▼ TODO: Replace with actual patch generation logic.
    """
    if not traces:
        log.warning("generate_patch called with no traces, skipping.")
        return None

    log.info(f"generate_patch called with {len(traces)} traces. Generating a placeholder patch.")

    # This is a dummy patch. In a real scenario, this would be dynamically generated.
    # The 'before' content should ideally be fetched from the actual target file.
    # For this placeholder, we'll use a simplified 'before'.
    # IMPORTANT: The target file 'osiris_policy/strategy.py' must exist or be creatable by _apply_patch.
    # For initial testing, _apply_patch can create it.

    # Ensure the target directory and a placeholder 'before' content exists for the dummy patch to be meaningful.
    # This part would normally be handled by fetching current content of `patch_target_file`.
    # For this placeholder, we assume `_apply_patch` can handle creating the file if it doesn't exist,
    # and `_rollback` can write back the 'before' state.

    patch_target_file = Path("osiris_policy/strategy.py")
    current_content = ""
    try:
        if patch_target_file.exists():
            current_content = patch_target_file.read_text()
        else:
            # If the file doesn't exist, we can prime it with some initial content
            # or ensure _apply_patch creates it. For simplicity, assume it might not exist.
            log.info(f"Target file {patch_target_file} does not exist. 'before' will be empty for this placeholder.")
            # It's also possible to create a dummy file here for the placeholder to work more realistically.
            # patch_target_file.parent.mkdir(parents=True, exist_ok=True)
            # patch_target_file.write_text("# Initial placeholder content
")
            # current_content = "# Initial placeholder content
"
    except Exception as e:
        log.error(f"Error reading target file {patch_target_file} for 'before' state: {e}")
        # Decide how to handle: skip patch, or use empty 'before'
        # return None

    placeholder_patch = {
        "target": str(patch_target_file), # Ensure it's a string
        "before": current_content,
        "after": current_content + "\n# Placeholder patch: DGM was here at " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n",
        "rationale": "This is a placeholder patch generated by the DGM meta_loop for testing purposes."
    }

    log.info(f"Generated placeholder patch: {placeholder_patch}")
    # Simulate some processing time
    await asyncio.sleep(1)
    return placeholder_patch

async def _prove_patch(patch: dict) -> bool:
    """
    Placeholder for submitting patch to a verifier daemon.
    ▼ TODO: Replace with actual verifier daemon interaction.
    """
    if not patch:
        log.warning("_prove_patch called with no patch, skipping.")
        return False

    log.info(f"Submitting patch for {patch.get('target')} to verifier daemon (placeholder).")
    log.debug(f"Patch content: {patch}") # Log full patch at debug level

    # Simulate verifier interaction time
    await asyncio.sleep(0.5)

    # Placeholder: Assume patch is always approved
    log.info(f"Patch for {patch.get('target')} approved by verifier daemon (placeholder).")
    return True

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
    module_name = str(tgt.with_suffix('')).replace('/', '.')
    try:
        module = importlib.import_module(module_name)
        importlib.reload(module)
    except ModuleNotFoundError:
        # This can happen if the module is being created for the first time
        # or if it's not in a package.
        # Attempt to load it based on its path directly if it's a new top-level module
        # This part can be tricky and might need adjustment based on project structure
        log.warning(f"Module {module_name} not found directly, attempting to load spec.")
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
            await asyncio.sleep(5); continue

        patch = await generate_patch(traces)
        if not patch:
            log.info("No patch generated, continuing.")
            continue

        # ▼ Add call to _prove_patch here
        patch_is_proven = await _prove_patch(patch)
        if not patch_is_proven:
            log.warning(f"Patch for {patch.get('target')} was not approved by verifier, skipping application.")
            # Optionally, add to a different Redis log for rejected patches
            # REDIS.lpush("dgm:rejected_patches", json.dumps(patch))
            continue

        if _apply_patch(patch):
            # Evaluate reward on the same trace batch
            new_r = sum(proofable_reward(t, {}) for t in traces)
            if new_r >= 0:   # simple non-regression gate for now
                REDIS.lpush(APPLIED_LOG, json.dumps(patch | {"reward": new_r}))
                log.info("Patch applied ✔️  reward=%.4f", new_r)
            else:
                log.warning("Patch rolled back, reward=%.4f", new_r)
                _rollback(patch)
        else:
            log.error(f"Failed to apply patch for target: {patch.get('target')}")
            # _rollback(patch) # Consider if rollback is safe if apply itself failed.

def _rollback(patch: dict):
    """Rolls back the patch by writing the 'before' content to the target file and reloading the module."""
    tgt = Path(patch["target"])
    tgt.write_text(patch["before"])
    importlib.invalidate_caches()
    module_name = str(tgt.with_suffix('')).replace('/', '.')
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
    # Ensure json is imported if this script is run directly
    import json
    asyncio.run(meta_loop())
