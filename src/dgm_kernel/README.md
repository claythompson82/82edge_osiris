# DGM Kernel

The `dgm_kernel` package implements a basic version of the Darwin–Gödel Machine (DGM) self–improvement loop. It watches recent run traces, proposes patches to its own modules and applies those changes when a simple proof check succeeds. This approach lets Osiris iteratively refine its trading strategies with minimal human input.

## Self‑Improvement Loop

The kernel runs continuously and executes the following cycle:

1. **Generate** – `fetch_recent_traces()` pulls the latest execution traces from Redis. `generate_patch()` then drafts a patch, currently by returning the example patch under `resources/dummy_patch.py.txt`.
2. **Prove** – `_verify_patch()` invokes `prove_patch()` to ensure the patch is safe. The prover rejects edits touching forbidden paths such as `.env` or `secrets/` and returns a Pylint score for quality. The loop requires a positive reward and sufficient lint score before accepting a patch.
3. **Apply** – `_apply_patch()` writes the patched content to disk and hot reloads the target module with `importlib.reload()`. Successful patches are logged to `dgm:applied_patches` in Redis along with their reward and lint score.
4. **Rollback** – If a patch reduces the `proofable_reward` on the same traces it was generated from, `_rollback()` restores the previous version and records the traces in `dgm:rolled_back_traces`.

This process mirrors the theoretical DGM "Generate → Prove → Apply → Self‑Test" loop and provides a foundation for automated policy evolution.

## Prover Details

The `prove_patch()` function performs lightweight validation:

- Empty diffs or patches are rejected.
- Patches that modify sensitive files (e.g. `.env`, anything under `secrets/`, or snapshot folders) are blocked.
- If the diff contains the token `STUB`, the patch is automatically approved to maintain backward compatibility during early experimentation.
- `_get_pylint_score()` can run `pylint` on the patch code to yield a score used by `_verify_patch()`.

This logic is intentionally simple and meant as a starting point for more rigorous proof procedures.

## Feedback and Adapter Updates

Rewards returned by `proofable_reward()` (currently a stub in `llm_sidecar.reward`) provide immediate feedback on each patch. Longer‑term improvement comes from nightly QLoRA fine‑tuning: feedback written to LanceDB is used to update the LoRA adapters loaded by the LLM sidecar. As the adapters evolve, the kernel can propose increasingly effective patches.

## Configuration and Extension Points

- **Redis connection** – change `REDIS_HOST` and `REDIS_PORT` environment variables if Redis is not reachable at the defaults in `meta_loop.py`.
- **Patch generator** – replace `llm_client.draft_patch()` with a call to your own LLM or search routine.
- **Reward function** – implement a richer `proofable_reward()` to measure the impact of a patch on recent traces.
- **Prover** – extend `prove_patch()` with additional safety checks or external verification services.

## Testing

Unit tests covering the prover and meta‑loop live under `tests/dgm_kernel/`. Install the test requirements and run:

```bash
pip install -r requirements-tests.txt
pytest tests/dgm_kernel
```

The meta‑loop can also be executed once for manual testing:

```bash
python -m dgm_kernel.meta_loop --once
```

