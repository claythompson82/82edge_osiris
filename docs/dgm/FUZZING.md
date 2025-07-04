# Fuzzing the DGM Kernel

The `fuzz_runner` module executes a suite of small Python files to stress the
mutation logic. Use the provided corpus under `corpus/mutations` or supply your
own directory.

Run the fuzzer from the repo root:

```bash
python -m dgm_kernel.fuzz_runner corpus/mutations
```

The runner will parse each snippet and report any failures.
