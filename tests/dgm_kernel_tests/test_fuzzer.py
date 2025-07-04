import ast

from dgm_kernel.mutation_fuzzer import fuzzer, load_corpus


def test_fuzzer_parsable_and_size() -> None:
    base = load_corpus()[0]
    out: list[str] = []
    for _ in range(100):
        mutated = fuzzer.fuzz_source(base)
        ast.parse(mutated)
        out.append(mutated)
    avg = sum(len(s) for s in out) / len(out)
    assert avg <= 2 * len(base)
