import random

from dgm_kernel.ab_mutator import choose_mutation


def test_distribution_stable() -> None:
    random.seed(42)
    results = [choose_mutation([]) for _ in range(1000)]
    comment_ratio = results.count("ASTInsertComment") / len(results)
    rename_ratio = results.count("ASTRenameIdentifier") / len(results)
    assert 0.75 <= comment_ratio <= 0.85
    assert 0.15 <= rename_ratio <= 0.25
