from __future__ import annotations

import random
from typing import Sequence, Mapping, Any


_DEFAULT_PROBABILITIES = {
    "ASTInsertComment": 0.8,
    "ASTRenameIdentifier": 0.2,
}


def choose_mutation(traces: Sequence[Mapping[str, Any]] | None = None) -> str:
    """Return the mutation strategy name for the next patch.

    Selection is randomized with an 80% chance of ``ASTInsertComment`` and a 20%
    chance of ``ASTRenameIdentifier``. *traces* are currently ignored but
    included for future use.
    """
    _ = traces  # unused placeholder for future heuristics
    pick = random.random()
    if pick < _DEFAULT_PROBABILITIES["ASTInsertComment"]:
        return "ASTInsertComment"
    return "ASTRenameIdentifier"
