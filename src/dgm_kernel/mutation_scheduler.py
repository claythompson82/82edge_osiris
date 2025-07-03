"""Adaptive mutation strategy scheduler for DGM."""

from __future__ import annotations

class MutationScheduler:
    """Simple scheduler that escalates mutation strategies when rewards drop."""

    def __init__(self) -> None:
        self._current = "ASTInsertComment"
        self._bad_streak = 0
        self._phase = 0

    def next_strategy(self, prev_reward: float | None) -> str:
        """Return the next mutation strategy based on previous reward."""
        if prev_reward is None or prev_reward > 0:
            self._bad_streak = 0
        else:
            self._bad_streak += 1

        if self._phase == 0 and self._bad_streak >= 3:
            self._current = "ASTRenameIdentifier"
            self._phase = 1
            self._bad_streak = 0
        elif self._phase == 1 and self._bad_streak >= 3:
            self._current = "ASTInsertCommentAndRename"
            self._phase = 2
            self._bad_streak = 0

        return self._current

