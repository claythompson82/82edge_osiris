from __future__ import annotations

from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    """Abstract classifier interface."""

    @abstractmethod
    def predict(self, text: str) -> tuple[str, float]:
        """Return a tuple of ``(intent, confidence)`` for the input ``text``."""
        raise NotImplementedError


class DummyClassifier(BaseClassifier):
    """Very small keyword-based intent classifier."""

    _keywords = {
        "buy": "BUY",
        "sell": "SELL",
    }

    def predict(self, text: str) -> tuple[str, float]:
        lowered = text.lower()
        for kw, intent in self._keywords.items():
            if kw in lowered:
                return intent, 0.8
        return "UNKNOWN", 0.0


# Singleton instance used by the router
CLASSIFIER: BaseClassifier = DummyClassifier()
