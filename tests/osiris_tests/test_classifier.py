from __future__ import annotations

import importlib.util
from pathlib import Path

# Load the classifier module without triggering osiris package imports
_spec = importlib.util.spec_from_file_location(
    "osiris.intent_classifier",
    Path(__file__).resolve().parents[2] / "src/osiris/intent_classifier.py",
)
_classifier_module = importlib.util.module_from_spec(_spec)
assert _spec.loader
_spec.loader.exec_module(_classifier_module)
CLASSIFIER = _classifier_module.CLASSIFIER


def test_buy_intent() -> None:
    intent, conf = CLASSIFIER.predict("please BUY the stock")
    assert intent == "BUY"
    assert conf == 0.8


def test_sell_intent() -> None:
    intent, conf = CLASSIFIER.predict("time to SELL everything")
    assert intent == "SELL"
    assert conf == 0.8


def test_unknown_intent() -> None:
    intent, conf = CLASSIFIER.predict("hold the line")
    assert intent == "UNKNOWN"
    assert conf == 0.0
