"""Pattern based intent matcher used for routing requests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import yaml


@dataclass
class _Intent:
    name: str
    patterns: List[re.Pattern[str]]
    route: str


def _compile(pattern: str) -> re.Pattern[str]:
    """Convert a simple glob pattern to a regular expression."""
    regex = "^" + re.escape(pattern).replace("\\*", ".*") + "$"
    return re.compile(regex, re.IGNORECASE)


def _load_intents() -> List[_Intent]:
    path = Path(__file__).with_name("intent_patterns.yaml")
    data = yaml.safe_load(path.read_text()) or {}
    items = data.get("intents", [])
    intents: List[_Intent] = []
    for item in items:
        raw_patterns = item.get("patterns", [])
        compiled = [_compile(p) for p in raw_patterns]
        intents.append(
            _Intent(name=item.get("name", "UNKNOWN"), patterns=compiled, route=item.get("route", "UNKNOWN"))
        )
    return intents


_INTENTS: List[_Intent] = _load_intents()


class IntentRouter:
    """Match text against YAML patterns and suggest a routing target."""

    @staticmethod
    def get_intent(text: str) -> Tuple[str, str]:
        for intent in _INTENTS:
            for pattern in intent.patterns:
                if pattern.search(text):
                    return intent.name, intent.route
        return "UNKNOWN", "UNKNOWN"
