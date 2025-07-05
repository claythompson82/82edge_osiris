codex/extend-router-with-fallback-chain
from __future__ import annotations

from enum import Enum, auto
from typing import Tuple, Dict, Any

from osiris_policy import orchestrator


class SelectedRoute(Enum):
    ORCHESTRATOR = auto()
    HERMES = auto()
    PHI3 = auto()
    AZR = auto()
    UNKNOWN = auto()


def _pattern_match(text: str) -> Tuple[SelectedRoute, str, float]:
    """Very small rules-based matcher for quick patterns."""
    txt = text.lower().strip()
    if txt.startswith("cmd:"):
        return SelectedRoute.ORCHESTRATOR, "command", 1.0
    if "quick fact" in txt:
        return SelectedRoute.PHI3, "quick_fact", 1.0
    return SelectedRoute.UNKNOWN, "", 0.0


def _classify(text: str) -> Tuple[SelectedRoute, str, float]:
    """Extremely naive classifier used when patterns fail."""
    txt = text.lower()
    if any(word in txt for word in ["buy", "sell", "trade"]):
        return SelectedRoute.ORCHESTRATOR, "trade", 0.6
    if any(word in txt for word in ["who", "what", "when", "where", "why"]):
        return SelectedRoute.HERMES, "question", 0.6
    return SelectedRoute.UNKNOWN, "", 0.0


def decide_route(text: str) -> Tuple[SelectedRoute, str, float]:
    """Return the best route for ``text``."""
    route, intent, conf = _pattern_match(text)
    if route is SelectedRoute.UNKNOWN:
        route, intent, conf = _classify(text)
    return route, intent, conf


# ---------------------------------------------------------------------------
#  Stubbed handlers
# ---------------------------------------------------------------------------

def _handle_hermes(intent: str, text: str, _context: Dict[str, Any]) -> str:
    return f"[HERMES] {text}"


def _handle_phi3(intent: str, text: str, _context: Dict[str, Any]) -> str:
    return f"[PHI3] {text}"


def _handle_azr(intent: str, text: str, _context: Dict[str, Any]) -> str:
    return f"[AZR] {text}"


# Order of fallbacks when a handler fails
_FALLBACK_ORDER = [
    SelectedRoute.ORCHESTRATOR,
    SelectedRoute.HERMES,
    SelectedRoute.PHI3,
    SelectedRoute.AZR,
]


def route_and_respond(text: str, context: Dict[str, Any] | None = None) -> str:
    """Route ``text`` and return a response string."""
    if context is None:
        context = {}

    route, intent, _ = decide_route(text)

    if route is SelectedRoute.UNKNOWN:
        return "I'm sorry, could you rephrase?"

    start = _FALLBACK_ORDER.index(route)
    for r in _FALLBACK_ORDER[start:]:
        try:
            if r is SelectedRoute.ORCHESTRATOR:
                response = orchestrator.handle_intent(intent, text)
            elif r is SelectedRoute.HERMES:
                response = _handle_hermes(intent, text, context)
            elif r is SelectedRoute.PHI3:
                response = _handle_phi3(intent, text, context)
            else:  # SelectedRoute.AZR
                response = _handle_azr(intent, text, context)
            if response:
                return response
        except Exception:
            continue

    return "I'm sorry, could you rephrase?"
=======
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
        return "UNKNOWN", "UNKNOWN
      main
