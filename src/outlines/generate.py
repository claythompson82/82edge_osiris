"""Simplified outlines.generate module for tests."""
from typing import Any, Dict, Callable, Awaitable


def json(model: Any, schema: Dict[str, Any], tokenizer: Any = None) -> Callable[[str, int], Awaitable[Dict[str, Any]]]:
    async def _generator(prompt: str, max_tokens: int) -> Dict[str, Any]:
        return {}
    return _generator


def text(model: Any, tokenizer: Any = None) -> Callable[[str, int], Awaitable[str]]:
    async def _generator(prompt: str, max_tokens: int) -> str:
        return ""
    return _generator
