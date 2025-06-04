import sys
from unittest.mock import MagicMock

import pytest

# Stub heavy optional dependencies so hermes_plugin and loader import cleanly
for name in [
    "torch",
    "torchaudio",
    "chatterbox",
    "sentry_sdk",
    "lancedb",
    "lancedb.pydantic",
    "transformers",
    "optimum",
    "optimum.onnxruntime",
    "peft",
]:
    sys.modules.setdefault(name, MagicMock())

from llm_sidecar.hermes_plugin import score_with_hermes
from llm_sidecar.loader import get_latest_adapter_dir


class DummyTensor:
    def __init__(self, length: int = 1):
        self._shape = (1, length)

    @property
    def shape(self):
        return self._shape


class DummyInputDict(dict):
    def to(self, device):
        return self


class DummyTokenizer:
    eos_token_id = 0

    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None

    def __call__(self, text, return_tensors=None, truncation=True, max_length=None):
        self.last_prompt = text
        return DummyInputDict({"input_ids": DummyTensor()})

    def decode(self, tokens, skip_special_tokens=True):
        return self.response


class DummyModel:
    def generate(self, **inputs):
        return [[0]]


def test_score_with_hermes_parses_number(monkeypatch):
    tokenizer = DummyTokenizer("7")
    model = DummyModel()
    monkeypatch.setattr(
        "llm_sidecar.hermes_plugin.get_hermes_model_and_tokenizer",
        lambda: (model, tokenizer),
    )
    result = score_with_hermes({"foo": "bar"}, context="ctx")
    assert pytest.approx(result) == 0.7
    assert "Context:\nctx" in tokenizer.last_prompt


def test_score_with_hermes_invalid_number(monkeypatch):
    tokenizer = DummyTokenizer("eleven")
    model = DummyModel()
    monkeypatch.setattr(
        "llm_sidecar.hermes_plugin.get_hermes_model_and_tokenizer",
        lambda: (model, tokenizer),
    )
    result = score_with_hermes({"foo": "bar"})
    assert result == -1.0


def test_score_with_hermes_out_of_range(monkeypatch):
    tokenizer = DummyTokenizer("15")
    model = DummyModel()
    monkeypatch.setattr(
        "llm_sidecar.hermes_plugin.get_hermes_model_and_tokenizer",
        lambda: (model, tokenizer),
    )
    result = score_with_hermes({"foo": "bar"})
    assert result == -1.0


def test_get_latest_adapter_dir(tmp_path):
    first = tmp_path / "20240101"
    first.mkdir()
    (first / "adapter_config.json").write_text("{}")
    second = tmp_path / "20240505"
    second.mkdir()
    (second / "adapter_config.json").write_text("{}")
    latest = get_latest_adapter_dir(str(tmp_path))
    assert latest == str(second)


def test_get_latest_adapter_dir_no_valid(tmp_path):
    (tmp_path / "20240101").mkdir()
    assert get_latest_adapter_dir(str(tmp_path)) is None


def test_get_latest_adapter_dir_missing_base(tmp_path):
    missing = tmp_path / "none"
    assert get_latest_adapter_dir(str(missing)) is None
