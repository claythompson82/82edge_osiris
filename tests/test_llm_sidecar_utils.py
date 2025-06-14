import pytest

# Dummy classes to use in the test
class DummyTokenizer:
    def __init__(self, value):
        self.value = value
        self.last_prompt = None

    def decode(self, tokens, **kwargs):
        # Always return a valid "7" string if no tokens provided, to prevent NoneType crash
        if tokens is None:
            return str(self.value)
        # Defensive: if tokens is an empty list, still return the value
        if isinstance(tokens, (list, tuple)) and not tokens:
            return str(self.value)
        return str(tokens[0])

class DummyModel:
    def __call__(self, *args, **kwargs):
        # Simulate LLM response, structure expected by score_with_hermes
        return [[7]]

def test_score_with_hermes_parses_number(monkeypatch):
    from llm_sidecar.hermes_plugin import score_with_hermes

    tokenizer = DummyTokenizer("7")
    model = DummyModel()

    # Patch the model/tokenizer getter
    monkeypatch.setattr(
        "llm_sidecar.hermes_plugin.get_hermes_model_and_tokenizer",
        lambda: (model, tokenizer)
    )
    # Run function, expecting result of 0.7
    result = score_with_hermes({"foo": "bar"}, context="ctx")
    assert abs(result - 0.7) < 1e-6
