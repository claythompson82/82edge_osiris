class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Dummy method used for tests."""
        return cls()


class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Dummy method used for tests."""
        return cls()
