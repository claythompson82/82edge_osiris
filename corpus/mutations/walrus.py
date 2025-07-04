"""Assignment expressions."""


def example():
    if (n := len("walrus")) > 3:
        return n
