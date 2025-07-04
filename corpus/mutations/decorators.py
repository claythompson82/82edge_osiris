"""Decorator usage."""


def deco(func):
    def wrapper(*args, **kw):
        return func(*args, **kw)

    return wrapper


@deco
def greet(name):
    return f"hi {name}"
