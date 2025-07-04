"""Context manager usage."""


class Ctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


with Ctx() as c:
    pass
