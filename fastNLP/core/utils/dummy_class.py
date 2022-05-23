__all__ = []

class DummyClass:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        return lambda *args, **kwargs: ...

    def __call__(self, *args, **kwargs):
        pass