import importlib


def is_binder_available() -> bool:
    return importlib.util.find_spec("binder") is not None
