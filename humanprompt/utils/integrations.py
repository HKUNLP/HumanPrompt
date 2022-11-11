def is_binder_available() -> bool:
    try:
        import binder  # noqa: F401

        return True
    except ImportError:
        return False
