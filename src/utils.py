def print_e(desc: str, e: Exception | None = None, add_msg: str = "") -> None:
    """Print a formatted error message and exit the process.

    Args:
        desc: short description of the error
        e: optional exception to display
        add_msg: optional additional message to display
    """
    print("=" * 5, desc, "=" * 5)
    if (e):
        print(e)
        print("=" * 10)
    print(add_msg)
    exit()
