def print_e(desc: str, e: Exception = None, add_msg: str = ""):
    print("=" * 5, desc, "=" * 5)
    if (e):
        print(e)
    print("=" * 10)
    print(add_msg)
    exit()
