from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent


class Eval:
    pass


class Glove:
    num_threads = 8