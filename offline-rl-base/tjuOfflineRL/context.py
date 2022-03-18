from contextlib import contextmanager
from typing import Iterator, List

PARALLEL_FLAG_STACK: List[bool] = [False]


def get_parallel_flag() -> bool:
    return PARALLEL_FLAG_STACK[-1]


@contextmanager
def parallel() -> Iterator[None]:
    PARALLEL_FLAG_STACK.append(True)
    yield
    PARALLEL_FLAG_STACK.pop(-1)


@contextmanager
def disable_parallel() -> Iterator[None]:
    PARALLEL_FLAG_STACK.append(False)
    yield
    PARALLEL_FLAG_STACK.pop(-1)
