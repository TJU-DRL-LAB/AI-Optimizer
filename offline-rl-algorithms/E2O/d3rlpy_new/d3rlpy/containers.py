from typing import (
    Callable,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeVar,
)

T = TypeVar("T")


class FIFOQueue(Generic[T]):
    """Simple FIFO queue implementation.

    Random access of this queue object is O(1).

    """

    _maxlen: int
    _drop_callback: Optional[Callable[[T], None]]
    _buffer: List[Optional[T]]
    _cursor: int
    _size: int
    _index: int

    def __init__(
        self, maxlen: int, drop_callback: Optional[Callable[[T], None]] = None
    ):
        self._maxlen = maxlen
        self._drop_callback = drop_callback
        self._buffer = [None for _ in range(maxlen)]
        self._cursor = 0
        self._size = 0
        self._index = 0

    def append(self, item: T) -> None:
        # call drop callback if necessary
        cur_item = self._buffer[self._cursor]
        if cur_item and self._drop_callback:
            self._drop_callback(cur_item)

        self._buffer[self._cursor] = item

        # increment cursor
        self._cursor += 1
        if self._cursor == self._maxlen:
            self._cursor = 0
        self._size = min(self._size + 1, self._maxlen)

    def extend(self, items: Sequence[T]) -> None:
        for item in items:
            self.append(item)

    def __getitem__(self, index: int) -> T:
        assert index < self._size

        # handle negative indexing
        if index < 0:
            index = self._size + index

        item = self._buffer[index]
        assert item is not None
        return item

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[T]:
        self._index = 0
        return self

    def __next__(self) -> T:
        if self._index >= self._size:
            raise StopIteration
        item = self._buffer[self._index]
        assert item is not None
        self._index += 1
        return item
