from typing import Iterator, Any, Iterable


def count_digits(num: int) -> int:
    """
    Counts the number of digits in an integer:
    >>> count_digits(1), count_digits(2)
    (1, 1)

    >>count_digits(10), count_digits(12)
    (2, 2)

    >>> count_digits(0)
    1
    """
    digits = 1
    num = abs(num) // 10
    while abs(num) > 0:
        digits += 1
        num = num // 10
    return digits


def slice_len(slc: slice) -> int:
    """
    Returns the length of a slice
    >>> slice_len(slice(5))
    5

    >>> slice_len(slice(1, 5))
    4

    >>> slice_len(slice(1, 5, 2))
    2
    """
    inc = slc.step or 1

    if slc.start is None:
        return slc.stop // inc
    else:
        return (slc.stop - slc.start) // inc


class NotTqdm:
    def __init__(self, iterator: Iterable | None = None, **kwargs):
        self.iterator = iterator

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self) -> Iterator[Any]:
        if self.iterator is None:
            raise TypeError("NotTqdm expected an iterable when used as an iterator.")
        for item in self.iterator:
            yield item

    def update(self):
        pass
