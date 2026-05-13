import math
from typing import Iterator, Any, Iterable
import json


def format_bytes(n: int, digits: int = 2) -> str:
    """
    Format the given number of bytes into byte units.
    >>> format_bytes(10)
    '10b'

    >>> format_bytes(1000)
    '1000b'

    >>> format_bytes(512+1024)
    '1.50Kb'

    >>> format_bytes(1024*1024*1.25)
    '1.25Mb'

    Digits defaults to 2, but can be specified.
    >>> format_bytes(512+1024, digits=1)
    '1.5Kb'

    The number of digits does not have an effect on integer values
    >>> format_bytes(1000, digits=1)
    '1000b'

    """
    negative = n < 0
    units = ["b", "Kb", "Mb", "Gb", "Tb", "Pb", "Eb"]
    i = 0
    value = abs(float(n))
    while value >= 1024 and i < len(units) - 1:
        value /= 1024.0
        i += 1
    prefix = "-" if negative else ""
    if value.is_integer():
        return f"{prefix}{int(value)}{units[i]}"
    else:
        return f"{prefix}{value:.{digits}f}{units[i]}"


def parse_bytes(bytes_str) -> int:
    """
    Parse the given string into the number of bytes.
    >>> parse_bytes('10b')
    10

    >>> parse_bytes('1.50Kb')
    1536

    >>> parse_bytes('1.25Mb')
    1310720

    >>> parse_bytes('1250Kb')
    1280000

    >>> parse_bytes('0.025Kb')
    26

    Works with output of format bytes
    >>> parse_bytes(format_bytes(512+1024))
    1536

    >>> parse_bytes(format_bytes(512+1024, digits=1))
    1536

    Provides the cailing of any fractions:
    >>> parse_bytes('1.1b')
    2

    """

    values = dict(
        Kb=1024, Mb=1024**2, Gb=1024**3, Tb=1024**4, Pb=1024**5, Eb=1024**6, b=1
    )

    bytes_str = bytes_str.strip()
    value = None
    for tail, multiplier in values.items():
        if bytes_str.endswith(tail):
            value = float(bytes_str.removesuffix(tail)) * multiplier
            break
    else:
        raise ValueError(f"Did not understand the suffix of {bytes_str}.")

    if value is None:
        raise RuntimeError("Suffix found, but value was invalid")

    return int(math.ceil(value))


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


def slice_range(slc: slice) -> range:
    """
    Returns the range of the slice:
    >>> slice_range(slice(5))
    range(0, 5)

    >>> slice_range(slice(1, 5))
    range(1, 5)

    >>> slice_range(slice(1, 5, 2))
    range(1, 5, 2)
    """
    if slc.start is None and slc.step is None:
        return range(slc.stop)
    elif slc.step is None:
        return range(slc.start, slc.stop)
    elif slc.start is None:
        return range(0, slc.stop, slc.step)
    else:
        return range(slc.start, slc.stop, slc.step)


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


def json_add(filename, *keys, value):
    old_data = {}
    if filename.exists():
        with open(filename, "r") as fd:
            old_data = json.load(fd)
    if len(keys) >= 1:
        new_data = old_data
        for key in keys[:-1]:
            if key not in new_data:
                new_data[key] = {}
            new_data = new_data[key]
        new_data[keys[-1]] = value
    else:
        assert isinstance(value, dict)
        old_data.update(value)
    with open(filename, "w") as fd:
        json.dump(old_data, fd, indent=2)
