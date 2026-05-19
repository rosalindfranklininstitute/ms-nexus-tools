# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Iterator, Any, Iterable
from bisect import bisect_right, bisect_left
import json

import numpy as np

from .bounds import Shape
from .dtypes import Number1D, Number


def format_bytes(n: Number, digits: int = 2) -> str:
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

    >>count_digits(100), count_digits(314)
    (3, 3)

    >>count_digits(-100), count_digits(-10)
    (3, 2)

    >>> count_digits(0)
    1
    """
    digits = 1
    num = abs(num) // 10
    while num > 0:
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


def slice_from_values(start: Number, stop: Number, values: Number1D) -> slice:
    start_index = bisect_left(values, start)
    stop_index = bisect_right(values, stop)
    return slice(start_index, stop_index)


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


def indices(shape: Shape, axis=None) -> Iterator[tuple[slice | int, ...]]:
    if axis is None:
        yield (slice(None) for ii in range(len(shape)))
    else:
        if isinstance(axis, int):
            axis = [axis]
        axis = np.sort(axis)
        iterable_shape = [shape[ii] for ii in axis]
        ndims = len(shape)
        slices = np.array([0 if ii in axis else slice(None) for ii in range(ndims)])
        for values in np.ndindex(*iterable_shape):
            slices[axis] = values
            yield tuple(slices)


def iterate(array: np.ndarray, axis=None) -> Iterator[np.ndarray]:
    for slc in indices(array.shape, axis):
        yield array[*slc]


def reduce_shape(shape: Shape, axis=None) -> Shape:
    """
    Returns the data shape for the given axis.
    >>> reduce_shape((1,2,3))
    (1, 2, 3)

    >>> reduce_shape((1,2,3), axis=-1)
    (1, 2)

    >>> reduce_shape((1,2,3), axis=0)
    (2, 3)

    >>> reduce_shape((1,2,3), axis=(0, -1))
    (2,)
    """
    if axis is None:
        return shape
    else:
        if isinstance(axis, int):
            axis = [axis]
        ndim = len(shape)
        axis = np.sort([a if a >= 0 else a + ndim for a in axis])
        return Shape(v for ii, v in enumerate(shape) if ii not in axis)
