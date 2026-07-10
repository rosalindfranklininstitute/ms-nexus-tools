# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TypeAlias
from functools import reduce
import numpy as np

Shape: TypeAlias = tuple[int, ...]


class Bounds(list[int]):
    def __repr__(self) -> str:
        return f"({', '.join([str(c) for c in self])}))"

    @property
    def shape(self) -> Shape:
        return tuple(self)

    @property
    def total(self) -> float:
        return reduce(lambda x, y: x * y, self)

    def full_chunk(self) -> "Chunk":
        return Chunk([slice(0, c) for c in self])


class Chunk(list[slice]):
    def __repr__(self) -> str:
        return f"({', '.join([f'{s.start}:{s.stop}' for s in self])})"

    def range(self, index: int) -> range:
        return range(self[index].start, self[index].stop)

    def arange(self, index: int, step: int | float = 1) -> np.ndarray:
        return np.arange(self[index].start, self[index].stop, step)

    @property
    def start(self) -> Shape:
        return tuple([c.start for c in self])

    @property
    def shape(self) -> Shape:
        return tuple([c.stop - c.start for c in self])
