from dataclasses import dataclass, field
from typing import TypeAlias, overload, Iterable
from functools import reduce

from typing import reveal_type

from icecream import ic

Shape: TypeAlias = tuple[int, ...]


class Bounds(list[int]):
    def __repr__(self) -> str:
        return f"({', '.join([str(c) for c in self])}))"

    @property
    def shape(self) -> Shape:
        return tuple([c for c in self])

    @property
    def total(self) -> float:
        return reduce(lambda x, y: x * y, self)


class Chunk(list[slice]):
    def __repr__(self) -> str:
        return f"({', '.join([f'{s.start}:{s.stop}' for s in self])}))"

    def range(self, index: int) -> range:
        return range(self[index].start, self[index].stop)

    @property
    def shape(self) -> Shape:
        return tuple([c.stop - c.start for c in self])


@dataclass
class ContainedBounds:
    outer_shape: Shape
    inner_shape: Shape
    offset: Shape
    dimensions: int = field(init=False)

    @staticmethod
    def from_chunk(outer_shape: Shape, inner_chunk: Chunk) -> "ContainedBounds":
        inner_shape = []
        offset = []
        for i, c in enumerate(inner_chunk):
            inner_shape.append(c.stop - c.start if c.stop >= 0 else outer_shape[i])
            offset.append(c.start)
        return ContainedBounds(outer_shape, tuple(inner_shape), tuple(offset))

    def __post_init__(self):
        assert len(self.outer_shape) == len(self.inner_shape)
        assert len(self.outer_shape) == len(self.offset)

        for ii, c in enumerate(self.inner_shape):
            assert c + self.offset[ii] <= self.outer_shape[ii]

        self.dimensions = len(self.outer_shape)

    def inner_slices(self) -> Chunk:
        return Chunk(
            [
                slice(self.offset[ii], self.offset[ii] + c)
                for ii, c in enumerate(self.inner_shape)
            ]
        )

    def inner_chunk(self, outer_chunk: Chunk) -> Chunk:
        assert len(outer_chunk) == self.dimensions

        return Chunk(
            [
                slice(s, e)
                for s, e in zip(
                    self.inner_index(*[int(c.start) for c in outer_chunk]),
                    self.inner_index(*[int(c.stop) for c in outer_chunk]),
                )
            ]
        )

    def outer_chunk(self, inner_chunk: Chunk) -> Chunk:
        return Chunk(
            [
                slice(
                    c.start + self.offset[ii],
                    c.stop + self.offset[ii],
                )
                for ii, c in enumerate(inner_chunk)
            ]
        )

    def inner_index(self, *outer_index: int) -> list[int]:
        assert len(outer_index) == self.dimensions
        results = []
        for ii, c in enumerate(outer_index):
            v = (c if c >= 0 else self.inner_shape[ii]) - self.offset[ii]
            if v < 0 or v >= self.inner_shape[ii]:
                raise IndexError(f" Outer value {c} is outside of inner range")
            results.append(v)
        return results

    def outer_index(self, *inner_index: int) -> list[int]:
        assert len(inner_index) == self.dimensions
        return [
            (c if c >= 0 else self.inner_shape[ii]) + self.offset[ii]
            for ii, c in enumerate(inner_index)
        ]
