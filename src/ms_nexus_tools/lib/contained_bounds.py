# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
import math
import itertools

from dataclasses import dataclass, field
from typing import TypeAlias, overload, Iterable
from functools import reduce
import numpy as np

from .bounds import Bounds, Chunk, Shape
from .exceptions import NoDataError, InnerDataNotContainedError


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
            if c == 0 or self.outer_shape[ii] == 0:
                raise NoDataError(f"No data in dimension {ii + 1}.")
            elif c + self.offset[ii] > self.outer_shape[ii]:
                raise InnerDataNotContainedError(
                    f"Inner shape not contained in outer shape on diemsnion {ii + 1}."
                )

        self.dimensions = len(self.outer_shape)

    def inner_index(self, *outer_index: int) -> list[int]:
        """
        Returns the equivalent index into the inner chunk for the given outer index.
        >>> cbounds = ContainedBounds.from_chunk(
        ...     (100,100,100),
        ...     [slice(10,20), slice(20,30), slice(30,40)],
        ... )
        >>> cbounds.inner_index(11,21,31)
        [1, 1, 1]
        >>> cbounds.inner_index(11,21,-61)
        [1, 1, 9]
        """
        assert len(outer_index) == self.dimensions
        results = []
        for ii, c in enumerate(outer_index):
            v = (c if c >= 0 else self.outer_shape[ii] + c) - self.offset[ii]
            if v < 0 or v >= self.inner_shape[ii]:
                raise IndexError(f" Outer value {c} is outside of inner range")
            results.append(v)
        return results

    def inner_slices(self) -> Chunk:
        """
        Returns a slice into the outer data that will return the contained data.
        >>> cbounds = ContainedBounds.from_chunk(
        ...     (100,100,100),
        ...     [slice(10,20), slice(20,30), slice(30,40)],
        ... )
        >>> cbounds.inner_slices()
        (10:20, 20:30, 30:40)
        """

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

    def outer_index(self, *inner_index: int) -> list[int]:
        """
        Returns the equivalent index into the outer chunk for the given inner index.
        >>> cbounds = ContainedBounds.from_chunk(
        ...     (100,100,100),
        ...     [slice(10,20), slice(20,30), slice(30,40)],
        ... )
        >>> cbounds.outer_index(1,1,1)
        [11, 21, 31]
        >>> cbounds.outer_index(1,1,-1)
        [11, 21, 39]
        """
        assert len(inner_index) == self.dimensions
        return [
            (c if c >= 0 else (self.inner_shape[ii] + c)) + self.offset[ii]
            for ii, c in enumerate(inner_index)
        ]

    def outer_chunk(self, inner_chunk: Chunk) -> Chunk:
        return Chunk(
            [
                slice(s, e)
                for s, e in zip(
                    self.outer_index(*[int(c.start) for c in inner_chunk]),
                    self.outer_index(*[int(c.stop) for c in inner_chunk]),
                )
            ]
        )

    def chunk_edges(self, chunk_shape: Shape) -> list[list[int]]:
        """
        Returns the list of edges of the chunking along each axis.
        The chunks are aligned with the outer data origin
        (so the first and a last in ech diemnsion may be smaller than chunk_shape.)
        Each element in the array is a tuple of outer_chunk, inner_chunk.

        >>> cbounds = ContainedBounds.from_chunk(
        ...     (100,100,100),
        ...     [slice(10,20), slice(20,30), slice(30,40)],
        ... )
        >>> cbounds.chunk_edges((10,10,10))
        [[10, 20], [20, 30], [30, 40]]
        >>> cbounds = ContainedBounds.from_chunk(
        ...     (100,100,100),
        ...     [slice(11,21), slice(11,31), slice(30,40)],
        ... )
        >>> cbounds.chunk_edges((10,10,10))
        [[11, 20, 21], [11, 20, 30, 31], [30, 40]]
        """
        inner_stops = [off + inn for off, inn in zip(self.offset, self.inner_shape)]
        start_counts = [off // cs for off, cs in zip(self.offset, chunk_shape)]
        end_counts = [
            math.ceil(stop / cs) for stop, cs in zip(inner_stops, chunk_shape)
        ]
        chunk_indices = [
            [a for a in range(sc + 1, ec)] for sc, ec in zip(start_counts, end_counts)
        ]

        chunk_edges = [
            [start, *[ii * cs for ii in inx], stop]
            for inx, cs, start, stop in zip(
                chunk_indices, chunk_shape, self.offset, inner_stops
            )
        ]
        return chunk_edges

    def chunks(self, chunk_shape: Shape) -> list[tuple[Chunk, Chunk]]:
        """
        Returns the list of chunks of chunk shape that will cover the inner data.
        The chunks are aligned with the outer data origin
        (so the first and a last in ech diemnsion may be smaller than chunk_shape.)
        Each element in the array is a tuple of outer_chunk, inner_chunk.

        >>> cbounds = ContainedBounds.from_chunk(
        ...     (100,100,100),
        ...     [slice(10,20), slice(20,30), slice(30,40)],
        ... )
        >>> cbounds.chunks((10,10,10))
        [((10:20, 20:30, 30:40), (0:10, 0:10, 0:10))]
        >>> cbounds.chunks((5,10,10))
        [((10:15, 20:30, 30:40), (0:5, 0:10, 0:10)), ((15:20, 20:30, 30:40), (5:10, 0:10, 0:10))]
        >>> cbounds = ContainedBounds.from_chunk(
        ...     (100,100,100),
        ...     [slice(11,21), slice(20,30), slice(30,40)],
        ... )
        >>> cbounds.chunks((10,10,10))
        [((11:20, 20:30, 30:40), (0:9, 0:10, 0:10)), ((20:21, 20:30, 30:40), (9:10, 0:10, 0:10))]
        """

        inner_stops = [off + inn for off, inn in zip(self.offset, self.inner_shape)]
        start_counts = [off // cs for off, cs in zip(self.offset, chunk_shape)]
        end_counts = [
            math.ceil(stop / cs) for stop, cs in zip(inner_stops, chunk_shape)
        ]

        chunk_indices = itertools.product(
            *[[a for a in range(sc, ec)] for sc, ec in zip(start_counts, end_counts)]
        )

        results = []
        for inx in chunk_indices:
            outer_chunk = Chunk(
                [
                    slice(max(ii * cs, start), min((ii + 1) * cs, stop))
                    for ii, cs, start, stop in zip(
                        inx, chunk_shape, self.offset, inner_stops
                    )
                ]
            )
            inner_chunk = Chunk(
                [
                    slice(slc.start - off, slc.stop - off)
                    for slc, off in zip(outer_chunk, self.offset)
                ]
            )
            results.append((outer_chunk, inner_chunk))

        return results
