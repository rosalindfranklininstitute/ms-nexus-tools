# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Generator, NamedTuple
from dataclasses import dataclass, field
import math
import itertools

import numpy as np

from icecream import ic

from .bounds import Chunk, Shape, Bounds


def count_chunks_to_cover(data_shape: Shape, chunk_shape: Shape) -> list[int]:
    """
    Return the number of chunks of chunk_shape are required to cover the data of shape data_shape.
    """
    return [math.ceil(data_shape[i] / c) for i, c in enumerate(chunk_shape)]


def _count_priorities(
    priorities: Shape | list[int],
) -> Generator[tuple[list[int], int]]:
    count = 0
    rev_prior = [r for r in np.argsort(priorities)]
    while count < len(priorities):
        priority = priorities[rev_prior[count]]
        dimensions = []
        while count < len(priorities) and priorities[rev_prior[count]] == priority:
            dimensions.append(rev_prior[count])
            count += 1
        yield dimensions, len(priorities) - count


class Chunker:
    """
    A class for calculating the chunking of a data blob based on the data,
    shape, the number of entries in the data and the priority of each axis.
    """

    def __init__(self):
        self.data_shape: Shape
        self.priorities: Shape
        self.n_dims: int

        self.chunk_shape: Shape
        self.chunk_count: Shape
        self.chunk_items: int
        self.n_chunks: int

    @staticmethod
    def from_max_item_count(
        data_shape: Shape,
        priorities: Shape,
        items_per_chunk: int | None,
    ) -> "Chunker":
        """
        Returns chunking where each chunk has at most item_per_chunk items.
        Lower priorities will have more values per chunk.
        So that:
        >>> Chunker.from_max_item_count(data_shape=(100,100), priorities=(1,2), items_per_chunk=10).chunk_shape
        (10, 1)

        and
        >>> Chunker.from_max_item_count(data_shape=(100,100), priorities=(2,1), items_per_chunk=10).chunk_shape
        (1, 10)
        """
        chunker = Chunker()
        assert len(data_shape) == len(priorities)
        chunker.data_shape = data_shape
        chunker.priorities = priorities
        chunker.n_dims = len(chunker.data_shape)

        chunker.chunk_shape, chunker.chunk_count = chunker._calculate_from_max_count(
            items_per_chunk
        )
        chunker.n_chunks = int(np.prod(chunker.chunk_count))
        chunker.chunk_items = int(np.prod(chunker.chunk_shape))
        return chunker

    def _calculate_from_max_count(self, max_items_per_chunk) -> tuple[Shape, Shape]:

        chunk_shape = [1 for _ in self.priorities]

        remaining_count = max_items_per_chunk
        for dimensions, remaining in _count_priorities(self.priorities):
            n_dims = len(dimensions)
            capacity = np.prod([self.data_shape[i] for i in dimensions])
            if capacity > remaining_count:
                dim_data_shape = np.array([self.data_shape[d] for d in dimensions])
                min_dim = np.min(dim_data_shape)
                weightings = dim_data_shape / min_dim

                items_per_dim = np.pow(
                    remaining_count / np.prod(weightings), 1 / n_dims
                )

                total = 1
                for d, w in zip(dimensions, weightings):
                    chunk_shape[d] = math.ceil(w * items_per_dim)
                    total *= chunk_shape[d]
                normal_w = weightings / np.max(weightings)
                while total > remaining_count:
                    total = 1
                    for d, w in zip(dimensions, normal_w):
                        chunk_shape[d] = max(chunk_shape[d] - w, 1)
                        total *= math.ceil(chunk_shape[d])
                for d in dimensions:
                    chunk_shape[d] = math.ceil(chunk_shape[d])
            else:
                for d in dimensions:
                    chunk_shape[d] = self.data_shape[d]

            remaining_count = max(remaining_count / capacity, 1)

        chunk_count = [
            math.ceil(self.data_shape[i] / c) for i, c in enumerate(chunk_shape)
        ]

        return tuple(chunk_shape), tuple(chunk_count)

    @staticmethod
    def from_min_chunks(
        data_shape: Shape,
        priorities: Shape,
        chunk_count: int,
    ) -> "Chunker":
        """
        Returns chunking where there are at most chunk_count chunks.
        Lower priorities will have more chunks.
        So that:
        >>> Chunker.from_min_chunks(data_shape=(100,100), priorities=(1,2), chunk_count=10).chunk_count
        (10, 1)

        and
        >>> Chunker.from_min_chunks(data_shape=(100,100), priorities=(2,1), chunk_count=10).chunk_count
        (1, 10)
        """
        chunker = Chunker()
        assert len(data_shape) == len(priorities)
        chunker.data_shape = data_shape
        chunker.priorities = priorities
        chunker.n_dims = len(chunker.data_shape)

        chunker.chunk_shape, chunker.chunk_count = chunker._calculate_from_min_chunks(
            chunk_count
        )
        chunker.n_chunks = int(np.prod(chunker.chunk_count))
        chunker.chunk_items = int(np.prod(chunker.chunk_shape))
        return chunker

    def _calculate_from_min_chunks(self, min_chunk_count):
        chunk_count = [1 for _ in self.priorities]

        max_remaining_chunks = min_chunk_count

        for dimensions, remaining in _count_priorities(self.priorities):
            dim_data_shape = np.array([self.data_shape[d] for d in dimensions])
            max_dim_chunks = np.prod(dim_data_shape)
            dim_data_shape = [self.data_shape[d] for d in dimensions]
            if max_dim_chunks <= max_remaining_chunks:
                for d in dimensions:
                    chunk_count[d] = self.data_shape[d]
                max_remaining_chunks = math.ceil(max_remaining_chunks / max_dim_chunks)
            else:
                for d, c in self._calculate_same_priority_from_min_chunks(
                    dimensions, max_remaining_chunks
                ):
                    chunk_count[d] = c
                max_dim_chunks = np.prod([self.data_shape[d] for d in dimensions])
                max_remaining_chunks = math.ceil(max_remaining_chunks / max_dim_chunks)
            if max_remaining_chunks == 1:
                break

        chunk_shape = [
            math.ceil(self.data_shape[i] / c) for i, c in enumerate(chunk_count)
        ]

        return tuple(chunk_shape), tuple(chunk_count)

    def _calculate_same_priority_from_min_chunks(
        self, dimensions: list[int], max_remaining_chunks: int
    ) -> list[tuple[int, int]]:
        assert np.prod([self.data_shape[d] for d in dimensions]) > max_remaining_chunks

        remaining_chunks = max_remaining_chunks
        below_average = [d for d in dimensions if self.data_shape[d] == 1]
        above_average = [d for d in dimensions if self.data_shape[d] > 1]
        previous_length = -1

        chunks_per_dim = -1

        while previous_length != len(below_average):
            previous_length = len(below_average)
            chunks_per_dim = math.ceil(
                np.pow(remaining_chunks, (1 / len(above_average)))
            )

            below_average.extend(
                [d for d in above_average if self.data_shape[d] < chunks_per_dim]
            )
            above_average = [
                d for d in above_average if self.data_shape[d] >= chunks_per_dim
            ]

            below_average_capacity = np.prod(
                [self.data_shape[d] for d in below_average]
            )
            remaining_chunks = math.ceil(max_remaining_chunks / below_average_capacity)

        assert chunks_per_dim > 0

        above_average_counts = []
        for d in above_average:
            chunk_count = chunks_per_dim
            chunk_shape = min(
                self.data_shape[d],
                math.ceil(self.data_shape[d] / chunk_count),
            )
            if chunk_shape * chunk_count >= (self.data_shape[d] + chunk_shape):
                if chunk_shape == 1:
                    chunk_count = self.data_shape[d]
                else:
                    chunk_shape -= 1
                    chunk_count = math.ceil(self.data_shape[d] / chunk_shape)
            above_average_counts.append((d, chunk_count))

        return [(d, self.data_shape[d]) for d in below_average] + above_average_counts

    @staticmethod
    def from_chunk_shape(
        data_shape: Shape,
        chunk_shape: Shape,
    ) -> "Chunker":
        chunker = Chunker()
        assert len(data_shape) == len(chunk_shape)
        chunker.data_shape = data_shape
        chunker.priorities = tuple([1 for _ in data_shape])
        chunker.n_dims = len(chunker.data_shape)

        chunker.chunk_shape = chunk_shape
        chunker.chunk_count = tuple(
            [math.ceil(d / c) for c, d in zip(chunk_shape, data_shape)]
        )
        chunker.n_chunks = int(np.prod(chunker.chunk_count))
        chunker.chunk_items = int(np.prod(chunker.chunk_shape))
        return chunker

    @staticmethod
    def find_chunk_multiple(
        data_shape, chunk_shape, max_item_count, priorities: Shape | None = None
    ) -> "Chunker":
        """
        Find the chunker that covers data_shape
        with chunks that are intiger multiples of chunk_shape,
        with at most max_item_count items per chunk.
        If priority is not provided all arangements of (1,2,3,...) are searched.
        If priority is provided, then only that priority is used.

        For example:

        >>> Chunker.find_chunk_multiple((100,100,100), (10,10,10), 2000).chunk_shape
        (10, 10, 20)

        >>> Chunker.find_chunk_multiple((100,100,100), (10,10,10), 3000).chunk_shape
        (10, 10, 30)

        >>> Chunker.find_chunk_multiple((100,100,100), (10,10,10), 3500).chunk_shape
        (10, 10, 30)

        >>> Chunker.find_chunk_multiple((100,100,100), (10,10,10), 4000).chunk_shape
        (10, 20, 20)

        >>> Chunker.find_chunk_multiple((100,100,100), (10,10,10), 4000, priorities=(3,2,1)).chunk_shape
        (10, 10, 40)

        >>> Chunker.find_chunk_multiple((100,100,100), (25,10,4), 3000).chunk_shape
        (25, 10, 12)

        >>> Chunker.find_chunk_multiple((100,100,100), (25,10,4), 4000).chunk_shape
        (25, 20, 8)
        """

        chunked_data_shape = tuple(
            int(math.floor(d // c)) for d, c in zip(data_shape, chunk_shape)
        )
        items_per_chunk = int(np.prod(chunk_shape))
        if max_item_count < items_per_chunk:
            raise ValueError(
                f"The Maximum item count ({max_item_count}) must be greater than the number of the items in a chunk ({items_per_chunk})"
            )
        max_item_count = max_item_count // items_per_chunk

        if priorities is not None:
            process_priorities = [priorities]
        else:
            last_first = [ii for ii in range(len(data_shape), 0, -1)]
            process_priorities = [
                Shape(
                    last_first[jj] if -jj > -ii else 1
                    for jj in range(0, len(data_shape))
                )
                for ii in range(len(data_shape))
            ]

        final_item_count = 0
        final_chunker = None

        for current_priorities in process_priorities:
            chunker = Chunker.from_max_item_count(
                chunked_data_shape, current_priorities, max_item_count
            )
            memory_chunk_shape = tuple(
                c * cs for c, cs in zip(chunk_shape, chunker.chunk_shape)
            )
            memory_item_count = np.prod(memory_chunk_shape)

            if chunker is None or memory_item_count > final_item_count:
                final_item_count = memory_item_count
                final_chunker = chunker

        assert final_chunker is not None
        return Chunker.from_chunk_shape(
            data_shape=data_shape,
            chunk_shape=tuple(
                c * cs for c, cs in zip(chunk_shape, final_chunker.chunk_shape)
            ),
        )

    def __repr__(self) -> str:
        return f"data: {self.data_shape} p: {self.priorities} cshape: {self.chunk_shape} ccount: {self.chunk_count}"

    def _chunk(self, dimension: int, count: int) -> slice:
        assert count * self.chunk_shape[dimension] < self.data_shape[dimension]
        return slice(
            int(count * self.chunk_shape[dimension]),
            int(
                min(
                    (count + 1) * self.chunk_shape[dimension],
                    self.data_shape[dimension],
                )
            ),
        )

    def chunks(self) -> Generator[Chunk]:
        """
        A generator that yeilds all the chunks covered by this chunker.
        """

        indices = np.zeros((self.n_dims,))
        for chunk_inx in range(self.n_chunks):
            for ii in range(self.n_dims):
                indices[ii] += 1
                if indices[ii] >= self.chunk_count[ii]:
                    indices[ii] = 0
                    continue
                else:
                    break
            yield Chunk([self._chunk(ii, indices[ii]) for ii in range(self.n_dims)])
        return

    def bulk_chunks(self) -> list[Chunk]:
        """
        Returns a minimal set of chunks that covers the space, as cut along
        chunking boundaries.
        If all dinensions have more than one data point, this always returns
        2^ndims chunks. e.g. for data shaped (n,m,o) this will return 2^3 = 8 chunks.
        """
        dims = len(self.data_shape)

        dim_chunks = []
        for ii in range(dims):
            chunk = self._chunk(ii, self.chunk_count[ii] - 1)
            if self.data_shape[ii] > 1:
                dim_chunks.append([slice(0, chunk.start), chunk])
            else:
                dim_chunks.append([slice(0, 1)])

        return [Chunk(i) for i in itertools.product(*dim_chunks)]

    def chunk_for_position(self, position: tuple[int, ...]) -> Chunk:
        return Chunk(
            [
                self._chunk(ii, p // self.chunk_shape[ii])
                for ii, p in enumerate(position)
            ]
        )

    def chunk_for_index(self, index: tuple[int, ...]) -> Chunk:
        return Chunk([self._chunk(ii, jj) for ii, jj in enumerate(index)])

    def edges_of_axis(
        self, index: int, start: int | None = None, end: int | None = None
    ):

        chunk_length = self.chunk_shape[index]
        data_length = self.data_shape[index]
        start = start if start is not None else 0
        end = end if end is not None else data_length

        result = [chunk_length * ii for ii in range(self.chunk_count[index])]
        result = [r for r in result if r >= start and r < end]
        result.append(end)
        return result
