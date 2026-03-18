from typing import Generator, NamedTuple
from dataclasses import dataclass, field
import math

import numpy as np

from icecream import ic

from .bounds import Chunk, Shape, Bounds


def approximate_gb(int4_count: float) -> float:
    return (int4_count / 1024 / 1024 / 1024) * 4


def approximate_int4_count(gb: float) -> float:
    return (gb / 4.0) * 1024 * 1024 * 1024


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
    shape, the number of entries in the data and, the priority of each axis.
    Lower values of priority will have more values per chunk.
    So that:
    >>> Chunker(data_shape=(100,100), priorities=(1,2), count=10).chunk_shape
    (10, 1)

    and
    >>> Chunker(data_shape=(10,10), priorities=(2,1), count=20).chunk_shape
    (2, 10)
    """

    def __init__(self):
        self.data_shape: Shape
        self.priorities: Shape
        self.n_dims: int

        self.chunk_shape: Shape
        self.chunk_count: Shape
        self.n_chunks: int

    @staticmethod
    def from_item_count(
        data_shape: Shape,
        priorities: Shape,
        min_items_per_chunk: int,
    ) -> "Chunker":
        chunker = Chunker()
        assert len(data_shape) == len(priorities)
        chunker.data_shape = data_shape
        chunker.priorities = priorities
        chunker.n_dims = len(chunker.data_shape)

        chunker.chunk_shape, chunker.chunk_count = chunker._calculate_from_min_count(
            min_items_per_chunk
        )
        chunker.n_chunks = int(np.prod(chunker.chunk_count))
        return chunker

    def _calculate_from_min_count(self, min_items_per_chunk) -> tuple[Shape, Shape]:

        chunk_shape = [1 for _ in self.priorities]

        remaining_count = min_items_per_chunk
        for dimensions, remaining in _count_priorities(self.priorities):
            n_dims = len(dimensions)
            capacity = np.prod([self.data_shape[i] for i in dimensions])
            if capacity > remaining_count:
                dim_data_shape = np.array([self.data_shape[d] for d in dimensions])
                min_dim = np.min(dim_data_shape)
                weightings = dim_data_shape / min_dim

                chunks_per_dim = np.pow(
                    remaining_count / np.prod(weightings), 1 / n_dims
                )

                for d, w in zip(dimensions, weightings):
                    chunk_shape[d] = math.ceil(w * chunks_per_dim)
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
        min_chunk_count: int,
    ) -> "Chunker":
        chunker = Chunker()
        assert len(data_shape) == len(priorities)
        chunker.data_shape = data_shape
        chunker.priorities = priorities
        chunker.n_dims = len(chunker.data_shape)

        chunker.chunk_shape, chunker.chunk_count = chunker._calculate_from_min_chunks(
            min_chunk_count
        )
        chunker.n_chunks = int(np.prod(chunker.chunk_count))
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
        return chunker

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

    def chunk_for_position(self, position: tuple[int, ...]) -> Chunk:
        return Chunk(
            [
                self._chunk(ii, p // self.chunk_shape[ii])
                for ii, p in enumerate(position)
            ]
        )


@dataclass
class ChunkBounds:
    layer: slice
    width: slice
    height: slice
    spectra: slice

    def __hash__(self):
        return hash(
            (
                self.layer.start,
                self.layer.stop,
                self.width.start,
                self.width.stop,
                self.height.start,
                self.height.stop,
                self.spectra.start,
                self.spectra.stop,
            )
        )

    def approximate_size_gb(self) -> float:
        return approximate_gb(
            float(self.layer.stop - self.layer.start)
            * float(self.width.stop - self.width.start)
            * float(self.height.stop - self.height.start)
            * float(self.spectra.stop - self.spectra.start)
        )

    def layer_count(self) -> int:
        return self.layer.stop - self.layer.start

    def layer_width(self) -> int:
        return self.width.stop - self.width.start

    def layer_height(self) -> int:
        return self.height.stop - self.height.start

    def spectrum_length(self) -> int:
        return self.spectra.stop - self.spectra.start

    def layer_range(self) -> range:
        return range(self.layer.start, self.layer.stop)

    def width_range(self) -> range:
        return range(self.width.start, self.width.stop)

    def height_range(self) -> range:
        return range(self.height.start, self.height.stop)

    def spectra_range(self) -> range:
        return range(self.spectra.start, self.spectra.stop)

    def to_bound_dict(self) -> dict[str, int]:
        return dict(
            layer_start=self.layer.start,
            layer_stop=self.layer.stop,
            width_stat=self.width.start,
            width_stop=self.width.stop,
            height_stat=self.height.start,
            height_stop=self.height.stop,
            spectra_stat=self.spectra.start,
            spectra_stop=self.spectra.stop,
        )

    def shape(self) -> tuple[int, int, int, int]:
        return (
            self.layer.stop - self.layer.start,
            self.width.stop - self.width.start,
            self.height.stop - self.height.start,
            self.spectra.stop - self.spectra.start,
        )

    def count(self) -> int:
        return int(np.prod(self.shape()))


@dataclass
class ImageBounds:
    layer_count: int
    layer_width: int
    layer_height: int
    spectrum_length: int
    shape: tuple[int, int, int, int] = field(init=False)

    def __post_init__(self):
        self.shape = (
            self.layer_count,
            self.layer_width,
            self.layer_height,
            self.spectrum_length,
        )

    def to_bounds(self) -> Bounds:
        return Bounds(self.shape)


@dataclass
class MemoryInfo:
    total_gb: float
    max_chunk_gb: float
    min_chunk_count: int

    @staticmethod
    def calculate(
        chunk_count_min: int,
        gb_max: float | None,
        processors: int,
        bounds: Bounds,
    ) -> "MemoryInfo":
        total_gb = approximate_gb(bounds.total)

        assert processors >= 1

        if chunk_count_min <= 0:
            chunk_count_min = 1

        chunk_gb = total_gb / chunk_count_min
        if gb_max is not None:
            assert gb_max > 0.0
            if (chunk_gb * processors) > gb_max:
                chunk_count = math.ceil(total_gb / (gb_max / processors))
            else:
                chunk_count = chunk_count_min
        else:
            chunk_count = chunk_count_min
        chunk_gb = total_gb / chunk_count

        return MemoryInfo(
            total_gb=total_gb, max_chunk_gb=chunk_gb, min_chunk_count=chunk_count
        )


def calculate_chunks_from_memory(
    chunk_count_min: int,
    gb_max: float | None,
    processors: int,
    bounds: Bounds,
) -> tuple[list[ChunkBounds], list[ChunkBounds], MemoryInfo]:

    memory_info = MemoryInfo.calculate(chunk_count_min, gb_max, processors, bounds)

    spectra_chunker = Chunker.from_min_chunks(
        data_shape=bounds.shape,
        priorities=(1, 2, 2, 3),
        min_chunk_count=max(1, memory_info.min_chunk_count),
    )
    spectra_chunks = [ChunkBounds(*chunk) for chunk in spectra_chunker.chunks()]

    image_chunker = Chunker.from_min_chunks(
        data_shape=bounds.shape,
        priorities=(1, 3, 3, 2),
        min_chunk_count=max(1, memory_info.min_chunk_count),
    )
    image_chunks = [ChunkBounds(*chunk) for chunk in image_chunker.chunks()]

    return spectra_chunks, image_chunks, memory_info


if __name__ == "__main__":
    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(1, 2, 3), min_chunk_count=10000
    )

    n = 500
    for i in range(n):
        chunks = [c for c in chunker.chunks()]
        tuple_chunks = [c for c in chunker.chunks_tuple()]
        list_chunks = [c for c in chunker.chunks_list()]
        if ((100 * i) / n) % 10 == 0:
            print(i, n)
