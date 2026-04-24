from contextlib import AbstractContextManager
from typing import Any, Callable
from abc import abstractmethod

from pathlib import Path

import numpy as np

from .filter import Filter, Accumulator
from .bounds import ContainedBounds, Chunk, Shape
from .nxs import GenericAxis


class UnsupportedDataError(RuntimeError):
    pass


class AbstractQuerySource(AbstractContextManager):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def shape(self) -> Shape:
        pass

    @abstractmethod
    def mass_values(self) -> np.ndarray[tuple[int], Any]:
        pass

    @abstractmethod
    def instrament_metadata(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def experiment_metadata(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def fill_filters(
        self,
        layer: int,
        bins: list[int],
        xy: list[tuple[int, int]],
        totals: list[Filter],
        chunk: Chunk,
    ):
        pass

    def accumulated_spectrum(self, accumulator: Accumulator, layer: int) -> np.ndarray:
        raise UnsupportedDataError()

    def accumulated_image(self, accumulator: Accumulator, layer: int) -> np.ndarray:
        raise UnsupportedDataError()


class AbstractDataSource(AbstractContextManager):
    def __init__(self, cbounds: ContainedBounds):
        self.cbound = cbounds

    @abstractmethod
    def memory_chunk_priorities(self) -> Shape:
        pass

    @abstractmethod
    def chunk_read_count(self, memory_chunk: Chunk) -> int:
        pass

    @abstractmethod
    def axis(self) -> GenericAxis:
        pass

    @abstractmethod
    def fill_chunk(
        self,
        layer: int,
        memory_chunk: Chunk,
        totals: list[Filter],
        update: Callable[[int], None],
    ) -> np.ndarray:
        pass
