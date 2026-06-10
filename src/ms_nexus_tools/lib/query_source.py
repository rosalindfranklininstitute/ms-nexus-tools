# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import AbstractContextManager
from typing import Any, Callable
from abc import abstractmethod

from pathlib import Path

import numpy as np

from .bounds import Chunk, Shape
from .exceptions import UnsupportedDataError
from .filter import Filter
from .normalisation import Accumulator


class AbstractQuerySource(AbstractContextManager):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def shape(self) -> Shape:
        """
        Return the shape of the data.
        """
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
