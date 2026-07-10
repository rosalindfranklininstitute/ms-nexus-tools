# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import AbstractContextManager
from typing import Any
from abc import abstractmethod


import numpy as np

from .bounds import Chunk, Shape
from .exceptions import UnsupportedDataError
from .mz_filter import MzFilter
from .normalisation import Accumulator


class AbstractQuerySource(AbstractContextManager):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def shape(self) -> Shape:
        """
        Return the shape of the data.
        """

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
        totals: list[MzFilter],
        chunk: Chunk,
    ) -> None:
        pass

    def accumulated_spectrum(self, accumulator: Accumulator, layer: int) -> np.ndarray:
        raise UnsupportedDataError()

    def accumulated_image(self, accumulator: Accumulator, layer: int) -> np.ndarray:
        raise UnsupportedDataError()
