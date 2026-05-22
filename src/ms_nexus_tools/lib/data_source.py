# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

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


class AbstractDataSource(AbstractContextManager):
    def __init__(self, cbounds: ContainedBounds):
        self.cbound = cbounds

    @abstractmethod
    def output_chunks(self, max_items_per_chunk: int) -> dict[str, Shape]:
        """
        Returns the names and chunking of the desired output array.
        For examlpe simple image data (x,y, spectra) with shape (32,32,184000)
        might produce:
        'images': (32,32,1)
        'spectra': (1,1,184000)
        """
        pass

    @abstractmethod
    def output_accumulations(self) -> dict[str, tuple[int]]:
        """
        Returns the names and lists of axis that should have
        the be accumulated (summed and max) and stored.
        For examlpe simple image data (x,y, spectra):
        might produce:
        'total_images': (2) # Accumulate over the spectra
        'total_spectra': (0,1) # Accumulate over the images
        """
        pass

    @abstractmethod
    def chunk_read_count(self, memory_chunk: Chunk) -> int:
        """
        Returns the number of read operations needed to fill the provided memory chunk.
        """
        pass

    @abstractmethod
    def axis(self) -> GenericAxis:
        """
        Returns the axis that should be used when storing the data.
        """
        pass

    @abstractmethod
    def fill_chunk(
        self,
        memory_chunk: Chunk,
        update: Callable[[int], None],
    ) -> np.ndarray:
        """
        Read data from the source in the region specified by memory_chunk and return that data.

        Parameters:
        memory_chunk:   The bounds of the data to read.
        update:         A callback to update progress.
                        The total of the progress counter is sum([chunk_read_count(mc) for mc in all_memory_chunks])

        """
        pass
