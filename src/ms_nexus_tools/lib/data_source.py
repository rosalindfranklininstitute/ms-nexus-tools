# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
from ms_nexus_tools.lib.dtypes import Any1D, Int1D32, Int1Dp

from contextlib import AbstractContextManager
from typing import Any, Callable, NamedTuple
from abc import abstractmethod
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import numpy as np
import sparse

from .bounds import Chunk, Shape
from .exceptions import UnsupportedDataError


class AxisDensity(Enum):
    CONTINUOUS = 1
    SPARSE = 2


@dataclass
class Axis:
    name: str
    primary_axis: int  # NOTE: Assumption: an axis only defines 1 dimension. Nexus is more poweful than this.
    secondary_axes: list[int]
    density: AxisDensity
    dtype: np.generic
    units: str | None = None


class DataShape(NamedTuple):
    shape: Shape
    density: float


class MultiCOO(NamedTuple):
    coords: np.ndarray[tuple[int, ...], np.dtype[np.int32]]
    signal: Any1D
    axis: list[Any1D]

    def sort(self, shape) -> "MultiCOO":
        # Inspired by sparse.COO
        # See https://github.com/pydata/sparse/blob/main/LICENSE
        # This is the BSD 3-clause license

        linear = np.ravel_multi_index(self.coords, shape)
        if np.all(np.diff(linear) >= 0):
            return self
        order = np.argsort(linear)

        return MultiCOO(
            coords=self.coords[:, order],
            signal=self.signal[order],
            axis=[a[order] for a in self.axis],
        )

    def sum_duplicates(self, shape: Shape, count=False) -> tuple["MultiCOO", Int1Dp]:
        # Inspired by sparse.COO
        # See https://github.com/pydata/sparse/blob/main/LICENSE
        # This is the BSD 3-clause license
        linear: Int1Dp = np.ravel_multi_index(self.coords, shape)
        unique_mask = np.diff(linear) != 0

        counts = np.array([], dtype=np.intp)

        if unique_mask.sum() == len(unique_mask):
            return self, np.ones((len(linear),), dtype=np.intp) if count else counts

        unique_mask = np.append(True, unique_mask)

        coords = self.coords[:, unique_mask]
        (unique_inds,) = np.nonzero(unique_mask)
        if count:
            counts = np.diff(unique_inds)
            counts = np.append(counts, len(linear) - unique_inds[-1])

        return MultiCOO(
            coords=coords,
            signal=np.add.reduceat(self.signal, unique_inds, dtype=self.signal.dtype),
            axis=[
                np.add.reduceat(a, unique_inds, dtype=self.signal.dtype)
                for a in self.axis
            ],
        ), counts


class AbstractDataSource(AbstractContextManager):
    @abstractmethod
    def instrament_metadata(self) -> dict[str, Any]:
        """
        Returns a dictionary of values that will be stored as the instrament metadata.
        """
        pass

    @abstractmethod
    def experiment_metadata(self) -> dict[str, Any]:
        """
        Returns a dictionary of values that will be stored as the experiment metadata.
        """
        pass

    @abstractmethod
    def shape(self) -> DataShape:
        """
        Return the shape of the data.
        """
        pass

    @abstractmethod
    def signal_type(self) -> np.generic:
        """
        Returns the type for data.
        """

    @abstractmethod
    def output_chunks(self) -> dict[str, Shape]:
        """
        Returns the names and chunking priorities of the desired output array.
        For examlpe simple image data (x,y, spectra) with shape (32,32,184000)
        might produce:
        'images':   (1,1,2) -> (32,32,1)
        'spectra':  (2,2,1) -> (1,1,184000)
        """
        pass

    @abstractmethod
    def chunk_read_count(self, memory_chunk: Shape) -> int:
        """
        Returns the number of read operations needed to fill the provided memory chunk.
        """
        pass

    @abstractmethod
    def axis_definitions(self) -> list[Axis]:
        """
        Returns the axis that should be used when storing the data.
        For examlpe simple image data (x,y, spectra):
        axis(0) : Axis('x', 0, [], CONTINUOUS, 'um')
        axis(1) : Axis('y', 1, [], CONTINUOUS, 'um')
        If is it continuous:
        axis(2) : Axis('mz', 2, [], CONTINUOUS, 'mz')
        if it is only peaks:
        axis(2) : Axis('mz', 2, [0,1], SPARSE, 'mz')
        """
        pass

    @abstractmethod
    def continuous_axis_values(self, axis: Axis) -> np.ndarray:
        """
        Returns the values for the specified continuous axis.
        """
        pass

    @abstractmethod
    def sparse_axis_edges(self, axis: Axis) -> np.ndarray:
        """
        Returns the bin edges used to histogram the given sparse axis.
        This is used for generting the output accumulations accros this axis, if required.
        """
        pass

    @abstractmethod
    def output_accumulations(self) -> dict[str, tuple[str, ...]]:
        """
        Returns the names and lists of axis that should be
        accumulated (summed and max).
        For examlpe simple image data (x,y, spectra):
        might produce:
        'total_images':     ('mz') # Accumulate over the spectra
        'total_spectra':    ('x','y') # Accumulate over the images
        """
        pass

    @abstractmethod
    def fill_chunk(
        self,
        memory_chunk: Chunk,
        fill_axis: list[Axis],
        update: Callable[[int], None],
    ) -> np.ndarray | MultiCOO:
        """
        Read data from the source in the region specified by
        memory_chunk and return that data. Also return the data
        any sparse axis.

        Parameters:
        memory_chunk:   The bounds of the data to read.
        fill_axis:      The list of sparce axis to fill.
        update:         A callback to update progress.
                        The total of the progress counter is
                        sum([chunk_read_count(mc) for mc in all_memory_chunks])
        Returns:
        The data from the source, and the data for all the sparse axes, ordered in the same order as in the fill_axis.
        If dense :
        -> return_data.shape == self.shape()
        If sparse there is an extra dimension for storing signal and each sparse axis:
        -> return_data.shape[0:-1] == self.shape() and return_data.shape[-1] = len(fill_axis)+1

        """
        pass
