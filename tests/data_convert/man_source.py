from pathlib import Path
import json
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
import numpy.typing as npt

from ms_nexus_tools.lib.bounds import Chunk, Shape
from ms_nexus_tools.lib.data_source import (
    AbstractDataSource,
    Axis,
    MultiCOO,
    DataShape,
    AxisDensity,
)
from ms_nexus_tools.lib.dtypes import Int1D32

import h5py
from icecream import ic


class ManData:
    def __init__(self):
        self.colors: dict[str, dict[str, list]]
        self.men: list[dict[str, np.ndarray]]

        with open(Path(__file__).parent / "colors.json", "r") as fle:
            data = json.load(fle)

        self.colors = data

        self.men = []
        for ii in range(4):
            man_pos = []
            man_int = []
            with open(Path(__file__).parent / f"man{ii + 1}.txt") as fle:
                for row, line in enumerate(fle):
                    for col, colour in enumerate(line.strip()):
                        colour_mz = self.colors[colour]["mz"]
                        colour_int = self.colors[colour]["int"]
                        man_pos.extend([[col, row, mz] for mz in colour_mz])
                        man_int.extend(colour_int)

            self.men.append(dict(pos=np.array(man_pos).T, int=np.array(man_int)))

        self.total_pos = self.men[0]["pos"]
        self.total_int = self.men[0]["int"]
        for ii, man in enumerate(self.men):
            if ii == 0:
                continue
            pos = man["pos"]
            pos[2, :] += ii * 60
            self.total_pos = np.concatenate([self.total_pos, pos], axis=1)
            self.total_int = np.concatenate([self.total_int, man["int"]])

        self.shape = Shape([8, 8, 240])
        self.dense = np.zeros(self.shape)
        count = 0
        for pos, intensity in zip(self.total_pos.T, self.total_int):
            self.dense[*pos] = intensity

        self.density = self.total_int.shape[0] / np.prod(self.shape)


class ManSource(AbstractDataSource):
    def __init__(
        self,
        man_data: ManData,
        supplimentary_axes: list[Axis] = [],
        mz_binning=1,
    ):

        self.man_data = man_data
        self.axes: dict[str, Axis] = dict(
            x=Axis("x", 0, [], AxisDensity.CONTINUOUS, np.int16, "m"),
            y=Axis("y", 1, [], AxisDensity.CONTINUOUS, np.int16, "m"),
            mz=Axis("mz", 2, [], AxisDensity.CONTINUOUS, np.int16, "mz"),
        )
        for axis in supplimentary_axes:
            self.axes[axis.name] = axis

        self.axis_order: list[str] = []
        axis_primary: list[int] = []
        self.any_sparse = False
        for key, axis in self.axes.items():
            assert key == axis.name
            axis_primary.append(axis.primary_axis)
            self.axis_order.append(axis.name)
            self.any_sparse |= axis.density == AxisDensity.SPARSE
        order = np.argsort(axis_primary)
        self.axis_order = [self.axis_order[oo] for oo in order]
        self.mz_binning = mz_binning

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def instrament_metadata(self) -> dict[str, Any]:
        """
        Returns a dictionary of values that will be stored as the instrament metadata.
        """
        return {}

    def experiment_metadata(self) -> dict[str, Any]:
        """
        Returns a dictionary of values that will be stored as the experiment metadata.
        """
        return {}

    def shape(self) -> DataShape:
        """
        Return the shape of the data.
        """
        if self.any_sparse:
            shape = [0, 0, 0]
            for ax in self.axes.values():
                if shape[ax.primary_axis] == 0:
                    ss = self.man_data.shape[ax.primary_axis]
                    if ax.density == AxisDensity.SPARSE:
                        shape[ax.primary_axis] = ss // self.mz_binning
                    else:
                        shape[ax.primary_axis] = ss
            return DataShape(Shape(shape), self.man_data.density)
            # return DataShape(self.man_data.shape, self.man_data.density)
        else:
            return DataShape(self.man_data.shape, 1.0)

    def signal_type(self) -> npt.DTypeLike:
        """
        Returns the type for data.
        """
        return np.int16

    def output_chunks(self) -> dict[str, Shape]:
        """
        Returns the names and chunking priorities of the desired output array.
        For examlpe simple image data (x,y, spectra) with shape (32,32,184000)
        might produce:
        'images':   (1,1,2) -> (32,32,1)
        'spectra':  (2,2,1) -> (1,1,184000)
        """
        return dict(images=(1, 1, 2), spectra=(2, 2, 1))

    def chunk_read_count(self, memory_chunk: Shape) -> int:
        """
        Returns the number of read operations needed to fill the provided memory chunk.
        """
        return np.prod(memory_chunk[0:2])

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
        return [self.axes[ax_name] for ax_name in self.axis_order]

    def continuous_axis_values(self, axis: Axis) -> np.ndarray:
        """
        Returns the values for the specified continuous axis.
        """
        if axis.name not in self.axes:
            raise ValueError(f"Unknown axis requested: {axis.name}")
        elif self.axes[axis.name].density != AxisDensity.CONTINUOUS:
            raise ValueError(f"Unknown continuous axis requested: {axis.name}")
        else:
            return np.arange(self.man_data.shape[axis.primary_axis])

    def sparse_axis_edges(self, axis: Axis) -> np.ndarray:
        """
        Returns the bin edges used to histogram the given sparse axis.
        This is used for generting the output accumulations accros this axis, if required.
        """
        if axis.name not in self.axes:
            raise ValueError(f"Unknown axis requested: {axis.name}")
        elif self.axes[axis.name].density != AxisDensity.SPARSE:
            raise ValueError(f"Unknown sparse axis requested: {axis.name}")
        else:
            return np.arange(
                0,
                self.man_data.shape[axis.primary_axis] + self.mz_binning,
                self.mz_binning,
            )

    def output_accumulations(self) -> dict[str, tuple[str, ...]]:
        """
        Returns the names and lists of axis that should be
        accumulated (summed and max).
        For examlpe simple image data (x,y, spectra):
        might produce:
        'total_images':     ('mz') # Accumulate over the spectra
        'total_spectra':    ('x','y') # Accumulate over the images
        """
        return dict(total_image=("mz",), total_spectra=("x", "y"))

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
        if self.any_sparse:
            mask = np.full((self.man_data.total_int.shape[0],), True)

            # TODO: should this be moved into the data_converter? It is a standard part of all the sparse converters.
            pos = self.man_data.total_pos[:, :]
            edges = self.sparse_axis_edges(fill_axis[0]) + 0.1
            labels = np.searchsorted(edges, pos[2, :])
            highest_mz = len(edges) - 1
            labels[labels == highest_mz] = highest_mz
            pos[2, :] = labels
            for ii in range(3):
                mask &= (pos[ii, :] >= memory_chunk[ii].start) & (
                    pos[ii, :] < memory_chunk[ii].stop
                )
            return MultiCOO(
                pos[:, mask],
                self.man_data.total_int[mask],
                [self.man_data.total_pos[2, mask]],
            )
        else:
            return self.man_data.dense[*memory_chunk]


if __name__ == "__main__":
    man_data = ManData()

    man_source = ManSource(man_data)
    ic(man_source.axis_order)
    ic(man_source.any_sparse)

    man_source = ManSource(
        man_data, [Axis("time", 0, [], AxisDensity.CONTINUOUS, np.int16)]
    )
    ic(man_source.axis_order)
    ic(man_source.any_sparse)

    man_source = ManSource(
        man_data, [Axis("mz", 2, [1, 2], AxisDensity.SPARSE, np.int16)]
    )
    ic(man_source.axis_order)
    ic(man_source.any_sparse)
