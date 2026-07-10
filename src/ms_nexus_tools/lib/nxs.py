# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
import h5py

from typing import Any, NamedTuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import numpy.typing as npt

from .bounds import Shape, Chunk
from .contained_bounds import ContainedBounds
from .chunker import Chunker
from .mz_filter import Accumulator

from nexusformat.nexus import NXentry, NXfield, NXdata, nxload
from nexusformat.nexus.tree import (
    NXgroup,
    NXsubentry,
    NXinstrument,
    NXprocess,
    NXparameters,
    NXlinkfield,
)


@dataclass
class NxAxis:
    name: str
    indices: list[int]
    field: NXfield

    @staticmethod
    def create(
        values,
        name: str,
        indices: list[int],
        unit: str | None = None,
        chunk_shape: Shape | None = None,
    ) -> "NxAxis":
        field = NXfield(values, name=name, chunks=chunk_shape)
        if unit is not None:
            field.attrs["unit"] = unit

        return NxAxis(name=name, indices=indices, field=field)

    @staticmethod
    def create_empty(
        name: str,
        indices: list[int],
        dtype: npt.DTypeLike,
        shape: Shape,
        compression: str | None = None,
        compression_opts: Any = None,
        chunks: Shape | None = None,
        unit: str | None = None,
        fillvalue: int | float | None = None,
    ) -> "NxAxis":
        field = NXfield(
            name=name,
            dtype=dtype,
            shape=shape,
            compression=compression,
            compression_opts=compression_opts,
            chunks=chunks,
            fillvalue=fillvalue,
        )
        if unit is not None:
            field.attrs["unit"] = unit

        return NxAxis(name=name, indices=indices, field=field)

    def add_to_group(self, group: NXdata) -> None:
        group.attrs[f"{self.name}_indices"] = self.indices
        group[self.name] = self.field

    def __len__(self) -> int:
        return len(self.field)

    def __getitem__(self, index):
        return self.field[index]

    def copy_with_incremented_indices(self, inc: int) -> "NxAxis":
        return NxAxis(
            name=self.name,
            indices=[i + inc for i in self.indices],
            field=self.field,
        )


class NxAxes(list[list[NxAxis]]):
    def default_list(self) -> list[str]:
        return [v[0].name for v in self]

    def list_all(self) -> list[NxAxis]:
        results = []
        for v in self:
            results.extend(v)
        return results

    def add_to_group(self, group: NXdata) -> None:
        group.attrs["axes"] = self.default_list()
        for ax in self.list_all():
            ax.add_to_group(group)


class FieldOptions(NamedTuple):
    compression: Any
    compression_opts: int | None
    max_bytes_per_chunk: int
    shuffle: bool


class NexusFile:
    def __init__(self, filename: Path, mode: str = "r", locking=None):
        self.filename = filename

        self._mode = mode
        self._file = nxload(filename, mode, locking=locking)

        if mode == "w" or mode == "w-" or mode == "x":
            self._file["entry"] = NXentry()
            self.root = self._file["entry"]
            self.root["instrument"] = NXinstrument()
            self.root["experiment"] = NXparameters()
        else:
            self.root = self._file["entry"]

    def close(self) -> None:
        self._file.nxfile.close()

    def as_context(self) -> h5py.File:
        return self._file.nxfile

    def _get_instrument(self) -> NXinstrument:
        return self.root["instrument"]

    def _set_instrument(self, value: NXinstrument) -> None:
        assert isinstance(value, NXinstrument)
        self.root["instrument"] = value

    instrument = property(
        _get_instrument,
        _set_instrument,
        None,
        "The instrument group",
    )

    def _get_experiment(self) -> NXparameters:
        return self.root["experiment"]

    def _set_experiment(self, value: NXparameters) -> None:
        assert isinstance(value, NXparameters)
        self.root["experiment"] = value

    experiment = property(
        _get_experiment,
        _set_experiment,
        None,
        "The experiment group",
    )

    def link_data(self, data_path: str) -> None:
        assert self._mode != "r"

        signal_path = f"{data_path}/signal"
        assert signal_path in self.root

        axes_path = f"{data_path}/axes"
        errors_path = f"{data_path}/errors"
        weights_path = f"{data_path}/weights"

        self.root["data"] = NXdata(
            NXlinkfield(signal=self.root[signal_path]),
            axes=self.root.get(axes_path, None),
            errors=self.root.get(errors_path, None),
            weights=self.root.get(weights_path, None),
        )

    def set_data(
        self,
        field: NXfield,
        axes: NxAxes,
        errors: NXfield | None = None,
        weights: NXfield | None = None,
    ) -> NXdata:
        self.root["data"] = NXdata(field, errors=errors, weight=weights)
        axes.add_to_group(self.root["data"])
        return self.root["data"]

    def create_subentry(
        self,
        name: str,
        field: NXfield,
        axes: NxAxes,
        errors: NXfield | None = None,
        weights: NXfield | None = None,
    ) -> NXsubentry:
        self.root[name] = NXsubentry(
            NXdata(signal=field, errors=errors, weight=weights),
        )
        axes.add_to_group(self.root[name]["data"])
        return self.root[name]


def create_field(
    dtype: npt.DTypeLike | None = None,
    shape: Shape | None = None,
    compression: str | None = None,
    compression_opts: Any = None,
    chunks: Shape | bool | None = None,
    value: Any = None,
    **kwargs,
) -> NXfield:
    assert value is not None or (dtype is not None and shape is not None)
    return NXfield(
        value=value,
        dtype=dtype,
        shape=shape,
        compression=compression,
        compression_opts=compression_opts,
        chunks=chunks,
        **kwargs,
    )


def create_chunked_subentry(
    nxs: NexusFile,
    name: str,
    field_options: FieldOptions,
    data_shape: Shape,
    priorities: Shape,
    axes: NxAxes,
) -> tuple[Chunker, NXsubentry]:
    assert len(data_shape) == len(priorities)
    assert len(data_shape) == len(axes)

    chunks = Chunker.from_max_item_count(
        data_shape=data_shape,
        priorities=priorities,
        items_per_chunk=field_options.max_bytes_per_chunk,
    )
    subentry = nxs.create_subentry(
        name,
        create_field(
            dtype="int32",
            shape=data_shape,
            compression=field_options.compression,
            compression_opts=field_options.compression_opts,
            chunks=chunks.chunk_shape,
            shuffle=field_options.shuffle,
        ),
        axes=axes,
    )
    return chunks, subentry


def create_group(
    *args,
    **kwargs,
) -> NXgroup:
    return NXgroup(
        *args,
        **kwargs,
    )


def write_from_data(
    out_path: Path,
    data: np.ndarray,
    x_microns: float,
    y_microns: float,
    mass: np.ndarray,
    mass_unit: str = "mz",
    compression: str = "gzip",
    compression_level: int = 4,
) -> NexusFile:
    axes = NxAxes(
        [
            [
                NxAxis.create(
                    values=np.arange(1, data.shape[0] + 1, 1.0),
                    name="layer",
                    indices=[1],
                ),
            ],
            [
                NxAxis.create(
                    values=np.arange(0, data.shape[1], 1.0) * x_microns,
                    name="x",
                    unit="micron",
                    indices=[2],
                ),
            ],
            [
                NxAxis.create(
                    values=np.arange(0, data.shape[2], 1.0) * y_microns,
                    name="y",
                    unit="micron",
                    indices=[2],
                ),
            ],
            [NxAxis.create(values=mass, name="mass", unit=mass_unit, indices=[3])],
        ],
    )

    nxs = NexusFile(out_path, mode="w")

    nxs.root["process"] = NXprocess()
    nxs.root["process"].attrs["name"] = "Nexus From data"
    nxs.root["process"].input = NXparameters(
        layers=data.shape[0],
        width=data.shape[1],
        height=data.shape[2],
        spectrum=data.shape[3],
    )

    nxs.set_data(
        create_field(
            dtype="int32",
            shape=data.shape,
            compression=compression,
            compression_opts=compression_level,
        ),
        axes=axes,
    )
    nxs.root.data.signal[:] = data[:]

    return nxs


def create_standard_file(
    data_shape: Shape,
    out_chunk: Chunk,
    out_path: Path,
    axes: NxAxes | None = None,
    field_options=FieldOptions(  # noqa: B008
        compression="gzip",
        compression_opts=4,
        max_bytes_per_chunk=1024 * 1024 * 8,
        shuffle=False,
    ),
) -> tuple[NexusFile, ContainedBounds, tuple[Chunker, ...]]:
    cbounds = ContainedBounds.from_chunk(outer_shape=data_shape, inner_chunk=out_chunk)

    if axes is None:
        axes = NxAxes(
            [
                [
                    NxAxis.create(
                        name="layer",
                        values=out_chunk.arange(0),
                        indices=[0],
                    ),
                ],
                [
                    NxAxis.create(
                        name="x",
                        values=out_chunk.arange(1),
                        indices=[1],
                    ),
                ],
                [
                    NxAxis.create(
                        name="y",
                        values=out_chunk.arange(2),
                        indices=[2],
                    ),
                ],
                [
                    NxAxis.create(
                        name="mass",
                        values=out_chunk.arange(3),
                        indices=[3],
                    ),
                ],
            ],
        )
    acc_count = len(Accumulator)
    total_axes = [
        [
            NxAxis.create(
                name="accumulator",
                values=[t.value for t in Accumulator],
                indices=[0],
            ),
        ],
    ]
    for axis_set in axes:
        inner_list = [ax.copy_with_incremented_indices(1) for ax in axis_set]
        total_axes.append(inner_list)

    nxs = NexusFile(out_path, mode="w")
    spectra_chunks, spectra = create_chunked_subentry(
        nxs,
        "spectra",
        field_options=field_options,
        data_shape=cbounds.inner_shape,
        priorities=(3, 2, 2, 1),
        axes=axes,
    )

    total_spectra_chunks, total_spectra = create_chunked_subentry(
        nxs,
        "total_spectra",
        field_options=field_options,
        data_shape=(acc_count, cbounds.inner_shape[0], cbounds.inner_shape[3]),
        priorities=(3, 2, 1),
        axes=NxAxes([total_axes[0], total_axes[1], total_axes[4]]),
    )

    image_chunks, images = create_chunked_subentry(
        nxs,
        "images",
        field_options=field_options,
        data_shape=cbounds.inner_shape,
        priorities=(3, 1, 1, 2),
        axes=axes,
    )

    total_image_chunks, total_images = create_chunked_subentry(
        nxs,
        "total_images",
        field_options=field_options,
        data_shape=(
            acc_count,
            cbounds.inner_shape[0],
            cbounds.inner_shape[1],
            cbounds.inner_shape[2],
        ),
        priorities=(3, 2, 1, 1),
        axes=NxAxes([total_axes[0], total_axes[1], total_axes[2], total_axes[3]]),
    )
    return (
        nxs,
        cbounds,
        (spectra_chunks, total_spectra_chunks, image_chunks, total_image_chunks),
    )
