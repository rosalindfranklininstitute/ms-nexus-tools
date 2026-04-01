from typing import Any, Self, NamedTuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .bounds import Shape, ContainedBounds, Chunk
from .chunking import Chunker
from .filter import Accumulator

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
class Axis:
    name: str
    indices: list[int]
    field: NXfield

    @staticmethod
    def create(values, name: str, indices: list[int], unit: str | None = None):
        field = NXfield(values, name=name)
        if unit is not None:
            field.attrs["unit"] = unit

        return Axis(name=name, indices=indices, field=field)

    def add_to_group(self, group: NXdata):
        group.attrs[f"{self.name}_indices"] = self.indices
        group[self.name] = self.field

    def __len__(self) -> int:
        return len(self.field)

    def __getitem__(self, index):
        return self.field[index]

    def copy_with_incremented_indices(self, inc: int) -> "Axis":
        return Axis(
            name=self.name, indices=[i + inc for i in self.indices], field=self.field
        )


class GenericAxis(list[list[Axis]]):
    def default_list(self) -> list[str]:
        return [v[0].name for v in self]

    def list_all(self) -> list[Axis]:
        results = []
        for v in self:
            results.extend(v)
        return results

    def add_to_group(self, group: NXdata):
        group.attrs["axes"] = ",".join(self.default_list())
        for ax in self.list_all():
            ax.add_to_group(group)


class NexusFile:
    def __init__(self, filename: Path, mode: str = "r"):
        self.filename = filename

        self._mode = mode
        self._file = nxload(filename, mode)

        if mode == "w" or mode == "w-" or mode == "x":
            self._file["entry"] = NXentry()
            self.root = self._file["entry"]
            self.root["instrument"] = NXinstrument()
        else:
            self.root = self._file["entry"]

    def as_context(self):
        return self._file.nxfile

    def _get_instrument(self):
        return self.root["instrument"]

    def _set_instrument(self, value: NXinstrument):
        assert isinstance(value, NXinstrument)
        self.root["instrument"] = value

    instrument = property(
        _get_instrument, _set_instrument, None, "The instrument group"
    )

    def link_data(self, data_path: str):
        assert self._mode != "r"

        signal_path = f"{data_path}/signal"
        assert signal_path in self.root

        axes_path = f"{data_path}/axes"
        errors_path = f"{data_path}/errors"
        weights_path = f"{data_path}/weights"

        def entry_or_none(path):
            self.root[path] if path in self.root else None

        self.root["data"] = NXdata(
            NXlinkfield(signal=self.root[signal_path]),
            axes=entry_or_none(axes_path),
            errors=entry_or_none(errors_path),
            weights=entry_or_none(weights_path),
        )

    def set_data(
        self,
        field: NXfield,
        axes: GenericAxis,
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
        axes: GenericAxis,
        errors: NXfield | None = None,
        weights: NXfield | None = None,
    ) -> NXsubentry:
        self.root[name] = NXsubentry(
            NXdata(signal=field, errors=errors, weight=weights)
        )
        axes.add_to_group(self.root[name]["data"])
        return self.root[name]


def create_field(
    dtype: str | None = None,
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
    min_items_per_chunk: int,
    memory_shape: Shape,
    data_shape: Shape,
    priorities: Shape,
    axes: GenericAxis,
) -> tuple[Chunker, NXsubentry]:
    assert len(data_shape) == len(priorities)
    assert len(data_shape) == len(axes)

    chunks = Chunker.from_item_count(
        data_shape=memory_shape,
        priorities=priorities,
        min_items_per_chunk=min_items_per_chunk,
    )
    subentry = nxs.create_subentry(
        name,
        create_field(
            dtype="int32",
            shape=data_shape,
            compression="gzip",
            compression_opts=4,
            chunks=chunks.chunk_shape,
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
    axes = GenericAxis(
        [
            [
                Axis.create(
                    values=np.arange(1, data.shape[0] + 1, 1.0),
                    name="layer",
                    indices=[1],
                )
            ],
            [
                Axis.create(
                    values=np.arange(0, data.shape[1], 1.0) * x_microns,
                    name="x",
                    unit="micron",
                    indices=[2],
                )
            ],
            [
                Axis.create(
                    values=np.arange(0, data.shape[2], 1.0) * y_microns,
                    name="y",
                    unit="micron",
                    indices=[2],
                )
            ],
            [Axis.create(values=mass, name="mass", unit=mass_unit, indices=[3])],
        ]
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
    axes: GenericAxis | None = None,
    min_items_per_chunk: int = 46000,
):
    cbounds = ContainedBounds.from_chunk(outer_shape=data_shape, inner_chunk=out_chunk)

    if axes is None:
        axes = GenericAxis(
            [
                [
                    Axis.create(
                        name="layer",
                        values=out_chunk.arange(0),
                        indices=[0],
                    )
                ],
                [
                    Axis.create(
                        name="x",
                        values=out_chunk.arange(1),
                        indices=[1],
                    )
                ],
                [
                    Axis.create(
                        name="y",
                        values=out_chunk.arange(2),
                        indices=[2],
                    )
                ],
                [
                    Axis.create(
                        name="mass",
                        values=out_chunk.arange(3),
                        indices=[3],
                    )
                ],
            ]
        )
    acc_count = len(Accumulator)
    total_axes = [
        [
            Axis.create(
                name="accumulator",
                values=[t.value for t in Accumulator],
                indices=[0],
            )
        ]
    ]
    for axis_set in axes:
        inner_list = []
        for ax in axis_set:
            inner_list.append(ax.copy_with_incremented_indices(1))
        total_axes.append(inner_list)

    nxs = NexusFile(out_path, mode="w")
    with nxs.as_context():
        spectra_chunks, spectra = create_chunked_subentry(
            nxs,
            "spectra",
            min_items_per_chunk=min_items_per_chunk,
            memory_shape=cbounds.outer_shape,
            data_shape=cbounds.inner_shape,
            priorities=(3, 2, 2, 1),
            axes=axes,
        )

        total_spectra_chunks, total_spectra = create_chunked_subentry(
            nxs,
            "total_spectra",
            min_items_per_chunk=min_items_per_chunk,
            memory_shape=(acc_count, cbounds.inner_shape[0], cbounds.inner_shape[3]),
            data_shape=(acc_count, cbounds.inner_shape[0], cbounds.inner_shape[3]),
            priorities=(3, 2, 1),
            axes=GenericAxis([total_axes[0], total_axes[1], total_axes[4]]),
        )

        image_chunks, images = create_chunked_subentry(
            nxs,
            "images",
            min_items_per_chunk=min_items_per_chunk,
            memory_shape=cbounds.outer_shape,
            data_shape=cbounds.inner_shape,
            priorities=(3, 1, 1, 2),
            axes=axes,
        )

        total_image_chunks, total_images = create_chunked_subentry(
            nxs,
            "total_images",
            min_items_per_chunk=min_items_per_chunk,
            memory_shape=(
                acc_count,
                cbounds.inner_shape[0],
                cbounds.inner_shape[1],
                cbounds.inner_shape[2],
            ),
            data_shape=(
                acc_count,
                cbounds.inner_shape[0],
                cbounds.inner_shape[1],
                cbounds.inner_shape[2],
            ),
            priorities=(3, 2, 1, 1),
            axes=GenericAxis(
                [total_axes[0], total_axes[1], total_axes[2], total_axes[3]]
            ),
        )
    return (
        cbounds,
        (spectra_chunks, total_spectra_chunks, image_chunks, total_image_chunks),
    )
