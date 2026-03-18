from typing import Any, Self, NamedTuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import shutil
import os

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
        if unit is not None:
            field = NXfield(values, name=name)
        else:
            field = NXfield(values, name=name, unit=unit)
        return Axis(name=name, indices=indices, field=field)

    def add_to_group(self, group: NXdata):
        group.attrs[f"{self.name}_indices"] = self.indices
        group[self.name] = self.field

    def __len__(self) -> int:
        return len(self.field)

    def __getitem__(self, index):
        return self.field[index]


class GenericAxis(list[list[Axis]]):
    def default_list(self) -> list[str]:
        return [v[0].name for v in self]

    def list_all(self) -> list[Axis]:
        results = []
        for v in self:
            results.extend(v)
        return results

    def add_to_group(self, group: NXdata):
        group.attrs["axes"] = self.default_list()
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
    shape: tuple[int, ...] | None = None,
    compression: str | None = None,
    compression_opts: Any = None,
    chunks: tuple[int, ...] | bool | None = None,
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
