from dataclasses import dataclass
from pathlib import Path
import numpy as np
import shutil
import os

from nexusformat.nexus import NXfield, NXdata, nxload
from nexusformat.nexus.tree import (
    NXentry,
    NXinstrument,
    NXprocess,
    NXparameters,
)


@dataclass
class ImageAxis:
    layer_axis: NXfield
    x_axis: NXfield
    y_axis: NXfield
    mass_axis: NXfield

    def as_list(self):
        return [self.layer_axis, self.x_axis, self.y_axis, self.mass_axis]


def write_nxs(
    out_path: Path,
    data: np.ndarray,
    x_microns: float,
    y_microns: float,
    mass: np.ndarray,
    mass_unit: str = "mz",
    compression: str = "gzip",
    compression_level: int = 4,
):
    axis = ImageAxis(
        layer_axis=NXfield(np.arange(1, data.shape[0] + 1, 1.0), name="layer"),
        x_axis=NXfield(
            np.arange(0, data.shape[1], 1.0) * x_microns, name="x", unit="micron"
        ),
        y_axis=NXfield(
            np.arange(0, data.shape[2], 1.0) * y_microns, name="y", unit="micron"
        ),
        mass_axis=NXfield(mass, name="mass", unit=mass_unit),
    )

    entry = NXentry()
    if os.path.exists(out_path):
        os.remove(out_path)
    entry.save(out_path)

    entry["instrument"] = NXinstrument()

    entry["process"] = NXprocess()
    entry["process"].attrs["name"] = "Nexus From data"
    entry["process"].input = NXparameters(
        layers=data.shape[0],
        width=data.shape[1],
        height=data.shape[2],
        spectrum=data.shape[3],
    )

    entry["data"] = NXdata(
        NXfield(
            dtype="int32",
            shape=data.shape,
            compression=compression,
            compression_opts=compression_level,
        ),
        axis.as_list(),
    )
    entry.data.signal[:] = data[:]
