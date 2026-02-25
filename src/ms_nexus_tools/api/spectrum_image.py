from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import bisect
import os

import h5py as h5
from nexusformat.nexus import NXfield, NXdata, nxload

import numpy as np
import matplotlib.pyplot as plt

from icecream import ic

from . import ion
from .api import arg_field, ArgType
from ..lib.chunking import ChunkBounds, ImageBounds
from ..lib.filetypes import DataType


class IndexType(Enum):
    INDEX = "index"
    DISTANCE = "dist"


@dataclass
class ProcessArgs:
    hdf_in_path: Path = arg_field(
        "-i",
        "--input",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        doc="The input file.",
    )
    img_out_path: Path = arg_field(
        "-o",
        "--output",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        doc="The output file.",
    )

    layer: int = arg_field(
        "-l",
        doc="The layer to take an image of. Starting from 0.",
        default=0,
    )

    start_width: float = arg_field(
        "-sw",
        doc="The start position within the width for the spectrum.",
        default=0,
    )

    end_width: float = arg_field(
        "-ew",
        doc="The end position within the width for the spectrum.",
        default=-1,
    )

    start_height: float = arg_field(
        "-sh",
        doc="The start position within the height for the spectrum.",
        default=0,
    )

    end_height: float = arg_field(
        "-eh",
        doc="The end position within the height for the spectrum.",
        default=-1,
    )

    indexing: IndexType = arg_field(
        doc="The value of the start and end fields.",
        choices=[t for t in IndexType],
        default=IndexType.INDEX,
    )

    filetype: DataType = arg_field(
        doc="The type of the input file.",
        choices=[t for t in DataType],
        default=DataType.NEXUS,
    )


def process(args: ProcessArgs):

    match args.filetype:
        case DataType.ION_H5:
            _, image_bounds, image_axis = ion.read_metadata(args.hdf_in_path)
            x_axis = image_axis.x_axis
            y_axis = image_axis.y_axis
            mass_axis = image_axis.mass_axis
        case DataType.ION_VDS:
            assert args.indexing != IndexType.DISTANCE
            x_axis = None
            y_axis = None
            mass_axis = None
            with h5.File(args.hdf_in_path, "r") as vds_in:
                image_bounds = ImageBounds(*vds_in["spectra"].shape)

        case DataType.NEXUS:
            data_root = nxload(args.hdf_in_path)
            data = data_root["entry"]["data"]["signal"]
            image_bounds = ImageBounds(*data.shape)
            x_axis = data_root["entry"]["data"]["x"]
            y_axis = data_root["entry"]["data"]["y"]
            mass_axis = data_root["entry"]["data"]["mass"]
        case _:
            raise NotImplementedError(f"Unimplemented filetype: {args.filetype}")

    width_slice = slice(None, None, None)
    if args.indexing == IndexType.DISTANCE:
        assert x_axis is not None
        assert y_axis is not None
        width_slice = slice(
            bisect.bisect_left(x_axis, args.start_width),
            bisect.bisect_right(x_axis, args.end_width),
            None,
        )
        height_slice = slice(
            bisect.bisect_left(x_axis, args.start_height),
            bisect.bisect_right(x_axis, args.end_height),
            None,
        )

    else:
        width_slice = slice(int(args.start_width), int(args.end_width), None)
        height_slice = slice(int(args.start_height), int(args.end_height), None)
        mass_axis = np.arange(image_bounds.spectrum_length)

    image = np.zeros((image_bounds.spectrum_length,))
    match args.filetype:
        case DataType.ION_H5:
            assert isinstance(image_bounds, ion.IONImageBounds)
            with h5.File(args.hdf_in_path, "r") as hdf_in:
                image = np.array(
                    [
                        np.sum(
                            hdf_in[image_bounds.image_path(args.layer, ss)][
                                width_slice, height_slice
                            ]
                        )
                        for ss in range(image_bounds.spectrum_length)
                    ]
                )
        case DataType.ION_VDS:
            with h5.File(args.hdf_in_path, "r") as vds_in:
                image = np.sum(
                    vds_in["images"][args.layer, width_slice, height_slice, :],
                    axis=(0, 1),
                )

        case DataType.NEXUS:
            data_root = nxload(args.hdf_in_path)
            data = data_root["entry"]["data"]["signal"]
            image = np.sum(data[args.layer, width_slice, height_slice, :], axis=(0, 1))

    _, ext = os.path.splitext(args.img_out_path)
    if ext == ".csv":
        np.savetxt(args.img_out_path, image, delimiter=",")
    else:
        assert mass_axis is not None
        if isinstance(image, NXfield):
            image = np.array(image)
        if not isinstance(mass_axis, np.ndarray):
            mass_axis = np.array(mass_axis)
        fig, ax = plt.subplots()
        fig.suptitle(
            f"showing sum of spectra in ({args.start_width}:{args.end_width}, {args.start_height}:{args.end_height})"
        )
        ax.plot(mass_axis, image)
        ax.set_xlabel("mz" if args.indexing == IndexType.DISTANCE else "index")
        ax.set_ylabel("intensity")
        fig.savefig(args.img_out_path)
