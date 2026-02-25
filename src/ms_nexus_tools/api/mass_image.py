from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import bisect
import os

import h5py as h5
from nexusformat.nexus import NXfield, NXdata, nxload

import numpy as np
from matplotlib.image import imsave

from icecream import ic

from . import ion
from .api import arg_field, ArgType
from ..lib.chunking import ChunkBounds, ImageBounds
from ..lib.filetypes import DataType


class IndexType(Enum):
    INDEX = "index"
    MASS = "mass"


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

    start: float = arg_field(
        "-s",
        doc="The start position within the spectrum for the image.",
        default=0,
    )

    end: float = arg_field(
        "-e",
        doc="The end position within the spectrum for the image.",
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
            mass_axis = image_axis.mass_axis
        case DataType.ION_VDS:
            assert args.indexing != IndexType.MASS
            mass_axis = None
            with h5.File(args.hdf_in_path, "r") as vds_in:
                image_bounds = ImageBounds(*vds_in["spectra"].shape)

        case DataType.NEXUS:
            data_root = nxload(args.hdf_in_path)
            data = data_root["entry"]["data"]["signal"]
            image_bounds = ImageBounds(*data.shape)
            mass_axis = data_root["entry"]["data"]["mass"]
        case _:
            raise NotImplementedError(f"Unimplemented filetype: {args.filetype}")

    spectra_slice = slice(None, None, None)
    if args.indexing == IndexType.MASS:
        assert mass_axis is not None
        spectra_slice = slice(
            bisect.bisect_left(mass_axis, args.start),
            bisect.bisect_right(mass_axis, args.end),
            None,
        )
    else:
        spectra_slice = slice(int(args.start), int(args.end), None)

    image = np.zeros((image_bounds.layer_width, image_bounds.layer_height))
    match args.filetype:
        case DataType.ION_H5:
            assert isinstance(image_bounds, ion.IONImageBounds)
            with h5.File(args.hdf_in_path, "r") as hdf_in:
                image = np.array(
                    [
                        np.sum(
                            hdf_in[image_bounds.spectrum_path(args.layer, ww, hh)][
                                spectra_slice
                            ],
                        )
                        for ww in range(image_bounds.layer_width)
                        for hh in range(image_bounds.layer_width)
                    ]
                )

        case DataType.ION_VDS:
            with h5.File(args.hdf_in_path, "r") as vds_in:
                image = np.sum(
                    vds_in["spectra"][args.layer, :, :, spectra_slice], axis=2
                )

        case DataType.NEXUS:
            data_root = nxload(args.hdf_in_path)
            data = data_root["entry"]["data"]["signal"]
            image = np.sum(data[args.layer, :, :, spectra_slice], axis=2)

    _, ext = os.path.splitext(args.img_out_path)
    if ext == ".csv":
        np.savetxt(args.img_out_path, image, delimiter=",")
    else:
        imsave(args.img_out_path, image, cmap="gray", origin="lower")
