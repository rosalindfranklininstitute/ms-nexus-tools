# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from typing import Any
from pathlib import Path
from dataclasses import dataclass

from pyimzml.ImzMLWriter import ImzMLWriter
from pyimzml.compression import NoCompression

from tqdm import tqdm
import numpy as np

from datargs import arg_field, FilePathType, ConfigFileArgs, InteractiveArgs, ArgType

from ..lib.nxs import NexusFile


@dataclass
class ProcessArgs(ConfigFileArgs, InteractiveArgs):
    in_path: Path = arg_field(
        "-i",
        "--input",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        help="The nxs file to process.",
        default=None,
        type=FilePathType(must_exist=True),
    )

    out_path: Path = arg_field(
        "-o",
        "--output",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        help="The output filename.",
        default=None,
        type=FilePathType(must_exist=False),
    )

    one_indexed: bool = arg_field(
        action="store_true",
        doc="If present the data will be written with indexing starting at 1.",
    )


def process(args: ProcessArgs, config: dict[str, Any] = {}) -> None:

    nx_file = NexusFile(args.in_path, mode="r")

    with nx_file.as_context() as nx:
        shape = nx.root.entry.spectra.data.signal.shape
        mz = nx.root.entry.spectra.data.mass.nxdata

        with ImzMLWriter(
            args.out_path,
            mz_dtype=np.float32,
            intensity_dtype=np.float32,
            mode="continuous",
            mz_compression=NoCompression(),
            intensity_compression=NoCompression(),
            spec_type="profile",
        ) as writer:
            xyz = [
                (x, y, z)
                for x, y, z in itertools.product(
                    range(shape[1]),
                    range(shape[2]),
                    range(shape[0]),
                )
            ]
            for x, y, z in tqdm(xyz):
                coords = (x + 1, y + 1, z + 1) if args.one_indexed else (x, y, z)
                writer.addSpectrum(
                    mz,
                    nx.root.entry.spectra.data.signal[z, x, y, :].astype(np.float32),
                    coords,
                )
