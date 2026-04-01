from typing import Any

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from icecream import ic, install

from datargs import (
    arg_field,
    ArgType,
    ConfigFileArgs,
    InteractiveArgs,
    FilePathType,
    DirPathType,
)
from .image_args import MassSliceArgs, WidthAndHeightSliceArgs, LayerSliceArgs
from .formula_args import FormulaArgs
from .mass_range_args import MassRangeArgs
from ..lib.chunking import Chunker, count_chunks_to_cover
from ..lib.filter import MassRangeTotalImage, Accumulator
from ..lib.nxs import NexusFile
from ..lib.utils import slice_len, count_digits
from ..lib.normalisation import Norm, norm
from . import (
    image_plot as nxtic,
    spectrum_plot as nxts,
    image_and_spectrum_plot as nxisp,
    kendrick_mass_defect_plot as nxkdm,
)

install()


class OriginLocatoin(Enum):
    UPPER_LEFT = "upper left"
    UPPER_RIGHT = "upper right"
    LOWER_LEFT = "lower left"
    LOWER_RIGHT = "lower right"


@dataclass
class ProcessArgs(
    ConfigFileArgs,
    InteractiveArgs,
    MassSliceArgs,
    WidthAndHeightSliceArgs,
    LayerSliceArgs,
    FormulaArgs,
    MassRangeArgs,
):
    in_path: Path = arg_field(
        "-i",
        "--input",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        help="The nxs file to process.",
        default=None,
        type=FilePathType(must_exist=True),
    )

    out_dir: Path = arg_field(
        "-o",
        "--output",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        help="The directory to place the requested images and spectra.",
        default=None,
        type=DirPathType(must_exist=False),
    )

    scaling: Norm = arg_field(
        doc="The scaling to use within each image.",
        choices=[t for t in Norm],
        default=Norm.NONE,
    )

    accumulator: Accumulator = arg_field(
        "--acc",
        doc="The method used to accumulate each image and spectra.",
        choices=[t for t in Accumulator],
        default=Accumulator.TIC,
    )

    sum_all_layers: bool = arg_field(
        action="store_true",
        doc="If present then the spectra and images will be summed across all the layers inthe file. If absent (the default) the images and spectra will be written out for each layer.",
    )

    plot_total_spectrum: bool = arg_field(
        action="store_true",
        doc="If present will also output the spectrum for all masses.",
    )

    plot_total_image: bool = arg_field(
        action="store_true", doc="If present will also output the image for all pixels."
    )

    plot_kdm: bool = arg_field(
        action="store_true",
        doc="If present will plot the total Kendrick Mass Defect per layer.",
    )

    origin: OriginLocatoin = arg_field(
        doc="The location of the origin in the images.",
        choices=[t for t in OriginLocatoin],
        default=OriginLocatoin.UPPER_LEFT,
    )


def process(args: ProcessArgs, config: dict[str, Any] = {}):

    assert args.in_path.exists(), f"The input file {args.in_path} was not found"

    nx_file = NexusFile(args.in_path, mode="r")

    if args.plot_total_image:
        tic_config = nxtic.PlotKwArgs.read_config(config, "total_ion_count")
    if args.plot_total_spectrum:
        ts_config = nxts.PlotKwArgs.read_config(config, "total_spectra")
    if args.plot_kdm:
        kdm_config = nxkdm.PlotKwArgs.read_config(config, "kendrick_mass_defect")
    isp_config = nxisp.PlotKwArgs.read_config(config, "calibration_plot")

    with nx_file.as_context() as nx:
        shape = nx.root.entry.spectra.data.signal.shape
        mass_values = nx.root.entry.spectra.data.mass.nxdata

        layer_slice = args.calculate_layer_slice(shape[0])
        width_slice, height_slice = args.calculate_width_and_height_slice(
            shape[1], shape[2]
        )
        mass_slice = args.calculate_mass_slice(mass_values)

        data_shape = (
            slice_len(layer_slice),
            slice_len(width_slice),
            slice_len(height_slice),
            slice_len(mass_slice),
        )

        formula_data, formula_images = args.get_formulae_filters(
            data_shape[1:], mass_values
        )
        mass_range_data, mass_images = args.get_mass_filters(
            data_shape[1:], mass_values
        )

        total_mass_width = sum([f.mass_index_width for f in formula_data])
        total_mass_width += sum([m.mass_index_width for m in mass_range_data])

        spectra_chunk_count = np.prod(
            count_chunks_to_cover(shape, nx.root.entry.spectra.data.signal.chunks)
        )

        images: list[MassRangeTotalImage] = [*mass_images, *formula_images]
        image_chunking = nx.root.entry.images.data.signal.chunks
        images_chunk_count = np.sum(
            [
                np.prod(
                    count_chunks_to_cover(
                        (*data_shape[0:3], img.width()), image_chunking
                    )
                )
                for img in images
            ]
        )

        layer_digits = count_digits(data_shape[0])
        for ll in range(layer_slice.start, layer_slice.stop):
            for image in images:
                image.clear()

            if images_chunk_count < spectra_chunk_count:
                for image in images:
                    for bb in image.range():
                        for inner_image in images:
                            inner_image.add_image(
                                bb, nx.root.entry.images.data.signal[ll, :, :, bb]
                            )
            else:
                for x in range(width_slice.start, width_slice.stop):
                    for y in range(width_slice.start, width_slice.stop):
                        for image in images:
                            image.add_spectra(
                                x, y, nx.root.entry.spectra.data.signal[ll, x, y, :]
                            )

            title = f"{args.in_path.stem}"
            args.plot_formulae_ranges(
                mass_values,
                formula_data,
                formula_images,
                args.accumulator,
                args.scaling,
                args.out_dir,
                f"{title}.layer_{ll + 1:0{layer_digits}}",
                isp_config,
            )

            args.plot_mass_ranges(
                mass_values,
                mass_range_data,
                mass_images,
                args.accumulator,
                args.scaling,
                args.out_dir,
                f"{title}.layer_{ll + 1:0{layer_digits}}",
                isp_config,
            )
            if args.plot_total_image:
                filename = f"{title}.layer_{ll + 1:0{layer_digits}}.image.png"
                nxtic.process(
                    nxtic.ProcessArgs(
                        title,
                        nx.root.entry.total_images.data.signal[ll, :, :].nxdata,
                        Path(args.out_dir, filename),
                        plot_args=tic_config,
                    )
                )

            if args.plot_total_spectrum:
                filename = f"{title}.layer_{ll + 1:0{layer_digits}}.spectrum.png"
                nxts.process(
                    nxts.ProcessArgs(
                        title,
                        mass_values,
                        nx.root.entry.total_spectra.data.signal[ll, :].nxdata,
                        Path(args.out_dir, filename),
                        plot_args=ts_config,
                    )
                )

            if args.plot_kdm:
                filename = f"{title}.layer_{ll + 1:0{layer_digits}}.kdm.png"
                nxkdm.process(
                    nxkdm.ProcessArgs(
                        title,
                        mass_values,
                        nx.root.entry.total_spectra.data.signal[ll, :].nxdata,
                        Path(args.out_dir, filename),
                        normalisation=nxkdm.Normalisation.QUADRATIC,
                        plot_args=kdm_config,
                    )
                )
