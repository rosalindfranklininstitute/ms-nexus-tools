from ms_nexus_tools.lib.bounds import Chunk
from typing import Any

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging
import itertools

import numpy as np
import scipy

from tqdm import tqdm


from datargs import (
    no_arg_field,
    arg_field,
    ArgType,
    ConfigFileArgs,
    InteractiveArgs,
    FilePathType,
    DirPathType,
)
from .image_args import MassSliceArgs, WidthAndHeightSliceArgs, LayerSliceArgs
from .mass_range_args import (
    MassRange,
    MassRangeArgs,
    MassCentreArgs,
    plot_mass_ranges,
    accumulate_mass_ranges,
)
from ..lib.data_source import AbstractQuerySource
from ..lib.chunking import Chunker, count_chunks_to_cover
from ..lib.filter import MassRangeTotalImage, Accumulator
from ..lib.image import OriginLocation, adjust_origin
from ..lib.nxs import NexusFile
from ..lib.utils import slice_len, count_digits
from ..lib.normalisation import Norm, norm, normalise
from . import (
    image_plot as nxtic,
    spectrum_plot as nxts,
    image_and_spectrum_plot as nxisp,
    kendrick_mass_defect_plot as nxkdm,
)

from icecream import ic, install

install()

logger = logging.getLogger(__name__)


@dataclass
class ProcessArgs(
    ConfigFileArgs,
    InteractiveArgs,
    MassSliceArgs,
    WidthAndHeightSliceArgs,
    LayerSliceArgs,
    MassCentreArgs,
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

    accumulate_masses: bool = arg_field(
        action="store_true",
        doc="If present then the mass images from the --mass-range will be accumulated into one image.",
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

    origin: OriginLocation = arg_field(
        doc="The location of the origin in the images.",
        choices=[t for t in OriginLocation],
        default=OriginLocation.UPPER_LEFT,
    )

    write_txt: bool = arg_field(
        action="store_true",
        doc="If present will write out all images and spectra to .txt files using numpy.savetxt.",
    )

    subpixels: float = arg_field(
        doc="""
        The number of pixels in the output image for every pixel in the input image. 
        i.e. --subpixels=2 will have twice the number of pixels as the original data. 
        See [scipy.ndimage.zoom](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html) for details. 
        The new pixel value will be calculated by interpolation.
        """,
        action="store",
        default=1.0,
    )

    interpolation_order: int = arg_field(
        "--interpolation",
        "--int-otder",
        doc="""
        The order of the spline to use for finding the values of subpixels. 
        0 is nearest neighbout, 1 is linear, etc. Maximum is 5. See [scipy.ndimage.zoom](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html) for details. 
        Values outside the image are treated as having a value of 0.
        Note: that interpolation takes place after normalisation (accumulation and scaling).
        Note: that interpolation takes place before writing images and txt files.
        """,
        action="store",
        default=0,
        choices=[i for i in range(5)],
    )

    query_source: AbstractQuerySource = no_arg_field(default=None)


def process(args: ProcessArgs, config: dict[str, Any] = {}):

    assert args.in_path.exists(), f"The input file {args.in_path} was not found"

    nx_file = NexusFile(args.in_path, mode="r")

    if args.plot_total_image or args.accumulate_masses:
        tic_config = nxtic.PlotKwArgs.read_config(config, "total_ion_count")
    if args.plot_total_spectrum:
        ts_config = nxts.PlotKwArgs.read_config(config, "total_spectra")
    if args.plot_kdm:
        kdm_config = nxkdm.PlotKwArgs.read_config(config, "kendrick_mass_defect")
    isp_config = nxisp.PlotKwArgs.read_config(config, "calibration_plot")

    with args.query_source as nx:
        shape = nx.shape()
        mass_values = nx.mass_values()

        layer_slice = args.calculate_layer_slice(shape[0])
        width_slice, height_slice = args.calculate_width_and_height_slice(
            shape[1], shape[2]
        )
        mass_slice = args.calculate_mass_slice(mass_values)

        inner_chunk = Chunk([layer_slice, width_slice, height_slice, mass_slice])
        data_shape = (
            slice_len(layer_slice),
            slice_len(width_slice),
            slice_len(height_slice),
            slice_len(mass_slice),
        )

        centre_data, centre_images = args.get_centre_filters(
            data_shape[1:], mass_values
        )
        range_data, range_images = args.get_mass_filters(data_shape[1:], mass_values)

        mass_data: list[MassRange] = [*range_data, *centre_data]
        mass_images: list[MassRangeTotalImage] = [*range_images, *centre_images]

        bins = set()
        for image in mass_images:
            bins.update([bb for bb in image.range()])
        bins: list[int] = sorted(bins)
        xy = [
            (x, y)
            for x, y in itertools.product(
                range(width_slice.start, width_slice.stop),
                range(height_slice.start, height_slice.stop),
            )
        ]

        layer_digits = count_digits(data_shape[0])
        for ll in range(layer_slice.start, layer_slice.stop):
            for image in mass_images:
                image.clear()

            nx.fill_filters(ll, bins, xy, mass_images, inner_chunk)

            title = f"{args.in_path.stem}"
            norm_title = f"({args.accumulator.value}/{args.scaling.value})"

            plot_mass_ranges(
                mass_values,
                mass_data,
                mass_images,
                args.accumulator,
                args.scaling,
                args.subpixels,
                args.interpolation_order,
                args.origin,
                args.out_dir,
                f"{title}.layer_{ll + 1:0{layer_digits}}",
                args.write_txt,
                isp_config,
            )

            if args.accumulate_masses:
                try:
                    accumulate_mass_ranges(
                        mass_images,
                        args.accumulator,
                        args.scaling,
                        args.subpixels,
                        args.interpolation_order,
                        args.origin,
                        args.out_dir,
                        f"{title}.layer_{ll + 1:0{layer_digits}}",
                        args.write_txt,
                        tic_config,
                    )
                except RuntimeError:
                    logger.warning(
                        "Requested an accumulated image, but there was no data for the masses specified."
                    )

            def total_spectra():
                spectra = nx.accumulated_spectrum(args.accumulator, ll)
                return normalise(spectra, args.scaling)

            def total_images():
                image = nx.accumulated_image(args.accumulator, ll)
                if abs(args.subpixels - 1.0) < 1e-2:
                    return adjust_origin(normalise(image, args.scaling), args.origin)
                else:
                    return scipy.ndimage.zoom(
                        adjust_origin(normalise(image, args.scaling), args.origin),
                        zoom=args.subpixels,
                        order=args.interpolation_order,
                        mode="constant",
                        cval=0.0,
                    )

            filename = f"{title}.layer_{ll + 1:0{layer_digits}}.{args.accumulator.value}_{args.scaling.value}"

            if args.write_txt:
                np.savetxt(
                    args.out_dir / f"{filename}.image.txt",
                    total_images(),
                )

                total_spectra_data = np.array([mass_values, total_spectra()]).T

                np.savetxt(
                    args.out_dir / f"{filename}.spectrum.txt", total_spectra_data
                )

            if args.plot_total_image:
                nxtic.process(
                    nxtic.ProcessArgs(
                        f"{title}: Total Image: {norm_title}",
                        total_images(),
                        args.out_dir / f"{filename}.image.png",
                        plot_args=tic_config,
                    )
                )

            if args.plot_total_spectrum:
                nxts.process(
                    nxts.ProcessArgs(
                        f"{title}: Total Spectrum: {norm_title}",
                        mass_values,
                        total_spectra(),
                        args.out_dir / f"{filename}.spectrum.png",
                        plot_args=ts_config,
                    )
                )

            if args.plot_kdm:
                nxkdm.process(
                    nxkdm.ProcessArgs(
                        f"{title}: Total KDM: {norm_title}",
                        mass_values,
                        total_spectra(),
                        args.out_dir / f"{filename}.kdm.png",
                        normalisation=nxkdm.Normalisation.QUADRATIC,
                        plot_args=kdm_config,
                    )
                )
