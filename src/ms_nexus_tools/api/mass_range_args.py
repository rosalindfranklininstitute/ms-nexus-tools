# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
import copy

import numpy as np
import scipy

from ..lib.bounds import Shape
from ..lib.image import OriginLocation, adjust_origin
from ..lib.filter import MassRangeTotalImage, Accumulator
from ..lib.normalisation import norm, Norm, IncrementalAccumulator
from . import compound as nxcomp

from datargs import arg_field

from .image_and_spectrum_plot import (
    AxCommand,
    PlotKwArgs as ISPKwArgs,
    ProcessArgs as ISPProcessArgs,
    process as isp_process,
)

from .image_plot import (
    PlotKwArgs as IPKwArgs,
    ProcessArgs as IPProcessArgs,
    process as ip_process,
)


class MassRange(NamedTuple):
    start_mass_index: int
    stop_mass_index: int
    mass_index_width: int
    title: str
    file_part: str
    markers: list[float]

    def slice(self) -> slice:
        return slice(self.start_mass_index, self.stop_mass_index)

    @staticmethod
    def from_mass_range(
        start_mass: float, stop_mass: float, mass_axis: np.ndarray
    ) -> "MassRange":
        start_mass_index = bisect_left(mass_axis, start_mass)
        stop_mass_index = bisect_right(mass_axis, stop_mass)

        return MassRange(
            start_mass_index=start_mass_index,
            stop_mass_index=stop_mass_index,
            mass_index_width=stop_mass_index - start_mass_index,
            title=f"{start_mass}-{stop_mass}",
            file_part=f"{start_mass}-{stop_mass}",
            markers=[],
        )

    @staticmethod
    def from_centre_and_width(
        centre: str, width: float, mass_axis: np.ndarray
    ) -> "MassRange":
        try:
            mass = float(centre)
        except ValueError:
            mass = nxcomp.process(
                nxcomp.ProcessArgs(centre)
            ).lightest_monoisotropic_mass

        half_width = width / 2
        start_mass_index = bisect_left(mass_axis, mass - half_width)
        stop_mass_index = bisect_right(mass_axis, mass + half_width)

        return MassRange(
            start_mass_index=start_mass_index,
            stop_mass_index=stop_mass_index,
            mass_index_width=stop_mass_index - start_mass_index,
            title=f"{centre} +- {width}",
            file_part=f"{centre}+{width}",
            markers=[mass],
        )


@dataclass
class MassRangeArgs:
    mass_range: list[list[float]] = arg_field(
        action="append",
        doc="Each instance of this adds a plot of the total spectra. This takes 2 parameters: a start mass and a stop mass. ",
        default_factory=list,
        nargs=2,
    )

    def calculate_mass_ranges(self, mass_values: np.ndarray) -> list[MassRange]:

        return [
            MassRange.from_mass_range(mr[0], mr[1], mass_values)
            for mr in self.mass_range
        ]

    def get_mass_filters(
        self, image_shape: Shape, mass_values: np.ndarray
    ) -> tuple[list[MassRange], list[MassRangeTotalImage]]:

        data = self.calculate_mass_ranges(mass_values)
        results = [
            (m, MassRangeTotalImage(image_shape, m.start_mass_index, m.stop_mass_index))
            for m in data
            if m.mass_index_width > 0
        ]
        return [r[0] for r in results], [r[1] for r in results]


@dataclass
class MassCentreArgs:
    mass_centre: list[str] = arg_field(
        action="append",
        doc="Each instance of this adds a plot of the total spectra centered around this mass or formula, with width --mass-width",
        default_factory=list,
    )

    mass_width: float = arg_field(
        doc="The mass range to plot around each mass centre.",
        default=1.0,
    )

    def calculate_centre_ranges(self, mass_values: np.ndarray) -> list[MassRange]:

        return [
            MassRange.from_centre_and_width(f, self.mass_width, mass_values)
            for f in self.mass_centre
        ]

    def get_centre_filters(
        self, image_shape: Shape, mass_values: np.ndarray
    ) -> tuple[list[MassRange], list[MassRangeTotalImage]]:

        data = self.calculate_centre_ranges(mass_values)
        results = [
            (
                m,
                MassRangeTotalImage(image_shape, m.start_mass_index, m.stop_mass_index),
            )
            for m in data
            if m.mass_index_width > 0
        ]
        return [r[0] for r in results], [r[1] for r in results]


from icecream import ic


def plot_mass_ranges(
    mass_values: np.ndarray,
    mass_data: list[MassRange],
    mass_images: list[MassRangeTotalImage],
    accumulator: Accumulator,
    normalisation: Norm,
    subpixels: float,
    interpolation_order: int,
    origin: OriginLocation,
    target_dir: Path,
    name: str,
    write_txt: bool,
    isp_config: ISPKwArgs,
):
    assert len(mass_data) == len(mass_images)

    for md, mi in zip(mass_data, mass_images):
        filename = f"{name}.{accumulator.value}_{normalisation.value}.{md.file_part}"
        title = f"{name}: ({accumulator.value}/{normalisation.value}): {md.title}"

        scaling = norm(mi.spectrum(accumulator), normalisation)
        if abs(subpixels - 1.0) < 1e-2:
            image = adjust_origin(
                mi.image(accumulator) / scaling,
                origin,
            )
        else:
            image = scipy.ndimage.zoom(
                adjust_origin(
                    mi.image(accumulator) / scaling,
                    origin,
                ),
                zoom=subpixels,
                order=interpolation_order,
                mode="constant",
                cval=0.0,
            )
        config = copy.copy(isp_config)
        for marker in md.markers:
            config.plot_axes_commands_and_kw_args = [
                AxCommand(
                    command="axvline",
                    kwargs=dict(x=marker, linewidth=0.5, linestyle=":"),
                )
            ]

        isp_process(
            ISPProcessArgs(
                title,
                mass_values[mi.slice()],
                mi.spectrum(accumulator) / scaling,
                image,
                target_dir / f"{filename}.png",
                plot_args=config,
            )
        )

        if write_txt:
            np.savetxt(
                target_dir / f"{filename}.image.txt",
                image,
            )

            total_spectra_data = np.array(
                [mass_values[mi.slice()], mi.spectrum(accumulator) / scaling]
            ).T

            np.savetxt(target_dir / f"{filename}.spectrum.txt", total_spectra_data)


def accumulate_mass_ranges(
    mass_images: list[MassRangeTotalImage],
    accumulator: Accumulator,
    normalisation: Norm,
    subpixels: float,
    interpolation_order: int,
    origin: OriginLocation,
    target_dir: Path,
    name: str,
    write_txt: bool,
    ip_config: IPKwArgs,
):

    filename = f"{name}.{accumulator.value}_{normalisation.value}.acc"
    title = f"{name}: ({accumulator.value}/{normalisation.value}): Accumumlated ranges"

    acc_image = MassRangeTotalImage.accumulate_images(mass_images, accumulator)
    if acc_image is None:
        raise RuntimeError("No data accumulated.")

    scaling = norm(acc_image, normalisation)
    if abs(subpixels - 1.0) < 1e-2:
        image = adjust_origin(
            acc_image / scaling,
            origin,
        )
    else:
        image = scipy.ndimage.zoom(
            adjust_origin(
                acc_image / scaling,
                origin,
            ),
            zoom=subpixels,
            order=interpolation_order,
            mode="constant",
            cval=0.0,
        )

    ip_process(
        IPProcessArgs(
            title,
            image,
            target_dir / f"{filename}.png",
            plot_args=ip_config,
        )
    )

    if write_txt:
        np.savetxt(target_dir / f"{filename}.txt", image)
