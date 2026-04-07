from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..lib.bounds import Shape
from ..lib.image import OriginLocation, adjust_origin
from ..lib.filter import MassRangeTotalImage, Accumulator
from ..lib.normalisation import norm, Norm, IncrementalAccumulator

from datargs import arg_field

from .image_and_spectrum_plot import (
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
    start_mass: float
    stop_mass: float
    start_mass_index: int
    stop_mass_index: int
    mass_index_width: int

    def slice(self) -> slice:
        return slice(self.start_mass_index, self.stop_mass_index)

    @staticmethod
    def from_mass_range(
        start_mass: float, stop_mass: float, mass_axis: np.ndarray
    ) -> "MassRange":
        start_mass_index = bisect_left(mass_axis, start_mass)
        stop_mass_index = bisect_right(mass_axis, stop_mass)

        return MassRange(
            start_mass=start_mass,
            stop_mass=stop_mass,
            start_mass_index=start_mass_index,
            stop_mass_index=stop_mass_index,
            mass_index_width=stop_mass_index - start_mass_index,
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

    def plot_mass_ranges(
        self,
        mass_values: np.ndarray,
        mass_data: list[MassRange],
        mass_images: list[MassRangeTotalImage],
        accumulator: Accumulator,
        normalisation: Norm,
        origin: OriginLocation,
        target_dir: Path,
        name: str,
        write_txt: bool,
        isp_config: ISPKwArgs,
    ):
        assert len(mass_data) == len(mass_images)

        for md, mi in zip(mass_data, mass_images):
            filename = f"{name}.{accumulator.value}_{normalisation.value}.{md.start_mass}-{md.stop_mass}"
            title = f"{name}: ({accumulator.value}/{normalisation.value}): {md.start_mass}-{md.stop_mass}"

            scaling = norm(mi.spectrum(accumulator), normalisation)
            isp_process(
                ISPProcessArgs(
                    title,
                    mass_values[mi.slice()],
                    mi.spectrum(accumulator) / scaling,
                    adjust_origin(
                        mi.image(accumulator) / scaling,
                        origin,
                    ),
                    target_dir / f"{filename}.png",
                    plot_args=isp_config,
                )
            )

            if write_txt:
                np.savetxt(
                    target_dir / f"{filename}.image.txt",
                    adjust_origin(
                        mi.image(accumulator) / scaling,
                        origin,
                    ),
                )

                total_spectra_data = np.array(
                    [mass_values[mi.slice()], mi.spectrum(accumulator) / scaling]
                ).T

                np.savetxt(target_dir / f"{filename}.spectrum.txt", total_spectra_data)

    def accumulate_mass_ranges(
        self,
        mass_data: list[MassRange],
        mass_images: list[MassRangeTotalImage],
        accumulator: Accumulator,
        normalisation: Norm,
        target_dir: Path,
        name: str,
        write_txt: bool,
        ip_config: IPKwArgs,
    ):

        assert len(mass_data) == len(mass_images)

        filename = f"{name}.{accumulator.value}_{normalisation.value}.acc"
        title = (
            f"{name}: ({accumulator.value}/{normalisation.value}): Accumumlated ranges"
        )

        image_acc = IncrementalAccumulator(axis=2)

        for md, mi in zip(mass_data, mass_images):
            image_acc.add(mi.image(accumulator))

        scaling = norm(image_acc[accumulator], normalisation)

        ip_process(
            IPProcessArgs(
                title,
                image_acc[accumulator] / scaling,
                target_dir / f"{filename}.png",
                plot_args=ip_config,
            )
        )

        if write_txt:
            np.savetxt(target_dir / f"{filename}.txt", image_acc[accumulator] / scaling)
