from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..lib.bounds import Shape
from ..lib.filter import MassRangeTotalImage, Accumulator
from ..lib.normalisation import norm, Norm

from datargs import arg_field

from .image_and_spectrum_plot import (
    PlotKwArgs as ISPKwArgs,
    ProcessArgs as ISPProcessArgs,
    process as isp_process,
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
        target_dir: Path,
        name: str,
        isp_config: ISPKwArgs,
    ):
        assert len(mass_data) == len(mass_images)

        for md, mi in zip(mass_data, mass_images):
            filename = f"{name}.{accumulator.value}_{normalisation.value}.{md.start_mass}-{md.stop_mass}.png"
            title = f"{name}: ({accumulator.value}/{normalisation.value}): {md.start_mass}-{md.stop_mass}"

            scaling = norm(mi.spectrum(accumulator), normalisation)
            isp_process(
                ISPProcessArgs(
                    title,
                    mass_values[mi.slice()],
                    mi.spectrum(accumulator) / scaling,
                    mi.image(accumulator) / scaling,
                    target_dir / filename,
                    plot_args=isp_config,
                )
            )
