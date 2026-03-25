import copy
from pathlib import Path
from typing import Any, NamedTuple
from dataclasses import dataclass
from bisect import bisect_left, bisect_right

import numpy as np

from .args import arg_field
from . import compound as nxcomp

from ..lib.bounds import Shape
from ..lib.filter import MassRangeTotalImage

from .image_and_spectrum_plot import (
    PlotKwArgs as ISPKwArgs,
    ProcessArgs as ISPProcessArgs,
    process as isp_process,
)


class FormulaAndRange(NamedTuple):
    formula: str
    mass: float
    start_mass_index: int
    stop_mass_index: int
    mass_index_width: int

    @staticmethod
    def from_formula_and_width(
        formula: str, width: float, mass_axis: np.ndarray
    ) -> "FormulaAndRange":
        mass = nxcomp.process(nxcomp.ProcessArgs(formula)).lightest_monoisotropic_mass
        half_width = width / 2
        start_mass_index = bisect_left(mass_axis, mass - half_width)
        stop_mass_index = bisect_right(mass_axis, mass + half_width)

        return FormulaAndRange(
            formula=formula,
            mass=mass,
            start_mass_index=start_mass_index,
            stop_mass_index=stop_mass_index,
            mass_index_width=stop_mass_index - start_mass_index,
        )


@dataclass
class FormulaArgs:
    formula: list[str] = arg_field(
        action="append",
        doc="Each instance of this adds a plot of the total spectra centered around this mass, with width --calibration-width",
        default_factory=list,
    )

    formula_width: float = arg_field(
        doc="The mass range to plot around each calibration/interest formula.",
        default=1.0,
    )

    def calculate_formulae_ranges(
        self, mass_values: np.ndarray
    ) -> list[FormulaAndRange]:

        return [
            FormulaAndRange.from_formula_and_width(f, self.formula_width, mass_values)
            for f in self.formula
        ]

    def get_formulae_filters(
        self, image_shape: Shape, mass_values: np.ndarray
    ) -> tuple[list[FormulaAndRange], list[MassRangeTotalImage]]:

        data = self.calculate_formulae_ranges(mass_values)
        results = [
            (
                m,
                MassRangeTotalImage(image_shape, m.start_mass_index, m.stop_mass_index),
            )
            for m in data
            if m.mass_index_width > 0
        ]
        return [r[0] for r in results], [r[1] for r in results]

    def plot_formulae_ranges(
        self,
        mass_values: np.ndarray,
        formulae_data: list[FormulaAndRange],
        formulae_images: list[MassRangeTotalImage],
        nxs_out_path: Path,
        isp_config: ISPKwArgs,
    ):

        isp_config = copy.copy(isp_config)

        assert len(formulae_data) == len(formulae_images)

        target_dir = Path(*nxs_out_path.parts[:-1])

        for fd, fi in zip(formulae_data, formulae_images):
            filename = f"{nxs_out_path.stem}.{fd.formula}.png"
            title = f"{nxs_out_path.stem}: {fd.formula}"

            isp_config.plot_axes_commands_and_kw_args.update(
                dict(axvline=dict(x=fd.mass, linewidth=0.5, linestyle=":"))
            )

            isp_process(
                ISPProcessArgs(
                    title,
                    mass_values[fi.slice()],
                    fi.total_spectrum,
                    fi.total_image,
                    target_dir / filename,
                    plot_args=isp_config,
                )
            )
