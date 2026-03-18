from typing import Any, Literal, NamedTuple
from collections.abc import Sequence
from dataclasses import dataclass
from bisect import bisect_left, bisect_right

import numpy as np

from .args import arg_field
from . import compound as nxcomp


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
    calibration_formulae: list[str] = arg_field(
        "--cal",
        "--formula",
        action="append",
        doc="Each instance of this adds a plot of the total spectra centered around this mass, with width --calibration-width",
        default_factory=list,
    )

    calibration_width: float = arg_field(
        "--cal-width",
        doc="The mass range to plot around each calibration/interest formula.",
        default=1.0,
    )

    def calculate_formulae_ranges(
        self, mass_values: np.ndarray
    ) -> list[FormulaAndRange]:

        return [
            FormulaAndRange.from_formula_and_width(
                f, self.calibration_width, mass_values
            )
            for f in self.calibration_formulae
        ]
