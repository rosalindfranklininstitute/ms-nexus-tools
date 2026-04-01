from collections.abc import Sequence
from dataclasses import dataclass
import bisect
from enum import Enum

import numpy as np

from datargs import arg_field


def _cycle(cycle: int | None, length: int) -> int:
    if cycle is None:
        return length
    elif cycle < 0:
        result = length + cycle
        if result < 0:
            raise IndexError(
                f"Index of {cycle} was greater than the dimension width of {length}"
            )
        return result
    elif cycle > length:
        raise IndexError(
            f"Index of {cycle} was greater than the dimension width of {length}"
        )
    else:
        return cycle


def _slice(start: int, end: int | None, length):
    start = _cycle(start, length)
    end = _cycle(end, length)
    return slice(start, end)


@dataclass
class LayerSliceArgs:
    start_layer: int = arg_field(
        "-sl",
        doc="The start layer for the spectrum.",
        default=0,
        defer=True,
    )

    end_layer: int = arg_field(
        "-el",
        doc="The end layer for the spectrum.",
        default=None,
        defer=True,
    )

    def calculate_layer_slice(self, layers: int) -> slice:
        return _slice(self.start_layer, self.end_layer, layers)


@dataclass
class WidthAndHeightSliceArgs:
    start_width: int = arg_field(
        "-sw",
        doc="The start position within the width for the spectrum.",
        default=0,
        defer=True,
    )

    end_width: int = arg_field(
        "-ew",
        doc="The end position within the width for the spectrum.",
        default=None,
        defer=True,
    )

    start_height: int = arg_field(
        "-sh",
        doc="The start position within the height for the spectrum.",
        default=0,
        defer=True,
    )

    end_height: int = arg_field(
        "-eh",
        doc="The end position within the height for the spectrum.",
        default=None,
        defer=True,
    )

    def calculate_width_and_height_slice(
        self, width: int, height: int
    ) -> tuple[slice, slice]:

        return _slice(self.start_width, self.end_width, width), _slice(
            self.start_height, self.end_height, height
        )


class MassMeasure(Enum):
    MASS = ("mass",)
    INDEX = "index"


@dataclass
class MassSliceArgs:
    use_mass: MassMeasure = arg_field(
        "-mv",
        "--mass-value",
        doc="If present treat the start and end mass values as masses instead of indices.",
        defer=True,
        choices=[t for t in MassMeasure],
        default=MassMeasure.MASS,
    )

    start_mass: float = arg_field(
        "-sm",
        doc="The start position within the spectrum for the image.",
        default=0,
        defer=True,
    )

    end_mass: float = arg_field(
        "-em",
        doc="The end position within the spectrum for the image.",
        default=None,
        defer=True,
    )

    def calculate_mass_slice(
        self, mass_axis: Sequence[int | float] | np.ndarray
    ) -> slice:
        if self.use_mass:
            return _slice(
                bisect.bisect_left(mass_axis, self.start_mass),
                bisect.bisect_right(mass_axis, self.end_mass)
                if self.end_mass is not None
                else None,
                len(mass_axis),
            )
        else:
            return _slice(
                int(self.start_mass),
                int(self.end_mass if self.end_mass is not None else None),
                len(mass_axis),
            )
