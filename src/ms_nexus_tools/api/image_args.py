from collections.abc import Sequence
from dataclasses import dataclass
import bisect

from .args import arg_field


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
        return slice(
            self.start_layer, self.end_layer if self.end_layer is not None else layers
        )


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
        return slice(
            self.start_width, self.end_width if self.end_width is not None else width
        ), slice(
            self.start_height,
            self.end_height if self.end_height is not None else height,
        )


@dataclass
class MassSliceArgs:
    use_mass: bool = arg_field(
        "-mv",
        "--mass-value",
        action="store_true",
        doc="If present treat the start and end mass values as masses instead of indices.",
        defer=True,
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

    def calculate_mass_slice(self, mass_axis: Sequence[int | float]) -> slice:
        if self.use_mass:
            return slice(
                bisect.bisect_left(mass_axis, self.start_mass),
                bisect.bisect_right(mass_axis, self.end_mass)
                if self.end_mass is not None
                else len(mass_axis),
                None,
            )
        else:
            return slice(
                int(self.start_mass),
                int(self.end_mass if self.end_mass is not None else len(mass_axis)),
                None,
            )
