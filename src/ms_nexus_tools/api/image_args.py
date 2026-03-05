from dataclasses import dataclass
import argparse
import bisect

from .api import arg_field


class LayerSliceArgs:
    start_layer: int = arg_field(
        "-sl",
        doc="The start layer for the spectrum.",
        default=0,
    )

    end_layer: int = arg_field(
        "-el",
        doc="The end layer for the spectrum.",
        default=-1,
    )

    def calculate_slice(self, args: argparse.Namespace) -> slice:
        return slice(args.start_layer, args.end_layer)


class WidthAndHeightSliceArgs:
    start_width: int = arg_field(
        "-sw",
        doc="The start position within the width for the spectrum.",
        default=0,
    )

    end_width: int = arg_field(
        "-ew",
        doc="The end position within the width for the spectrum.",
        default=-1,
    )

    start_height: int = arg_field(
        "-sh",
        doc="The start position within the height for the spectrum.",
        default=0,
    )

    end_height: int = arg_field(
        "-eh",
        doc="The end position within the height for the spectrum.",
        default=-1,
    )

    def calculate_slice(self, args: argparse.Namespace) -> tuple[slice, slice]:
        return slice(args.start_width, args.end_width), slice(
            args.start_height, args.end_height
        )


class MassSliceArgs:
    use_index_for_mass: bool = arg_field(
        "-im",
        "--index-mass",
        action="store_true",
        doc="If present treat the start and end mass values as indices instead of mass values.",
    )
    start_mass: float = arg_field(
        "-sm",
        doc="The start position within the spectrum for the image.",
        default=0,
    )

    end_mass: float = arg_field(
        "-em",
        doc="The end position within the spectrum for the image.",
        default=-1,
    )

    def calculate_slice(
        self, args: argparse.Namespace, mass_axis: list[float]
    ) -> slice:
        if not args.use_index_for_mass:
            return slice(
                bisect.bisect_left(mass_axis, args.start_mass),
                bisect.bisect_right(mass_axis, args.end_mass),
                None,
            )
        else:
            return slice(int(args.start_mass), int(args.end_mass), None)
