import re
from dataclasses import dataclass

import scipy

from .api import arg_field, ArgType
from ..lib import elements


@dataclass
class ProcessArgs:
    compound: str = arg_field(
        arg_type=ArgType.POSITIONAL,
        doc="The compound to calculate the properties of.",
    )


@dataclass
class CompoundProperties:
    formula: str
    average_mass: float = 0.0
    lightest_monoisotropic_mass: float = 0.0
    abundant_monoisotropic_mass: float = 0.0


def process(args: ProcessArgs) -> CompoundProperties:

    properties = CompoundProperties(args.compound)

    for val in re.finditer(r"([a-zA-Z]+)(\d+)?", properties.formula):
        element, count = val.groups()
        count = int(count) if count is not None else 1
        isotopes = elements.elements[element].isotopes
        properties.average_mass += (
            count * sum([iso.accurate * iso.abundance for iso in isotopes]) / 100
        )
        abundant = max(isotopes, key=lambda x: x.abundance)
        properties.lightest_monoisotropic_mass += count * isotopes[0].accurate
        properties.abundant_monoisotropic_mass += count * abundant.accurate

    return properties
