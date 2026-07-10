# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

import argparse

import datargs
from ..api import compound as comp_api


def compound() -> None:
    parser = argparse.ArgumentParser(prog="compound")

    datargs.add_arguments(parser, comp_api.ProcessArgs)

    args = parser.parse_args()

    process_args = comp_api.ProcessArgs(**vars(args))

    properties = comp_api.process(process_args)

    print(f"{properties.formula} has:")
    print(f"  {properties.average_mass}Da average mass")
    print(f"  {properties.lightest_monoisotropic_mass}Da lightest monoisotropic mass")
    print(
        f"  {properties.abundant_monoisotropic_mass}Da most abundant monoisotropic mass",
    )
