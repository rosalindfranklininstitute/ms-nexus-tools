import argparse

from ..api import api, compound as comp_api


def compound():
    parser = argparse.ArgumentParser(prog="compound")

    api.add_arguments(parser, comp_api.ProcessArgs)

    args = parser.parse_args()

    process_args = comp_api.ProcessArgs(**vars(args))

    properties = comp_api.process(process_args)

    print(f"{properties.formula} has:")
    print(f"  {properties.average_mass}Da average mass")
    print(f"  {properties.lightest_monoisotropic_mass}Da lightest monoisotropic mass")
    print(
        f"  {properties.abundant_monoisotropic_mass}Da most abundant monoisotropic mass"
    )
