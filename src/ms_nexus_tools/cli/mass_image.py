import argparse

import datargs
from ..api import mass_image


def massimage():
    parser = argparse.ArgumentParser(prog="mass_image")

    datargs.add_arguments(parser, mass_image.ProcessArgs)

    args = parser.parse_args()

    process_args = mass_image.ProcessArgs(**vars(args))

    mass_image.process(process_args)
