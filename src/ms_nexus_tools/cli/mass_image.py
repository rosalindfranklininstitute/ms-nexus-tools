import argparse

from ..api import mass_image, api


def massimage():
    parser = argparse.ArgumentParser(prog="mass_image")

    api.add_arguments(parser, mass_image.ProcessArgs)

    args = parser.parse_args()

    process_args = mass_image.ProcessArgs(**vars(args))

    mass_image.process(process_args)
