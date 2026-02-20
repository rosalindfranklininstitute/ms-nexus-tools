import argparse

from ..api import spectrum_image, api


def specimage():
    parser = argparse.ArgumentParser(prog="spectrum_image")

    api.add_arguments(parser, spectrum_image.ProcessArgs)

    args = parser.parse_args()

    process_args = spectrum_image.ProcessArgs(**vars(args))

    spectrum_image.process(process_args)
