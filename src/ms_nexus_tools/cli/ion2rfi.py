import argparse

from ..api import ion, args as nxargs


def ion2rfi():
    parser = argparse.ArgumentParser(prog="ION2RFI")

    nxargs.add_arguments(parser, ion.ProcessArgs)

    args = parser.parse_args()

    process_args = ion.ProcessArgs(**vars(args))

    ion.process(process_args)
