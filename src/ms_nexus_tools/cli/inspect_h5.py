import argparse

import h5py as h5

from ..lib import print_group


def inspect():

    parser = argparse.ArgumentParser(prog="inspect")
    parser.add_argument("filename", help="The file")
    parser.add_argument(
        "-d", "--depth", default=None, help="The maximum depth to print."
    )

    args = parser.parse_args()

    with h5.File(args.filename, "r") as hdfinfile:
        print_group(hdfinfile, max_depth=args.depth if args.depth is not None else -1)
