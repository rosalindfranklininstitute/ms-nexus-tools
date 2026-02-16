import argparse

import h5py as h5

from ..lib import print_group

parser = argparse.ArgumentParser(prog="inspect")
parser.add_argument("filename", help="The file")


args = parser.parse_args()

with h5.File(args.filename, "r") as hdfinfile:
    print_group(hdfinfile)
