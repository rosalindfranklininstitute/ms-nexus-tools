import argparse

from ..api import ion


def ion2rfi():
    parser = argparse.ArgumentParser(prog="ION2RFI")
    parser.add_argument("-i", "--input", help="The input file", required=True)
    parser.add_argument("-o", "--output", help="The output file", required=True)
    parser.add_argument(
        "-k",
        "--chunks",
        help="How many intermediate chunks to use. Default: 150.",
        default=150,
        type=int,
    )
    parser.add_argument(
        "-m",
        "--chunk-memory",
        help="The maximum memory a chunk should take, in Gb. Default: unbounded.",
        default=None,
        type=float,
    )

    parser.add_argument(
        "-j",
        "--processors",
        help="Process the data using sub-processors on disk using this many processors. If 1 it does not chunk onto disk, but remains in memory. Defaults: 1",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--no-spectra",
        help="If present, will not process the spectra part of the input file. Default: True",
        action="store_false",
        dest="spectra",
    )

    parser.add_argument(
        "--no-mass-images",
        help="If present, will not process the mass_images part of the input file. Default: True",
        action="store_false",
        dest="mass_images",
    )

    args = parser.parse_args()

    process_args = ion.ProcessArgs(
        args.input,
        args.output,
        args.chunks,
        args.chunk_memory,
        processors=args.processors,
        do_spectra=args.spectra,
        do_mass_images=args.mass_images,
        compression="gzip",
        compression_level=4,
    )

    ion.process(process_args)
