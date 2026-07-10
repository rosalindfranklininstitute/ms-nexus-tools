from functools import reduce
import h5py
import argparse

import numpy as np

from datargs.extra_types import FilePathType

from ..lib.chunker import count_chunks_to_cover
from ..lib.utils import format_bytes


def print_nx_data_stats() -> None:
    parser = argparse.ArgumentParser(
        "nx_data_stats",
        description="Reads all the high leve subentries prints some details about the data.",
    )

    parser.add_argument(
        "filename",
        help="The file that should be read.",
        type=FilePathType(must_exist=True),
    )

    args = parser.parse_args()

    with h5py.File(args.filename) as f:
        for name, dataset in f["/entry"].items():
            if "data" in dataset:
                shape = dataset["data/signal"].shape
                chunks = dataset["data/signal"].chunks
                dsid = dataset["data/signal"].id
                n = dsid.get_num_chunks()
                count = count_chunks_to_cover(shape, chunks)

                total_chunks = reduce(lambda x, y: x * y, count)
                total_items = reduce(lambda x, y: x * y, shape)
                dtype = dataset["data/signal"].dtype
                width = np.dtype(dtype).itemsize
                print(f"Details for dataset {name}")
                print(
                    f" Shape: {shape} giving {total_items} items ({dtype} giving {format_bytes(total_items * width)})",
                )
                print(
                    f" Chunk shape {chunks} and count {count} ({format_bytes(np.prod(chunks) * width)})",
                )
                print(f" Total chunks {total_chunks}, of which {n} were used.")
                print(f" Density {n / total_chunks:.2f}")
