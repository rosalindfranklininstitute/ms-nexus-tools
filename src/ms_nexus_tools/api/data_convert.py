# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
from ms_nexus_tools.lib.dtypes import Int1Dp

from typing import Any, reveal_type
from threading import Lock, local
import concurrent.futures as cfutures
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import numpy.typing as npt
import sparse
import dask.array as da

from icecream import ic

from tqdm import tqdm
from tqdm.dask import TqdmCallback

import hdf5plugin
from nexusformat.nexus import NXsubentry, NXdata

from datargs import (
    no_arg_field,
    arg_field,
    ArgType,
    ConfigFileArgs,
    InteractiveArgs,
    FilePathType,
)
from ..lib.data_source import (
    AbstractDataSource,
    AxisDensity,
    Axis,
    MultiCOO,
)
from ..lib.bounds import Chunk, Shape
from ..lib import ContainedBounds
from ..lib.chunker import Chunker, count_chunks_to_cover
from ..lib.nxs import NexusFile, FieldOptions, NxAxis, create_field, NxAxes
from ..lib.utils import format_bytes


def _count_subentry_name():
    return "item_counts"


def _items(d: dict[str, Any]) -> list[tuple[str, Any]]:
    return [(k, v) for k, v in d.items()]


@dataclass
class ProcessArgs(
    ConfigFileArgs,
    InteractiveArgs,
):
    in_path: Path = arg_field(
        "-i",
        "--input",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        help="The file to process.",
        default=None,
        type=FilePathType(must_exist=True),
    )

    out_path: Path = arg_field(
        "-o",
        "--output",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        help="The directory to place the requested images and spectra.",
        default=None,
        type=FilePathType(must_exist=False),
    )

    data_source: AbstractDataSource = no_arg_field(default=None)

    field_options: FieldOptions = no_arg_field(
        default=FieldOptions(
            compression=hdf5plugin.Blosc(),
            compression_opts=None,
            max_bytes_per_chunk=1024 * 1024 * 8,
            shuffle=True,
        )
    )

    memory_max_byte_count: int = no_arg_field(default=1024 * 1024 * 1024 * 4)  # 1Gb


def provision_subentries(
    nxs: NexusFile, args: ProcessArgs, full_shape: Shape, sparse: bool
) -> dict[str, Chunker]:
    data_priorities = args.data_source.output_chunks()
    if len(data_priorities) == 0:
        raise ValueError("At least one dataset must be provided.")
    for name, priorities in data_priorities.items():
        if len(name.strip()) == 0:
            raise ValueError(
                "An invalid name was returned for a data set. Names must not be empty."
            )
        if len(priorities) != len(full_shape):
            raise ValueError(
                f"An invalid set of priorities was returned for dataset {name}: there should be {len(full_shape)} items, but only {len(priorities)} were provided."
            )
    signal_item_width = np.dtype(args.data_source.signal_type()).itemsize
    data_chunks = {
        name: Chunker.from_max_item_count(
            data_shape=full_shape,
            priorities=priorities,
            items_per_chunk=int(
                args.field_options.max_bytes_per_chunk / signal_item_width
            ),
        )
        for name, priorities in data_priorities.items()
    }
    if sparse:
        counts_item_width = np.dtype(np.uint16).itemsize
        data_chunks[_count_subentry_name()] = Chunker.from_max_item_count(
            data_shape=full_shape,
            priorities=tuple(1 for _ in full_shape),
            items_per_chunk=int(
                args.field_options.max_bytes_per_chunk / counts_item_width
            ),
        )

    for name, chunker in data_chunks.items():
        nxs.root[name] = NXsubentry(
            NXdata(
                signal=create_field(
                    dtype=args.data_source.signal_type(),
                    shape=chunker.data_shape,
                    compression=args.field_options.compression,
                    compression_opts=args.field_options.compression_opts,
                    chunks=chunker.chunk_shape,
                    shuffle=args.field_options.shuffle,
                    fillvalue=0,
                )
            )
        )
    return data_chunks


def choose_memory_buffer(
    args: ProcessArgs,
    max_item_count: int,
    density: float,
    data_chunks: dict[str, Chunker],
) -> tuple[Chunker, int]:
    memory_chunks = {
        name: Chunker.find_chunk_multiple(
            chunker.data_shape,
            chunker.chunk_shape,
            max_item_count / density,
        )
        for name, chunker in data_chunks.items()
    }
    min_read_count = np.pow(2, 32, dtype=np.int64)
    min_read_name = ""
    for name, chunker in memory_chunks.items():
        read_count = 0
        for chunk in chunker.chunks():
            read_count += args.data_source.chunk_read_count(chunk.shape)
        if read_count < min_read_count:
            min_read_count = read_count
            min_read_name = name
    assert len(min_read_name) > 0
    return memory_chunks[min_read_name], min_read_count


def provision_data_axis(
    nxs: NexusFile, args: ProcessArgs, data_chunks: dict[str, Chunker]
) -> tuple[dict[str, Axis], bool]:
    axis_definitions = args.data_source.axis_definitions()

    any_sparse_axis = False
    for entry_name, chunker in data_chunks.items():
        # NOTE: Assumption: Only one axis per dimension.
        if len(axis_definitions) != len(chunker.data_shape):
            raise ValueError("Currently only one axis per dimension is supported.")
        group_axes = NxAxes()
        for axis in axis_definitions:
            match axis.density:
                case AxisDensity.CONTINUOUS:
                    if len(axis.secondary_axes) > 0:
                        raise ValueError(
                            "A continuouse axis should not have secondary axis."
                        )
                    values = args.data_source.continuous_axis_values(axis)
                    nx_axis = NxAxis.create(
                        values=values,
                        name=axis.name,
                        indices=[axis.primary_axis],
                        unit=axis.units,
                    )
                    group_axes.append([nx_axis])
                case AxisDensity.SPARSE:
                    any_sparse_axis = True
                    all_axis: list[int] = sorted(
                        [axis.primary_axis, *axis.secondary_axes]
                    )
                    values = args.data_source.sparse_axis_edges(axis)[1:]
                    nx_axis = NxAxis.create(
                        values=values,
                        name=axis.name,
                        indices=[axis.primary_axis],
                        unit=axis.units,
                    )
                    nx_axis_exact = NxAxis.create_empty(
                        name=f"{axis.name}_exact",
                        indices=all_axis,
                        unit=axis.units,
                        shape=tuple(chunker.data_shape[ii] for ii in all_axis),
                        compression=args.field_options.compression,
                        compression_opts=args.field_options.compression_opts,
                        chunks=chunker.chunk_shape,
                        dtype=axis.dtype,
                        fillvalue=np.nan,
                    )
                    group_axes.append([nx_axis, nx_axis_exact])
                case _:
                    raise ValueError(f"Unknown Axis density: {axis.density}")
        group_axes.add_to_group(nxs.root[entry_name]["data"])

    return {ax.name: ax for ax in axis_definitions}, any_sparse_axis


@dataclass
class Accumulation:
    name: str
    axis: tuple[int, ...]
    axis_edges: list[None | np.ndarray]
    shape: Shape

    contains_sparse_axes: bool = field(init=False)
    max_data: np.ndarray = field(init=False)
    sum_data: np.ndarray = field(init=False)
    ndim: int = field(init=False)
    has_data: bool = field(init=False)

    def __post_init__(self):

        self.contains_sparse_axes = False
        for edge in self.axis_edges:
            if edge is not None:
                self.contains_sparse_axes = True
                break

        self.max_data = np.zeros(self.shape)
        self.sum_data = np.zeros(self.shape)
        self.ndim = len(self.shape)
        self.has_data = False

    def add(self, data: np.ndarray | sparse.COO, chunk: Chunk):
        sub_chunk = Chunk([chunk[ii] for ii in range(data.ndim) if ii not in self.axis])

        max_data = np.maximum(
            self.max_data[*sub_chunk], data[*chunk].max(axis=self.axis)
        )
        sum_data = np.add(self.sum_data[*sub_chunk], data[*chunk].sum(axis=self.axis))

        if isinstance(max_data, sparse.COO):
            self.max_data[*sub_chunk] = max_data.todense()
        else:
            self.max_data[*sub_chunk] = max_data

        if isinstance(sum_data, sparse.COO):
            self.sum_data[*sub_chunk] = sum_data.todense()
        else:
            self.sum_data[*sub_chunk] = sum_data

        self.has_data = True


def provision_accumulation_subentries(
    nxs: NexusFile, args: ProcessArgs, shape: Shape, axis_definitions: dict[str, Axis]
) -> tuple[dict[str, Accumulation], dict[str, Accumulation]]:
    accumulations = args.data_source.output_accumulations()
    dtype = args.data_source.signal_type()

    final_accumulations: dict[str, Accumulation] = {}
    count_accumulations: dict[str, Accumulation] = {}

    # NOTE: Assumption: an axis only defines 1 dimension.
    for name, axes in accumulations.items():
        count = 0
        axis = []
        acc_shape = []
        edges = []

        group_axes = NxAxes()
        group_axes.append(
            [
                NxAxis.create(
                    values=["sum", "max"], name="accumulator", indices=[0], unit=""
                )
            ]
        )
        for ax_name, ax in axis_definitions.items():
            if ax_name in axes:
                axis.append(ax.primary_axis)
                count += 1
            else:
                match ax.density:
                    case AxisDensity.CONTINUOUS:
                        values = args.data_source.continuous_axis_values(ax)
                        edges.append(None)
                    case AxisDensity.SPARSE:
                        values = args.data_source.sparse_axis_edges(ax)[1:]
                        edges.append(values)
                acc_shape.append(len(values))
                nx_axis = NxAxis.create(
                    values=values,
                    name=ax.name,
                    indices=[ax.primary_axis + 1 - count],
                    unit=ax.units,
                )
                group_axes.append([nx_axis])

        final_accumulations[name] = Accumulation(
            name=name,
            axis=tuple(axis),
            axis_edges=edges,
            shape=Shape(acc_shape),
        )
        counts_name = f"{_count_subentry_name()}_{name}"
        count_accumulations[counts_name] = Accumulation(
            name=counts_name,
            axis=tuple(axis),
            axis_edges=edges,
            shape=Shape(acc_shape),
        )

        nxs.root[name] = NXsubentry(
            NXdata(
                signal=create_field(
                    dtype=dtype,
                    shape=(2, *acc_shape),
                    compression=args.field_options.compression,
                    compression_opts=args.field_options.compression_opts,
                    chunks=None,
                    shuffle=args.field_options.shuffle,
                    fillvalue=0,
                )
            )
        )
        nxs.root[counts_name] = NXsubentry(
            NXdata(
                signal=create_field(
                    dtype=np.uint16,
                    shape=(2, *acc_shape),
                    compression=args.field_options.compression,
                    compression_opts=args.field_options.compression_opts,
                    chunks=None,
                    shuffle=args.field_options.shuffle,
                    fillvalue=0,
                )
            )
        )
        group_axes.add_to_group(nxs.root[name]["data"])
        group_axes.add_to_group(nxs.root[counts_name]["data"])

    return final_accumulations, count_accumulations


def _unique(coords, shape):
    linear: Int1Dp = np.ravel_multi_index(coords, shape)
    order = np.argsort(linear)
    linear = linear[order]

    unique_mask = np.diff(linear) != 0
    unique_mask = np.append(True, unique_mask)

    return coords[:, order][:, unique_mask]


def write_data(
    nxs: NexusFile,
    args: ProcessArgs,
    memory_chunk: Chunk,
    full_shape: Shape,
    sparse_axis: list[Axis],
    chunk_data: np.ndarray | MultiCOO,
    data_chunks: dict[str, Chunker],
) -> tuple[np.ndarray | None] | tuple[sparse.COO, sparse.COO]:
    if len(sparse_axis) == 0:
        if not isinstance(chunk_data, np.ndarray):
            raise ValueError("Data is not sparse, expected a full block of data.")

        for data_entry in data_chunks:
            assert data_entry != _count_subentry_name()
            nxs.root[data_entry].data.signal[*memory_chunk] = chunk_data
        return chunk_data, None
    else:
        if len(sparse_axis) != 1:
            raise NotImplementedError("Only 1 sparse axis is supported.")
        if isinstance(chunk_data, np.ndarray):
            raise ValueError(
                "Recived a sparse axis, with dense data. The data should be sparse. "
            )

        try:
            final_data, counts = chunk_data.sort(full_shape).acc_duplicates(
                full_shape, count=True
            )
        except ValueError:
            ic(full_shape)
            ic(memory_chunk.shape)
            ic(chunk_data.coords.shape)
            ic(np.min(chunk_data.coords, axis=1))
            ic(np.max(chunk_data.coords, axis=1))
            raise

        signal_data = sparse.COO(
            coords=final_data.coords,
            data=final_data.signal,
            shape=full_shape,
            sorted=True,
            has_duplicates=False,
            prune=False,
        )

        axis_data = sparse.COO(
            coords=final_data.coords,
            data=final_data.axis[0],
            shape=full_shape,
            sorted=True,
            has_duplicates=False,
            prune=False,
        )

        count_data = sparse.COO(
            coords=final_data.coords,
            data=counts,
            shape=full_shape,
            sorted=True,
            has_duplicates=False,
            prune=False,
        )

        axis = sparse_axis[0]

        for data_entry, data_chunker in data_chunks.items():
            cbounds = ContainedBounds.from_chunk(data_chunker.data_shape, memory_chunk)

            chunk_edges = cbounds.chunk_edges(data_chunker.chunk_shape)

            coords = np.stack(
                [
                    np.searchsorted(
                        chunk_edges[ii], final_data.coords[ii, :], side="right"
                    )
                    for ii in range(final_data.coords.shape[0])
                ],
            )

            coords = _unique(coords - 1, data_chunker.chunk_count) + 1
            # coords = np.unique(coords, axis=1)

            unique_chunk_count = coords.shape[1]
            total_chunk_count = np.prod([len(edges) - 1 for edges in chunk_edges])

            # This is true where two coords are NOT adjacent.
            # Note that this assumes the coords array is flattened and sorted, so that you cannot zig zag through the data.
            is_adjacent = np.sum(np.abs(np.diff(coords, axis=1)), axis=0) != 1
            adjacent = np.argwhere(is_adjacent).flatten()
            adjacent = np.concatenate([[0], adjacent + 1])

            starts = coords[:, adjacent[:-1]]
            ends = coords[:, adjacent[1:] - 1]
            # Assert that each continuouse block is continuous in only one diemnsion.
            # In principle we could have contigouse blocks across multiple blocks,
            # but this requres them to be full, and the above method does not garantee that.
            assert np.all(np.sum((ends - starts != 0), axis=0) <= 1)

            for ii in tqdm(
                range(starts.shape[1]),
                desc=f"Writing chunks for {data_entry} (density: {unique_chunk_count / total_chunk_count:.2f})",
                leave=False,
            ):
                chunk = Chunk(
                    [
                        slice(chunk_edges[jj][sc - 1], chunk_edges[jj][ec], None)
                        for jj, (sc, ec) in enumerate(zip(starts[:, ii], ends[:, ii]))
                    ]
                )

                if data_entry == _count_subentry_name():
                    signal_chunk = count_data[*chunk]
                else:
                    signal_chunk = signal_data[*chunk]

                axis_chunk = axis_data[*chunk]
                signal_chunk = signal_data[*chunk]
                assert signal_chunk.nnz != 0
                assert axis_chunk.nnz != 0
                assert axis_chunk.nnz == signal_chunk.nnz

                nxs.root[data_entry].data.signal[*chunk] = signal_chunk.todense()
                nxs.root[data_entry].data[f"{axis.name}_exact"][*chunk] = (
                    axis_chunk.todense()
                )
        return signal_data, count_data


def accumulate_data(
    accumulations: dict[str, Accumulation],
    count_accumulations: dict[str, Accumulation],
    memory_chunk: Chunk,
    data: np.ndarray | sparse.COO,
    counts: None | sparse.COO,
):

    total = len(accumulations) + len(count_accumulations) if counts is not None else 0

    with tqdm(total=total, desc="Accumulating", leave=False) as progress:
        for accumulation in accumulations.values():
            accumulation.add(data, memory_chunk)
            progress.update()
        if counts is not None:
            for accumulation in count_accumulations.values():
                accumulation.add(counts, memory_chunk)
                progress.update()


def process(args: ProcessArgs, config: dict[str, Any] = {}):
    assert args.in_path.exists(), f"The input file {args.in_path} was not found"

    nxs = NexusFile(args.out_path, mode="w")
    with nxs.as_context():
        for key, value in args.data_source.instrament_metadata().items():
            nxs.instrument.attrs[key] = value
        for key, value in args.data_source.experiment_metadata().items():
            nxs.experiment.attrs[key] = value

        print(f"Processing file {args.in_path} and writing results to {args.out_path}")
        full_shape, density = args.data_source.shape()
        print(f" Giving a final data shape of {full_shape}")

        data_chunks = provision_subentries(nxs, args, full_shape, density != 1.0)
        axis_definitions, any_sparse_axis = provision_data_axis(nxs, args, data_chunks)

        signal_width = np.dtype(args.data_source.signal_type()).itemsize
        size_per_item = signal_width + np.sum(
            [np.dtype(ax.dtype).itemsize for ax in axis_definitions.values()]
        )

        memory_max_item_count = int(args.memory_max_byte_count / size_per_item)

        memory_chunks, total_read_count = choose_memory_buffer(
            args, memory_max_item_count, density, data_chunks
        )

        accumulations, count_accumulations = provision_accumulation_subentries(
            nxs, args, full_shape, axis_definitions
        )

        sparse_axis = [
            ax for ax in axis_definitions.values() if ax.density == AxisDensity.SPARSE
        ]

        print(
            f"Using a memory chunk shape {memory_chunks.chunk_shape} and count {memory_chunks.chunk_count}."
        )
        print(
            f" Dense usage: {np.prod(memory_chunks.chunk_shape)} items ({format_bytes(np.prod(memory_chunks.chunk_shape) * size_per_item)})."
        )
        if density < 1:
            print(
                f" Sparse usage: {int(np.prod(memory_chunks.chunk_shape) * density)} items ({format_bytes(np.prod(memory_chunks.chunk_shape) * size_per_item * density)}), worst case density {density:.2f}."
            )
        print("With data blocks:")
        print(
            f"maximum chunk size ({format_bytes(args.field_options.max_bytes_per_chunk)})"
        )
        for name, chunker in data_chunks.items():
            width = (
                np.dtype(np.uint16).itemsize
                if name == _count_subentry_name()
                else signal_width
            )
            print(
                f"    {name: >10}: chunk shape {chunker.chunk_shape} and total count {chunker.chunk_count} and memory count {count_chunks_to_cover(memory_chunks.chunk_shape, chunker.chunk_shape)}."
            )
            print(
                f"    {' ' * 10}: chunk size: {np.prod(chunker.chunk_shape)} items ({format_bytes(np.prod(chunker.chunk_shape) * width)})."
            )

        with tqdm(desc="Overall reads", total=total_read_count) as overall_reads_timer:
            data_source_lock = Lock()
            accumulation_lock = Lock()
            nexus_file_lock = Lock()

            def process_chunk(memory_chunk: Chunk):
                local_store = local()
                with data_source_lock:
                    local_store.chunk_data = args.data_source.fill_chunk(
                        memory_chunk, sparse_axis, overall_reads_timer.update
                    )

                with nexus_file_lock:
                    local_store.written_signal, local_store.written_count = write_data(
                        nxs,
                        args,
                        memory_chunk,
                        full_shape,
                        sparse_axis,
                        local_store.chunk_data,
                        data_chunks,
                    )
                    del local_store.chunk_data

                with accumulation_lock:
                    accumulate_data(
                        accumulations,
                        count_accumulations,
                        memory_chunk,
                        local_store.written_signal,
                        local_store.written_count,
                    )
                pass

            outer_chunks = [chunk for chunk in memory_chunks.chunks()]
            with cfutures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(process_chunk, memory_chunk)
                    for memory_chunk in outer_chunks
                ]

                for memory_chunk in tqdm(
                    cfutures.as_completed(futures),
                    total=len(outer_chunks),
                    desc="Memory chunks",
                    leave=True,
                ):
                    try:
                        memory_chunk.result()
                    except:
                        for f in futures:
                            f.cancel()
                        raise

        for name, accumulation in tqdm(
            [*_items(accumulations), *_items(count_accumulations)],
            desc="Writing accumulations",
        ):
            if accumulation.has_data:
                extra_slices = [slice(None) for _ in range(0, accumulation.ndim)]
                nxs.root[name].data.signal[0, *extra_slices] = accumulation.max_data
                nxs.root[name].data.signal[1, *extra_slices] = accumulation.sum_data
