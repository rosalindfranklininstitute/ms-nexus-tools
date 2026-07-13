# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Iterable, Generator
from threading import Lock, local
import concurrent.futures as cfutures
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import numpy.typing as npt
import sparse

from icecream import ic

from tqdm import tqdm

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
    InvalidAxisError,
)
from ..lib.multi_coo import (
    MultiCOO,
)
from ..lib.bounds import Chunk, Shape
from ..lib import ContainedBounds
from ..lib.chunker import Chunker, count_chunks_to_cover
from ..lib.nxs import (
    NexusFile,
    FieldOptions,
    NxAxis,
    create_field,
    NxAxes,
    create_group,
)
from ..lib.utils import format_bytes
from ..lib.dtypes import Intp1D, Bool1D


def _count_subentry_name() -> str:
    return "item_counts"


def _items(d: dict[str, Any]) -> list[tuple[str, Any]]:
    return list(d.items())


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

    chunk_max_byte_count: int = arg_field(
        "--chunk-bytes",
        help="The maximum number of bytes of a chunk in the output file.",
        default=1024 * 1024 * 8,
    )  # 8Mb

    memory_max_byte_count: int = arg_field(
        "--memory-bytes",
        help="The maximum number of bytes to use as the memory buffer. Each thread uses this much memory.",
        default=1024 * 1024 * 1024 * 4,
    )

    data_source: AbstractDataSource = no_arg_field(default=None)

    field_options: FieldOptions = no_arg_field(
        default=FieldOptions(
            compression=hdf5plugin.Blosc(),
            compression_opts=None,
            max_bytes_per_chunk=-1,
            shuffle=True,
        ),
    )


class DataChunks:
    def __init__(
        self,
        names: Iterable[str],
        chunkers: Iterable[Chunker],
        dtypes: Iterable[npt.DTypeLike],
    ):
        self.names: set[str] = set(names)
        self.chunkers: dict[str, Chunker] = dict(zip(names, chunkers, strict=True))
        self.dtypes: dict[str, npt.DTypeLike] = dict(zip(names, dtypes, strict=True))

    @staticmethod
    def from_dict(data: dict[str, tuple[Chunker, npt.DTypeLike]]) -> "DataChunks":
        names = set(data.keys())
        chunkers = [c[0] for c in data.values()]
        dtypes = [c[1] for c in data.values()]
        return DataChunks(names, chunkers, dtypes)

    def __setitem__(self, name: str, data: tuple[Chunker, npt.DTypeLike]) -> None:
        self.names.add(name)
        self.chunkers[name] = data[0]
        self.dtypes[name] = data[1]

    def __repr__(self) -> str:
        return ",\n".join(
            [f"{name} ({dtype}): {chunker}" for name, chunker, dtype in self.items()],
        )

    def chunker(self, name: str) -> Chunker:
        return self.chunkers[name]

    def dtype(self, name: str) -> npt.DTypeLike:
        return self.dtypes[name]

    def items(self) -> Generator[tuple[str, Chunker, npt.DTypeLike]]:
        for name in self.names:
            yield name, self.chunker(name), self.dtype(name)


def choose_memory_buffer(
    args: ProcessArgs,
    max_item_count: int,
    density: float,
    data_chunks: DataChunks,
) -> tuple[Chunker, int]:
    memory_chunks = {
        name: Chunker.find_chunk_multiple(
            chunker.data_shape,
            chunker.chunk_shape,
            max_item_count / density,
        )
        for name, chunker, _ in data_chunks.items()
    }

    min_read_count = np.pow(2, 32, dtype=np.int64)
    min_read_name = ""
    for name, chunker in memory_chunks.items():
        chunker.normalise()
        read_count = 0
        for chunk in chunker.chunks():
            read_count += args.data_source.chunk_read_count(chunk.shape)
        if read_count < min_read_count:
            min_read_count = read_count
            min_read_name = name
    assert len(min_read_name) > 0

    return memory_chunks[min_read_name], min_read_count


def choose_memory_buffer_and_data_chunks(
    args: ProcessArgs,
    full_shape: Shape,
    density: float,
) -> tuple[Chunker, int, int, DataChunks]:
    data_priorities = args.data_source.output_chunks()
    if len(data_priorities) == 0:
        raise ValueError("At least one dataset must be provided.")
    for name, priorities in data_priorities.items():
        if len(name.strip()) == 0:
            raise ValueError(
                "An invalid name was returned for a data set. Names must not be empty.",
            )
        if len(priorities) != len(full_shape):
            raise ValueError(
                f"An invalid set of priorities was returned for dataset {name}: there should be {len(full_shape)} items, but only {len(priorities)} were provided.",
            )
    signal_type = args.data_source.signal_type()
    signal_item_width = np.dtype(signal_type).itemsize
    data_max_items = {
        name: int(args.field_options.max_bytes_per_chunk / signal_item_width)
        for name in data_priorities
    }

    data_chunks = DataChunks([], [], [])

    for name, priorities in data_priorities.items():
        data_chunks[name] = (
            Chunker.from_max_item_count(
                data_shape=full_shape,
                priorities=priorities,
                items_per_chunk=data_max_items[name],
            ),
            signal_type,
        )

    axis_definitions = args.data_source.axis_definitions()
    if any(ax.density == AxisDensity.BINNED for ax in axis_definitions):
        counts_item_width = np.dtype(np.uint16).itemsize
        data_max_items[_count_subentry_name()] = int(
            args.field_options.max_bytes_per_chunk / counts_item_width,
        )
        data_chunks[_count_subentry_name()] = (
            Chunker.from_max_item_count(
                data_shape=full_shape,
                priorities=tuple(1 for _ in full_shape),
                items_per_chunk=data_max_items[_count_subentry_name()],
            ),
            np.uint16,
        )

    size_per_item = signal_item_width + np.sum(
        [
            np.dtype(ax.dtype).itemsize
            for ax in axis_definitions
            if ax.density == AxisDensity.BINNED
        ],
    )
    memory_max_item_count = int(args.memory_max_byte_count / size_per_item)
    memory_chunks, total_read_count = choose_memory_buffer(
        args,
        memory_max_item_count,
        density,
        data_chunks,
    )
    for name, chunker, dtype in data_chunks.items():
        data_chunks[name] = (
            Chunker.from_max_item_count(
                data_shape=chunker.data_shape,
                priorities=chunker.priorities,
                items_per_chunk=data_max_items[name],
                min_chunk_count=memory_chunks.chunk_count,
            ),
            dtype,
        )

    return memory_chunks, total_read_count, size_per_item, data_chunks


def provision_subentries(
    nxs: NexusFile,
    args: ProcessArgs,
    data_chunks: DataChunks,
) -> None:

    for name, chunker, dtype in data_chunks.items():
        nxs.root[name] = NXsubentry(
            NXdata(
                signal=create_field(
                    dtype=dtype,
                    shape=chunker.data_shape,
                    compression=args.field_options.compression,
                    compression_opts=args.field_options.compression_opts,
                    chunks=chunker.chunk_shape,
                    shuffle=args.field_options.shuffle,
                    fillvalue=0,
                ),
            ),
        )


def provision_data_axis(
    nxs: NexusFile,
    args: ProcessArgs,
    full_shape: Shape,
    data_chunks: DataChunks,
) -> tuple[dict[str, Axis], bool]:
    axis_definitions = args.data_source.axis_definitions()

    any_binned_axis = False
    for entry_name, chunker, _ in data_chunks.items():
        group_axes = NxAxes()
        for _ in full_shape:
            group_axes.append([])
        for axis in axis_definitions:
            match axis.density:
                case AxisDensity.CONTINUOUS:
                    values = args.data_source.continuous_axis_values(axis)
                    if len(values) != full_shape[axis.primary_axis]:
                        raise InvalidAxisError(
                            f"Expected {full_shape[axis.primary_axis]} values for {axis.name} but recived {len(values)}.",
                        )
                    nx_axis = NxAxis.create(
                        values=values,
                        name=axis.name,
                        indices=[axis.primary_axis],
                        unit=axis.units,
                    )
                    group_axes[axis.primary_axis].append(nx_axis)
                case AxisDensity.BINNED:
                    if axis.primary_axis != len(full_shape) - 1:
                        raise InvalidAxisError(
                            "Only BINNED axis with primary_axis == (last dimension) are supported."
                        )
                    any_binned_axis = True
                    all_axis: list[int] = list(range(axis.primary_axis + 1))
                    values = args.data_source.binned_axis_edges(axis)[1:]
                    if len(values) != full_shape[axis.primary_axis]:
                        raise InvalidAxisError(
                            f"Expected {full_shape[axis.primary_axis] + 1} edges for {axis.name} but recived {len(values) + 1}.",
                        )

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
                    group_axes[axis.primary_axis].extend([nx_axis, nx_axis_exact])
                case _:
                    raise InvalidAxisError(f"Unknown Axis density: {axis.density}")
        group_axes.add_to_group(nxs.root[entry_name]["data"])

    return {ax.name: ax for ax in axis_definitions}, any_binned_axis


@dataclass
class Accumulation:
    name: str
    axis: tuple[int, ...]
    axis_edges: list[None | np.ndarray]
    shape: Shape

    contains_binned_axes: bool = field(init=False)
    max_data: np.ndarray = field(init=False)
    sum_data: np.ndarray = field(init=False)
    ndim: int = field(init=False)
    has_data: bool = field(init=False)

    def __post_init__(self):

        self.contains_binned_axes = False
        for edge in self.axis_edges:
            if edge is not None:
                self.contains_binned_axes = True
                break

        self.max_data = np.zeros(self.shape)
        self.sum_data = np.zeros(self.shape)
        self.ndim = len(self.shape)
        self.has_data = False

    def add(self, data: np.ndarray | sparse.COO, chunk: Chunk) -> None:
        sub_chunk = Chunk([chunk[ii] for ii in range(data.ndim) if ii not in self.axis])

        sub_data = data[*chunk] if isinstance(data, sparse.COO) else data

        max_data = np.maximum(self.max_data[*sub_chunk], sub_data.max(axis=self.axis))
        sum_data = np.add(self.sum_data[*sub_chunk], sub_data.sum(axis=self.axis))

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
    nxs: NexusFile,
    args: ProcessArgs,
    shape: Shape,
    axis_definitions: dict[str, Axis],
) -> tuple[dict[str, Accumulation], dict[str, Accumulation]]:
    accumulations = args.data_source.output_accumulations()
    dtype = args.data_source.signal_type()

    final_accumulations: dict[str, Accumulation] = {}
    count_accumulations: dict[str, Accumulation] = {}

    for ac_name, axes in accumulations.items():
        axis_to_accumulate: Bool1D = np.full(shape=(len(shape),), fill_value=False)
        edges = []

        group_axes = NxAxes()
        group_axes.append(
            [
                NxAxis.create(
                    values=["sum", "max"],
                    name="accumulator",
                    indices=[0],
                    unit="",
                ),
            ],
        )
        has_binned_axis = False

        for ax_name, ax in axis_definitions.items():
            if ax_name in axes:
                axis_to_accumulate[ax.primary_axis] = True

        final_dim = np.sum(axis_to_accumulate == False)  # noqa: E712
        acc_shape: Intp1D = np.zeros((final_dim,), dtype=np.intp)
        for _ in range(final_dim):
            group_axes.append([])

        for ax in axis_definitions.values():
            if not axis_to_accumulate[ax.primary_axis]:
                match ax.density:
                    case AxisDensity.CONTINUOUS:
                        values = args.data_source.continuous_axis_values(ax)
                        edges.append(None)
                    case AxisDensity.BINNED:
                        values = args.data_source.binned_axis_edges(ax)[1:]
                        edges.append(values)
                        has_binned_axis = True
                    case _:
                        raise InvalidAxisError(f"Unknown Axis density: {ax.density}")
                count = len(values)
                new_index = ax.primary_axis - np.sum(
                    axis_to_accumulate[0 : ax.primary_axis],
                )
                if acc_shape[new_index] == 0:
                    acc_shape[new_index] = count
                elif count != acc_shape[new_index]:
                    raise InvalidAxisError(
                        f"Found conflicting sizes for {new_index}. Initially set to {acc_shape[new_index]} now trying to set to {count}",
                    )

                nx_axis = NxAxis.create(
                    values=values,
                    name=ax.name,
                    indices=[new_index + 1],
                    unit=ax.units,
                )
                group_axes[new_index + 1].append(nx_axis)

        final_axis = tuple(ii for ii, aa in enumerate(axis_to_accumulate) if aa)
        final_accumulations[ac_name] = Accumulation(
            name=ac_name,
            axis=final_axis,
            axis_edges=edges,
            shape=Shape(acc_shape.tolist()),
        )

        nxs.root[ac_name] = NXsubentry(
            NXdata(
                signal=create_field(
                    dtype=dtype,
                    shape=(2, *acc_shape),
                    compression=args.field_options.compression,
                    compression_opts=args.field_options.compression_opts,
                    chunks=None,
                    shuffle=args.field_options.shuffle,
                    fillvalue=0,
                ),
            ),
        )
        group_axes.add_to_group(nxs.root[ac_name]["data"])
        if has_binned_axis:
            counts_name = f"{_count_subentry_name()}_{ac_name}"
            count_accumulations[counts_name] = Accumulation(
                name=counts_name,
                axis=final_axis,
                axis_edges=edges,
                shape=Shape(acc_shape.tolist()),
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
                    ),
                ),
            )
            group_axes.add_to_group(nxs.root[counts_name]["data"])

    return final_accumulations, count_accumulations


def _unique(coords: np.ndarray, shape: Shape) -> np.ndarray:
    linear: Intp1D = np.ravel_multi_index(coords, shape)
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
    binned_axis: list[Axis],
    chunk_data: np.ndarray | MultiCOO,
    data_chunks: DataChunks,
) -> tuple[np.ndarray, None] | tuple[sparse.COO, sparse.COO]:
    if len(binned_axis) == 0:
        if not isinstance(chunk_data, np.ndarray):
            raise ValueError("Data is not binned, expected a full block of data.")

        for data_entry in data_chunks.names:
            assert data_entry != _count_subentry_name()
            nxs.root[data_entry].data.signal[*memory_chunk] = chunk_data
        return chunk_data, None
    if isinstance(chunk_data, np.ndarray):
        raise TypeError(
            "Recived a binned axis, with dense data. The data should be binned. ",
        )

    try:
        final_data, counts = chunk_data.sort(full_shape).acc_duplicates(
            full_shape,
            count=True,
        )
    except ValueError:
        ic(full_shape)
        ic(memory_chunk.shape)
        ic(chunk_data.coords.shape)
        ic(np.min(chunk_data.coords, axis=1))
        ic(np.max(chunk_data.coords, axis=1))
        raise

    coords_data = sparse.COO(
        coords=final_data.coords,
        data=np.arange(len(final_data.signal)),
        shape=full_shape,
        sorted=True,
        has_duplicates=False,
        prune=False,
    )

    signal_data = sparse.COO(
        coords=final_data.coords,
        data=final_data.signal,
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

    axis = binned_axis[0]

    for data_entry, data_chunker, _ in data_chunks.items():
        cbounds = ContainedBounds.from_chunk(data_chunker.data_shape, memory_chunk)

        chunk_edges = cbounds.chunk_edges(data_chunker.chunk_shape)

        coords = np.stack(
            [
                np.searchsorted(
                    chunk_edges[ii],
                    final_data.coords[ii, :],
                    side="right",
                )
                for ii in range(final_data.coords.shape[0])
            ],
        )

        coords = _unique(coords - 1, data_chunker.chunk_count) + 1

        unique_chunk_count = coords.shape[1]
        total_chunk_count = np.prod([len(edges) - 1 for edges in chunk_edges])

        # This is true where two coords are NOT adjacent.
        # Note: that this assumes the coords array is flattened and sorted, so that you cannot zig zag through the data.
        is_not_adjacent = np.diff(coords[-1, :]) != 1

        adjacent = np.argwhere(is_not_adjacent).flatten()
        adjacent = np.concatenate([[0], adjacent + 1, [coords.shape[1]]])

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
                    for jj, (sc, ec) in enumerate(
                        zip(starts[:, ii], ends[:, ii], strict=True),
                    )
                ],
            )

            if data_entry == _count_subentry_name():
                signal_chunk = count_data[*chunk]
            else:
                signal_chunk = signal_data[*chunk]

            assert signal_chunk.nnz != 0

            nxs.root[data_entry].data.signal[*chunk] = signal_chunk.todense()

            coords_chunk = coords_data[*chunk]
            assert coords_chunk.nnz != 0

            coords = tuple([coords_chunk.coords[i, :] for i in range(len(chunk.shape))])
            indices = coords_chunk.data

            # TODO @DMD: Here all the binned axis share the same data type.
            # https://github.com/orgs/rosalindfranklininstitute/projects/19/views/1?pane=issue&itemId=212408503
            dense_axis_values = np.full(chunk.shape, np.nan)

            for axis, axis_data in zip(binned_axis, final_data.axis, strict=True):
                dense_axis_values[coords] = axis_data[indices]
                nxs.root[data_entry].data[f"{axis.name}_exact"][*chunk] = (
                    dense_axis_values
                )
    return signal_data, count_data


def accumulate_data(
    accumulations: dict[str, Accumulation],
    count_accumulations: dict[str, Accumulation],
    memory_chunk: Chunk,
    data: np.ndarray | sparse.COO,
    counts: None | sparse.COO,
) -> None:

    total = len(accumulations) + len(count_accumulations) if counts is not None else 0

    with tqdm(total=total, desc="Accumulating", leave=False) as progress:
        for accumulation in accumulations.values():
            accumulation.add(data, memory_chunk)
            progress.update()
        if counts is not None:
            for accumulation in count_accumulations.values():
                accumulation.add(counts, memory_chunk)
                progress.update()


def add_items_to_group(items: dict[str, Any], root) -> None:
    for key, value in items.items():
        if isinstance(value, dict):
            if key not in root:
                root[key] = create_group()
            add_items_to_group(value, root[key])
        else:
            root.attrs[key] = value


def process(args: ProcessArgs, config: dict[str, Any] = {}) -> None:
    assert args.in_path.exists(), f"The input file {args.in_path} was not found"

    if args.field_options.max_bytes_per_chunk <= 0:
        args.field_options = FieldOptions(
            compression=args.field_options.compression,
            compression_opts=args.field_options.compression_opts,
            max_bytes_per_chunk=args.chunk_max_byte_count,
            shuffle=args.field_options.shuffle,
        )
    else:
        assert args.field_options.max_bytes_per_chunk == args.chunk_max_byte_count

    nxs = NexusFile(args.out_path, mode="w")
    with nxs.as_context():
        add_items_to_group(args.data_source.instrament_metadata(), nxs.instrument)
        add_items_to_group(args.data_source.experiment_metadata(), nxs.experiment)

        full_shape, density = args.data_source.shape()

        memory_chunks, total_read_count, size_per_item, data_chunks = (
            choose_memory_buffer_and_data_chunks(args, full_shape, density)
        )

        provision_subentries(nxs, args, data_chunks)

        axis_definitions, any_binned_axis = provision_data_axis(
            nxs,
            args,
            full_shape,
            data_chunks,
        )

        accumulations, count_accumulations = provision_accumulation_subentries(
            nxs,
            args,
            full_shape,
            axis_definitions,
        )

        binned_axis = [
            ax for ax in axis_definitions.values() if ax.density == AxisDensity.BINNED
        ]

        print(f"Processing file {args.in_path} and writing results to {args.out_path}")
        print(
            f" Giving a final data shape of {full_shape} (Raw {format_bytes(np.prod(full_shape) * size_per_item)})",
        )

        print(
            f"Using a memory chunk shape {memory_chunks.chunk_shape} and count {memory_chunks.chunk_count}.",
        )
        print(
            f" Dense usage: {np.prod(memory_chunks.chunk_shape)} items ({format_bytes(np.prod(memory_chunks.chunk_shape) * size_per_item)}).",
        )
        if len(binned_axis) > 0:
            print(
                f" Binned usage: {int(np.prod(memory_chunks.chunk_shape) * density)} items ({format_bytes(np.prod(memory_chunks.chunk_shape) * size_per_item * density)}), worst case density {density:.2f}.",
            )
        print("With data blocks:")
        print(
            f"maximum chunk size ({format_bytes(args.field_options.max_bytes_per_chunk)})",
        )
        for name, chunker, dtype in data_chunks.items():
            width = np.dtype(dtype).itemsize
            print(
                f"    {name: >10}: chunk shape {chunker.chunk_shape} and total count {chunker.chunk_count} and memory count {count_chunks_to_cover(memory_chunks.chunk_shape, chunker.chunk_shape)}.",
            )
            print(
                f"    {' ' * 10}: chunk size: {np.prod(chunker.chunk_shape)} items ({format_bytes(np.prod(chunker.chunk_shape) * width)}).",
            )

        with tqdm(desc="Overall reads", total=total_read_count) as overall_reads_timer:
            data_source_lock = Lock()
            accumulation_lock = Lock()
            nexus_file_lock = Lock()

            def process_chunk(memory_chunk: Chunk) -> None:
                local_store = local()
                with data_source_lock:
                    local_store.chunk_data = args.data_source.fill_chunk(
                        memory_chunk,
                        binned_axis,
                        overall_reads_timer.update,
                    )

                with nexus_file_lock:
                    local_store.written_signal, local_store.written_count = write_data(
                        nxs,
                        args,
                        memory_chunk,
                        full_shape,
                        binned_axis,
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

            outer_chunks = list(memory_chunks.chunks())
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
                extra_slices = [slice(None) for _ in range(accumulation.ndim)]
                nxs.root[name].data.signal[0, *extra_slices] = accumulation.max_data
                nxs.root[name].data.signal[1, *extra_slices] = accumulation.sum_data
