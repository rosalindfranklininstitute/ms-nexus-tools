from dataclasses import dataclass
import os
import shutil
from pathlib import Path
from multiprocessing import JoinableQueue as mQueue, Process
from threading import Thread
from queue import Queue as tQueue

import numpy as np
import h5py as h5

from nexusformat.nexus import NXfield, NXdata, nxload
from nexusformat.nexus.tree import (
    NXentry,
    NXlinkfield,
    NXsubentry,
    NXinstrument,
)

from ..lib import Timer, time_this
from ..lib.chunking import (
    ImageBounds,
    ChunkBounds,
    calculate_chunks,
    OnDiskArgs,
    InMemoryArgs,
    process_chunk_on_disk,
    process_chunk_in_memory,
)
from .api import arg_field, ArgType

timer_interval = 30


@dataclass
class ProcessArgs:
    hdf_in_path: Path = arg_field(
        "-i",
        "--input",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        doc="The input file.",
    )
    hdf_out_path: Path = arg_field(
        "-o",
        "--output",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        doc="The output file.",
    )
    tmp_data_path: Path = arg_field(
        default=Path("./tmp_data"),
        doc="The folder used to store intermediary files used in processing. This can be used for resuming interupted runns, if not overridden.",
    )

    chunk_count: int = arg_field(
        "-k", default=1, doc="How many intermediate chunks to use."
    )
    max_memory: float = arg_field(
        "-m",
        default=None,
        doc="The maximum memory a chunk should take, in Gb.",
    )

    processors: int = arg_field(
        "-j",
        default=1,
        doc="Process the data using sub-processors on disk using this many processors. If 1 it does not chunk onto disk, but remains in memory.",
    )
    on_disk: bool = arg_field(
        action="store_true",
        doc="If present will store intermediate chunks to the temp_data_path folder. Useful for resuming or when using multiple processors.",
    )
    use_subprocesses: bool = arg_field(
        action="store_true",
        doc="If present will use subprocesses instead of threads.",
    )

    compression: str = arg_field(
        doc="The type of compression to use, if any.", default="gzip"
    )
    compression_level: int = arg_field(
        doc="The level of compression to use, if appropriate.", default=4
    )

    do_spectra: bool = arg_field(
        "--no-spectra",
        action="store_false",
        doc="If present, will not process the spectra part of the input file.",
    )
    do_images: bool = arg_field(
        "--no-images",
        action="store_false",
        doc="If present, will not process the images part of the input file.",
    )


@dataclass
class ImageAxis:
    layer_axis: NXfield
    x_axis: NXfield
    y_axis: NXfield
    mass_axis: NXfield

    def as_list(self):
        return [self.layer_axis, self.x_axis, self.y_axis, self.mass_axis]


class IONImageBounds(ImageBounds):
    def spectrum_path(self, layer: int, w: int, h: int) -> str:
        return f"/Spectra/Layer{layer + 1:0{self._layer_count_digits}}/Pixel{w:0{self._layer_width_digits}},{h:0{self._layer_height_digits}}"

    def image_path(self, layer: int, bin: int) -> str:
        return f"/MassImages/Layer{layer + 1:0{self._layer_count_digits}}/Bin{bin:0{self._spectrum_length_digits}}"


def write_metadata(
    hdf_out_path: Path,
    instrument: NXinstrument,
    image_bounds: IONImageBounds,
    image_axis: ImageAxis,
    append: bool = False,
) -> None:
    with h5.File(hdf_out_path, "a" if append else "w") as hdf:
        metadata = hdf.require_group("ExperimentDetails").attrs

        def assign_or_assert(value, name):
            if name in metadata:
                assert metadata[name][0] == value
            else:
                metadata[name] = [value]

        for key, value in instrument.items():
            metadata.attrs[key] = value
        assign_or_assert(image_bounds.layer_height, "LayerDimensionX")
        assign_or_assert(image_bounds.layer_width, "LayerDimensionY")
        assign_or_assert(image_bounds.layer_height, "LayerDimensionX")
        assign_or_assert(image_bounds.layer_width, "LayerDimensionY")
        assign_or_assert(image_bounds.layer_count, "Layers")
        assign_or_assert(image_bounds.spectrum_length, "SpectrumLength")

        assign_or_assert(image_axis.x_axis[1], "ImageMicronsX")
        assign_or_assert(image_axis.y_axis[1], "ImageMicronsY")

        hdf.require_dataset(
            "ExperimentDetails/MassArray",
            dtype="int32",
            shape=(1, image_bounds.spectrum_length),
            data=image_axis.mass_axis[:],
            exact=True,
        )


def write_spectrum(
    hdf_out_path: Path,
    bounds: IONImageBounds,
    data: np.ndarray,
    append: bool = True,
) -> None:
    assert data.shape == bounds.shape
    with h5.File(hdf_out_path, "a" if append else "w") as hdf:
        for ll in range(bounds.layer_count):
            for ww in range(bounds.layer_width):
                for hh in range(bounds.layer_height):
                    hdf.require_dataset(
                        bounds.spectrum_path(ll, ww, hh),
                        dtype="int32",
                        shape=(1, bounds.spectrum_length),
                        data=data[ll, ww, hh, :],
                        exact=True,
                    )


def write_image(
    hdf_out_path: Path,
    bounds: IONImageBounds,
    data: np.ndarray,
    append: bool = True,
) -> None:
    assert data.shape == bounds.shape
    with h5.File(hdf_out_path, "a" if append else "w") as hdf:
        for ll in range(bounds.layer_count):
            for ss in range(bounds.spectrum_length):
                hdf.require_dataset(
                    bounds.image_path(ll, ss),
                    dtype="int32",
                    shape=(bounds.layer_width, bounds.layer_height),
                    data=data[ll, :, :, ss],
                    exact=True,
                )


def read_metadata(
    hdf_in_path: Path,
) -> tuple[NXinstrument, IONImageBounds, ImageAxis]:
    with h5.File(hdf_in_path, "r") as hdfinfile:
        metadata = hdfinfile["ExperimentDetails"].attrs
        chunk_height = metadata["LayerDimensionX"][0]
        chunk_width = metadata["LayerDimensionY"][0]
        layer_height = metadata["LayerDimensionX"][0]
        layer_width = metadata["LayerDimensionY"][0]
        layer_count = metadata["Layers"][0]
        spectrum_length = metadata["SpectrumLength"][0]
        x_microns = metadata["ImageMicronsX"][0]
        y_microns = metadata["ImageMicronsY"][0]

        instrument = NXinstrument()
        for key, value in metadata.items():
            instrument.attrs[key] = value

        mass_axis = NXfield(hdfinfile["ExperimentDetails/MassArray"][:], name="mass")

        image_bounds = IONImageBounds(
            int(layer_count), int(layer_width), int(layer_height), int(spectrum_length)
        )
    layer_axis = NXfield(np.arange(1, layer_count + 1, 1.0), name="layer")
    x_axis = NXfield(
        np.arange(0, chunk_width, 1.0) * x_microns, name="x", unit="micron"
    )
    y_axis = NXfield(
        np.arange(0, chunk_height, 1.0) * y_microns, name="y", unit="micron"
    )

    image_axis = ImageAxis(layer_axis, x_axis, y_axis, mass_axis)
    return instrument, image_bounds, image_axis


def create_spectra_vds(
    hdf_in_path: Path,
    hdf_vds_path: Path,
    bounds: IONImageBounds,
    append: bool = True,
):
    with h5.File(hdf_vds_path, "a" if append else "w") as vds:
        spectra_layout = h5.VirtualLayout(
            shape=bounds.shape,
            dtype="int32",
        )
        for layer in range(bounds.layer_count):
            for w in range(bounds.layer_width):
                for h in range(bounds.layer_height):
                    path = bounds.spectrum_path(layer, w, h)
                    spectra_layout[layer, w, h, :] = h5.VirtualSource(
                        hdf_in_path, path, shape=(1, bounds.spectrum_length)
                    )
        vds.create_virtual_dataset("spectra", spectra_layout, fillvalue=0)


def create_image_vds(
    hdf_in_path: Path, hdf_vds_path: Path, bounds: IONImageBounds, append: bool = True
):
    with h5.File(hdf_vds_path, "a" if append else "w") as vds:
        mass_layout = h5.VirtualLayout(
            shape=bounds.shape,
            dtype="int32",
        )
        for layer in range(bounds.layer_count):
            for bin in range(bounds.spectrum_length):
                path = bounds.image_path(layer, bin)
                mass_layout[layer, :, :, bin] = h5.VirtualSource(
                    hdf_in_path,
                    path,
                    shape=(bounds.layer_width, bounds.layer_height),
                )
        vds.create_virtual_dataset("images", mass_layout, fillvalue=0)


def queue_consumer(ii: int, function, queue: mQueue | tQueue):
    for args in iter(queue.get, "STOP"):
        function(args)
        queue.task_done()
    print(f"Queue consumer {ii + 1} finishing.")


def process_chunks_on_disk(
    chunks: list[ChunkBounds],
    data_path: str,
    use_subprocesses: bool,
    hdf_vds_path: Path,
    tmp_data_path: Path,
    hdf_out: NXdata,
    processes: int,
):
    if use_subprocesses:
        queue = mQueue()
    else:
        queue = tQueue()

    total = len(chunks)

    def report():
        return (total - queue.qsize()), total

    chunk_args: list[OnDiskArgs] = [
        OnDiskArgs(
            id=ii,
            vds_in=hdf_vds_path,
            data_path=data_path,
            chunk=chunk,
            hdf_out=tmp_data_path.joinpath(f"Chunked_{data_path}_{ii}.nxs"),
        )
        for ii, chunk in enumerate(chunks)
    ]
    with Timer(
        data_path, interval=timer_interval, skip_percent=5, report_callback=report
    ):
        for arg in chunk_args:
            queue.put(arg)

        if use_subprocesses:
            for ii in range(processes):
                Process(
                    target=queue_consumer, args=(ii, process_chunk_on_disk, queue)
                ).start()
        else:
            for ii in range(processes):
                Thread(
                    target=queue_consumer,
                    args=(ii, process_chunk_on_disk, queue),
                    daemon=True,
                ).start()

        queue.join()

    if use_subprocesses:
        for ii in range(processes):
            queue.put("STOP")

    with Timer(
        f"Collecting chunks for '{data_path}'",
        interval=timer_interval,
        skip_percent=5,
        total=len(chunk_args),
    ) as tmr:
        for ii, arg in enumerate(chunk_args):
            if arg.hdf_out.exists():
                chunk_root = nxload(arg.hdf_out)
                chunk = arg.chunk
                hdf_out.signal[
                    chunk.layer,
                    chunk.width,
                    chunk.height,
                    chunk.spectra,
                ] = chunk_root.entry.data[:, :, :, :]
            else:
                continue
            tmr.report(ii)

    queue.join()


def process_chunks_in_memory(
    chunks: list[ChunkBounds],
    data_path: str,
    vds_in: h5.File,
    hdf_out: NXdata,
    processes: int,
):
    queue = tQueue()
    total = len(chunks)

    def report():
        return (total - queue.qsize()), total

    with Timer(
        data_path, interval=timer_interval, skip_percent=5, report_callback=report
    ):
        for ii, chunk in enumerate(chunks):
            queue.put(
                InMemoryArgs(
                    id=ii,
                    vds_in=vds_in,
                    data_path=data_path,
                    chunk=chunk,
                    hdf_out=hdf_out,
                )
            )

        for ii in range(processes):
            Thread(
                target=queue_consumer,
                args=(ii, process_chunk_in_memory, queue),
                daemon=True,
            ).start()

        queue.join()


def process(args: ProcessArgs):

    with time_this("Overall"):
        hdf_out_path = args.hdf_out_path

        if os.path.exists(args.tmp_data_path):
            shutil.rmtree(args.tmp_data_path)
        os.makedirs(args.tmp_data_path)

        entry = NXentry()
        if os.path.exists(hdf_out_path):
            os.remove(hdf_out_path)
        entry.save(hdf_out_path)

        with time_this("metadata"):
            entry["instrument"], bounds, axis = read_metadata(args.hdf_in_path)

        spectra_chunks, image_chunks, memory_info = calculate_chunks(
            args.chunk_count, args.max_memory, args.processors, bounds
        )

        entry["spectra"] = NXsubentry(
            NXdata(
                NXfield(
                    dtype="int32",
                    shape=bounds.shape,
                    compression=args.compression,
                    compression_opts=args.compression_level,
                ),
                axis.as_list(),
            )
        )

        entry["images"] = NXsubentry(
            NXdata(
                NXfield(
                    dtype="int32",
                    shape=bounds.shape,
                    compression=args.compression,
                    compression_opts=args.compression_level,
                ),
                axis.as_list(),
            )
        )

        entry["data"] = NXdata(
            NXlinkfield(entry["spectra/data/signal"]), axis.as_list()
        )
        with time_this("VDS"):
            hdf_vds_path = args.tmp_data_path.joinpath("vds.h5")
            if args.do_spectra:
                create_spectra_vds(args.hdf_in_path, hdf_vds_path, bounds, append=False)
            if args.do_images:
                create_image_vds(
                    args.hdf_in_path, hdf_vds_path, bounds, append=args.do_spectra
                )

        with h5.File(args.hdf_in_path, "r"):
            with h5.File(hdf_vds_path, "r") as vds:
                if args.on_disk:
                    if args.do_spectra:
                        process_chunks_on_disk(
                            chunks=spectra_chunks,
                            data_path="spectra",
                            use_subprocesses=args.use_subprocesses,
                            hdf_vds_path=hdf_vds_path,
                            tmp_data_path=args.tmp_data_path,
                            hdf_out=entry.spectra.data,
                            processes=args.processors,
                        )

                    if args.do_images:
                        process_chunks_on_disk(
                            chunks=image_chunks,
                            data_path="images",
                            use_subprocesses=args.use_subprocesses,
                            hdf_vds_path=hdf_vds_path,
                            tmp_data_path=args.tmp_data_path,
                            hdf_out=entry.images.data,
                            processes=args.processors,
                        )
                else:
                    if args.do_spectra:
                        process_chunks_in_memory(
                            chunks=spectra_chunks,
                            data_path="spectra",
                            vds_in=vds,
                            hdf_out=entry.spectra.data,
                            processes=args.processors,
                        )

                    if args.do_images:
                        process_chunks_in_memory(
                            chunks=image_chunks,
                            data_path="images",
                            vds_in=vds,
                            hdf_out=entry.images.data,
                            processes=args.processors,
                        )
