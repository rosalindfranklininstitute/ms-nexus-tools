from dataclasses import dataclass
import os
import shutil
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import h5py as h5

from nexusformat.nexus import NXfield, NXdata, nxload
from nexusformat.nexus.tree import (
    NXentry,
    NXlinkfield,
    NXsubentry,
    NXinstrument,
    NXprocess,
    NXparameters,
)

from ..lib import Timer, time_this, chunking
from .api import arg_field, ArgType


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


class IONImageBounds(chunking.ImageBounds):
    def spectrum_path(self, layer: int, w: int, h: int) -> str:
        return f"/Spectra/Layer{layer + 1:0{self._layer_count_digits}}/Pixel{w:0{self._layer_width_digits}},{h:0{self._layer_height_digits}}"

    def mass_image_path(self, layer: int, bin: int) -> str:
        return f"/MassImages/Layer{layer + 1:0{self._layer_count_digits}}/Bin{bin:0{self._spectrum_length_digits}}"

    def shape(self) -> tuple[int, int, int, int]:
        return (
            self.layer_count,
            self.layer_width,
            self.layer_height,
            self.spectrum_length,
        )


@dataclass
class SubprocessArgs:
    id: int
    vds_in: Path
    data_path: str
    chunk: chunking.ChunkBounds
    hdf_out: Path


def process_chunk_on_disk(args: SubprocessArgs):
    if len(args.chunk.layer_range()) == 0 or len(args.chunk.spectra_range()) == 0:
        return

    entry = NXentry()
    process = NXprocess()
    process.attrs["name"] = "collect masses"
    process.input = NXparameters(
        hdf_in_file=args.vds_in,
    )
    entry["process"] = process

    try:
        entry["data"] = NXfield(
            dtype="int32",
            shape=[
                args.chunk.layer_count(),
                args.chunk.layer_width(),
                args.chunk.layer_height(),
                args.chunk.spectrum_length(),
            ],
        )

        with h5.File(args.vds_in, "r") as hdf:
            entry.data[:, :, :, :] = hdf[args.data_path][
                args.chunk.layer,
                args.chunk.width,
                args.chunk.height,
                args.chunk.spectra,
            ]
    except:
        print(args.chunk, flush=True)
        raise

    entry.save(args.hdf_out)


@dataclass
class ThreadArgs:
    id: int
    vds_in: h5.File
    data_path: str
    chunk: chunking.ChunkBounds
    hdf_out: NXdata


def process_chunk_in_memory(args: ThreadArgs):
    args.hdf_out.signal[
        args.chunk.layer, args.chunk.width, args.chunk.height, args.chunk.spectra
    ] = args.vds_in[args.data_path,][
        args.chunk.layer, args.chunk.width, args.chunk.height, args.chunk.spectra
    ]


def process_metadata(
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
    hdf_vds_path: Path,
    hdf_in_path: Path,
    bounds: IONImageBounds,
    append: bool = True,
):
    with h5.File(hdf_vds_path, "a" if append else "w") as vds:
        spectra_layout = h5.VirtualLayout(
            shape=bounds.shape(),
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
    hdf_vds_path: Path, hdf_in_path: Path, bounds: IONImageBounds, append: bool = True
):
    with h5.File(hdf_vds_path, "a" if append else "w") as vds:
        mass_layout = h5.VirtualLayout(
            shape=bounds.shape(),
            dtype="int32",
        )
        for layer in range(bounds.layer_count):
            for bin in range(bounds.spectrum_length):
                path = bounds.mass_image_path(layer, bin)
                mass_layout[layer, :, :, bin] = h5.VirtualSource(
                    hdf_in_path,
                    path,
                    shape=(bounds.layer_width, bounds.layer_height),
                )
        vds.create_virtual_dataset("mass_images", mass_layout, fillvalue=0)


def process_field_in_memory(
    data: NXdata, vds: h5.File, field_name: str, chunks: list[chunking.ChunkBounds]
):
    chunk_count = len(chunks)
    with Timer(field_name, interval=30, total=chunk_count, skip_percent=5) as tmr:
        for ii, chunk in enumerate(chunks):
            data.signal[chunk.layer, chunk.width, chunk.height, chunk.spectra] = vds[
                field_name
            ][chunk.layer, chunk.width, chunk.height, chunk.spectra]
            tmr.report(ii)


def process_field_on_disk(
    data: NXdata,
    processors: int,
    hdf_vds_path: Path,
    field_name: str,
    chunks: list[chunking.ChunkBounds],
    tmp_data_path: Path,
):
    with Timer(field_name, interval=30):
        chunked_bins: list[SubprocessArgs] = [
            SubprocessArgs(
                id=i,
                vds_in=hdf_vds_path,
                data_path=field_name,
                chunk=chunk,
                hdf_out=tmp_data_path.joinpath(f"Chunked_{field_name}_{i}.nxs"),
            )
            for i, chunk in enumerate(chunks)
        ]
        with Pool(processors) as p:
            print(
                f" Begining chunking: {len(chunked_bins)} chunks over {processors} processors."
            )
            p.map(process_chunk_on_disk, chunked_bins)

        for ii, chunk_args in enumerate(chunked_bins):
            if chunk_args.hdf_out.exists():
                chunk_root = nxload(chunk_args.hdf_out)
                chunk = chunk_args.chunk
                data.signal[
                    chunk.layer,
                    chunk.width,
                    chunk.height,
                    chunk.spectra,
                ] = chunk_root.entry.data[:, :, :, :]
            else:
                continue


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
            entry["instrument"], bounds, axis = process_metadata(args.hdf_in_path)

        spectra_chunks, image_chunks, memory_info = chunking.calculate_chunks(
            args.chunk_count, args.max_memory, args.processors, bounds
        )

        entry["spectra"] = NXsubentry(
            NXdata(
                NXfield(
                    dtype="int32",
                    shape=bounds.shape(),
                    compression=args.compression,
                    compression_opts=args.compression_level,
                ),
                axis.as_list(),
            )
        )

        entry["mass_images"] = NXsubentry(
            NXdata(
                NXfield(
                    dtype="int32",
                    shape=bounds.shape(),
                    compression=args.compression,
                    compression_opts=args.compression_level,
                ),
                axis.as_list(),
            )
        )

        entry["data"] = NXlinkfield("entry/spectra/data")
        with time_this("VDS"):
            hdf_vds_path = args.tmp_data_path.joinpath("vds.h5")
            if args.do_spectra:
                create_spectra_vds(hdf_vds_path, args.hdf_in_path, bounds, append=False)
            if args.do_images:
                create_image_vds(
                    hdf_vds_path, args.hdf_in_path, bounds, append=args.do_spectra
                )

        with h5.File(args.hdf_in_path, "r"):
            with h5.File(hdf_vds_path, "r") as vds:
                if not args.on_disk:
                    if args.do_spectra:
                        process_field_in_memory(
                            entry.spectra.data, vds, "spectra", spectra_chunks
                        )

                    if args.do_images:
                        process_field_in_memory(
                            entry.mass_images.data, vds, "mass_images", image_chunks
                        )
                else:
                    if args.do_spectra:
                        process_field_on_disk(
                            entry.spectra.data,
                            args.processors,
                            hdf_vds_path,
                            "spectra",
                            spectra_chunks,
                            args.tmp_data_path,
                        )

                    if args.do_images:
                        process_field_on_disk(
                            entry.mass_images.data,
                            args.processors,
                            hdf_vds_path,
                            "mass_images",
                            image_chunks,
                            args.tmp_data_path,
                        )
