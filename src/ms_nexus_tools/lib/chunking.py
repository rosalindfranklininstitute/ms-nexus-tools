from dataclasses import dataclass, field
import math
from pathlib import Path

import h5py as h5

from nexusformat.nexus import NXfield, NXdata
from nexusformat.nexus.tree import (
    NXentry,
    NXprocess,
    NXparameters,
)

from .utils import count_digits


@dataclass
class ImageBounds:
    layer_count: int
    layer_width: int
    layer_height: int
    spectrum_length: int
    shape: tuple[int, int, int, int] = field(init=False)

    def __post_init__(self):
        self.shape = (
            self.layer_count,
            self.layer_width,
            self.layer_height,
            self.spectrum_length,
        )
        self._layer_count_digits = count_digits(self.layer_count)
        self._layer_width_digits = count_digits(self.layer_width)
        self._layer_height_digits = count_digits(self.layer_height)
        self._spectrum_length_digits = count_digits(self.spectrum_length)


@dataclass
class ChunkBounds:
    layer: slice
    width: slice
    height: slice
    spectra: slice

    def __hash__(self):
        return hash(
            (
                self.layer.start,
                self.layer.stop,
                self.width.start,
                self.width.stop,
                self.height.start,
                self.height.stop,
                self.spectra.start,
                self.spectra.stop,
            )
        )

    def approximate_size_gb(self) -> float:
        return (
            float(self.layer.stop - self.layer.start)
            * float(self.width.stop - self.width.start)
            * float(self.height.stop - self.height.start)
            * float(self.spectra.stop - self.spectra.start)
            * 4
            / 1024
            / 1024
            / 1024
        )

    def layer_count(self) -> int:
        return self.layer.stop - self.layer.start

    def layer_width(self) -> int:
        return self.width.stop - self.width.start

    def layer_height(self) -> int:
        return self.height.stop - self.height.start

    def spectrum_length(self) -> int:
        return self.spectra.stop - self.spectra.start

    def layer_range(self) -> range:
        return range(self.layer.start, self.layer.stop)

    def width_range(self) -> range:
        return range(self.width.start, self.width.stop)

    def height_range(self) -> range:
        return range(self.height.start, self.height.stop)

    def spectra_range(self) -> range:
        return range(self.spectra.start, self.spectra.stop)

    def to_bound_dict(self) -> dict[str, int]:
        return dict(
            layer_start=self.layer.start,
            layer_stop=self.layer.stop,
            width_stat=self.width.start,
            width_stop=self.width.stop,
            height_stat=self.height.start,
            height_stop=self.height.stop,
            spectra_stat=self.spectra.start,
            spectra_stop=self.spectra.stop,
        )


@dataclass
class MemoryInfo:
    total_gb: float
    max_chunk_gb: float
    min_chunk_count: int

    @staticmethod
    def calculate(
        chunk_count_min: int,
        gb_max: float | None,
        processors: int,
        bounds: ImageBounds,
    ) -> MemoryInfo:
        total_gb = ChunkBounds(
            slice(0, bounds.layer_count),
            slice(0, bounds.layer_width),
            slice(0, bounds.layer_height),
            slice(0, bounds.spectrum_length),
        ).approximate_size_gb()

        assert processors >= 1

        if chunk_count_min <= 0:
            chunk_count_min = 1

        chunk_gb = total_gb / chunk_count_min
        if gb_max is not None:
            assert gb_max > 0.0
            if (chunk_gb * processors) > gb_max:
                chunk_count = math.ceil(total_gb / (gb_max / processors))
            else:
                chunk_count = chunk_count_min
        else:
            chunk_count = chunk_count_min
        chunk_gb = total_gb / chunk_count

        return MemoryInfo(
            total_gb=total_gb, max_chunk_gb=chunk_gb, min_chunk_count=chunk_count
        )


def _chunk(
    bounds: ImageBounds,
    layers_per_chunk,
    width_per_chunk,
    height_per_chunk,
    spectra_per_chunk,
) -> list[ChunkBounds]:

    chunks: list[ChunkBounds] = []
    for ii, layer_start in enumerate(range(0, bounds.layer_count, layers_per_chunk)):
        layer_end = min((ii + 1) * layers_per_chunk, bounds.layer_count)
        for jj, width_start in enumerate(range(0, bounds.layer_width, width_per_chunk)):
            width_end = min((jj + 1) * width_per_chunk, bounds.layer_width)
            for kk, height_start in enumerate(
                range(0, bounds.layer_height, height_per_chunk)
            ):
                height_end = min((kk + 1) * height_per_chunk, bounds.layer_height)
                for ll, spectra_start in enumerate(
                    range(0, bounds.spectrum_length, spectra_per_chunk)
                ):
                    spectra_end = min(
                        (ll + 1) * spectra_per_chunk, bounds.spectrum_length
                    )
                    chunks.append(
                        ChunkBounds(
                            layer=slice(layer_start, layer_end),
                            width=slice(width_start, width_end),
                            height=slice(height_start, height_end),
                            spectra=slice(spectra_start, spectra_end),
                        )
                    )
    return chunks


def chunk_image_dimensions(
    width: int, height: int, chunks_per_image: int
) -> tuple[int, int]:

    if chunks_per_image > (width * height):
        return 1, 1
    chunks_per_image_dimension = math.ceil(math.sqrt(chunks_per_image))
    if width < chunks_per_image_dimension:
        width_per_chunk = 1
        assert height > width
        assert height >= chunks_per_image_dimension
        height_per_chunk = max(1, math.floor(height / (chunks_per_image / width)))
        return width_per_chunk, height_per_chunk
    if height < chunks_per_image_dimension:
        height_per_chunk = 1
        assert width > height
        assert width >= chunks_per_image_dimension
        width_per_chunk = max(1, math.floor(width / (chunks_per_image / height)))
        return width_per_chunk, height_per_chunk

    assert width >= chunks_per_image_dimension
    assert height >= chunks_per_image_dimension
    width_per_chunk = math.floor(width / chunks_per_image_dimension)
    height_per_chunk = math.floor(height / chunks_per_image_dimension)
    return width_per_chunk, height_per_chunk


def _chunk_images_then_spectra(
    bounds: ImageBounds, chunks_per_layer: int, layers_per_chunk
) -> list[ChunkBounds]:

    width_per_chunk, height_per_chunk = chunk_image_dimensions(
        bounds.layer_width, bounds.layer_height, chunks_per_layer
    )
    chunks_per_image = (bounds.layer_width / width_per_chunk) * (
        bounds.layer_height / height_per_chunk
    )
    chunks_per_spectrum = math.ceil(chunks_per_layer / chunks_per_image)
    spectra_per_chunk = max(1, math.floor(bounds.spectrum_length / chunks_per_spectrum))

    return _chunk(
        bounds, layers_per_chunk, width_per_chunk, height_per_chunk, spectra_per_chunk
    )


def _chunk_spectra_then_images(
    bounds: ImageBounds, chunks_per_layer: int, layers_per_chunk: int
) -> list[ChunkBounds]:

    spectra_per_chunk = max(1, math.floor(bounds.spectrum_length / chunks_per_layer))
    chunks_per_spectrum = bounds.spectrum_length / spectra_per_chunk
    chunks_per_image = math.ceil(chunks_per_layer / chunks_per_spectrum)

    width_per_chunk, height_per_chunk = chunk_image_dimensions(
        bounds.layer_width, bounds.layer_height, chunks_per_image
    )

    return _chunk(
        bounds, layers_per_chunk, width_per_chunk, height_per_chunk, spectra_per_chunk
    )


def calculate_chunks(
    chunk_count_min: int,
    gb_max: float | None,
    processors: int,
    bounds: ImageBounds,
) -> tuple[list[ChunkBounds], list[ChunkBounds], MemoryInfo]:

    memory_info = MemoryInfo.calculate(chunk_count_min, gb_max, processors, bounds)

    if memory_info.min_chunk_count < bounds.layer_count:
        layers_per_chunk = max(
            1, math.floor(bounds.layer_count / memory_info.min_chunk_count)
        )
        chunks_per_layer = 1
    else:
        layers_per_chunk = 1
        chunks_per_layer = math.ceil(memory_info.min_chunk_count / bounds.layer_count)

    spectra_chunks = _chunk_images_then_spectra(
        bounds, chunks_per_layer, layers_per_chunk
    )

    image_chunks = _chunk_spectra_then_images(
        bounds, chunks_per_layer, layers_per_chunk
    )

    return spectra_chunks, image_chunks, memory_info


@dataclass
class OnDiskArgs:
    id: int
    vds_in: Path
    data_path: str
    chunk: ChunkBounds
    hdf_out: Path


def process_chunk_on_disk(args: OnDiskArgs):
    if len(args.chunk.layer_range()) == 0 or len(args.chunk.spectra_range()) == 0:
        return

    entry = NXentry()
    process = NXprocess()
    process.attrs["name"] = "collect {args.data_path}"
    process.input = NXparameters(
        hdf_in_file=args.vds_in,
        id=args.id,
        data_path=args.data_path,
        **args.chunk.to_bound_dict(),
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
class InMemoryArgs:
    id: int
    vds_in: h5.File
    data_path: str
    chunk: ChunkBounds
    hdf_out: NXdata


def process_chunk_in_memory(args: InMemoryArgs):
    args.hdf_out.signal[
        args.chunk.layer, args.chunk.width, args.chunk.height, args.chunk.spectra
    ] = args.vds_in[args.data_path][
        args.chunk.layer, args.chunk.width, args.chunk.height, args.chunk.spectra
    ]
