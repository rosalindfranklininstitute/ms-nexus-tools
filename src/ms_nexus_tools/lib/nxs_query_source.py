# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from pathlib import Path

import numpy as np

from tqdm import tqdm

from .bounds import Shape, Chunk
from .nxs import NexusFile
from .query_source import AbstractQuerySource
from .mz_filter import MzFilter, Accumulator
from .chunker import count_chunks_to_cover


class NxsQuerySource(AbstractQuerySource):
    def __init__(self, in_path: Path, tqdm=tqdm):
        super().__init__(name="RFI-MSI-NeXus")
        self.nx_file = NexusFile(in_path, mode="r")

        self._shape = self.nx_file.root.spectra.data.signal.shape
        self._mass_values = self.nx_file.root.spectra.data.mass.nxdata

        self._tqdm = tqdm

    def __enter__(self) -> "NxsQuerySource":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.nx_file.close()

    def shape(self) -> Shape:
        return self._shape

    def mass_values(self) -> np.ndarray[tuple[int], Any]:
        return self._mass_values

    def instrament_metadata(self) -> dict[str, Any]:
        return dict()

    def experiment_metadata(self) -> dict[str, Any]:
        return dict()

    def fill_filters(
        self,
        layer: int,
        bins: list[int],
        xy: list[tuple[int, int]],
        totals: list[MzFilter],
        chunk: Chunk,
    ) -> None:
        spectra_chunking = self.nx_file.root.spectra.data.signal.chunks
        self._spectra_chunk_count = np.prod(
            count_chunks_to_cover(self._shape, spectra_chunking),
        )

        image_chunking = self.nx_file.root.images.data.signal.chunks
        self._images_chunk_count = np.prod(
            count_chunks_to_cover((*self._shape[0:3], len(bins)), image_chunking),
        )  # this is postentially very optimistic

        if self._images_chunk_count < self._spectra_chunk_count:
            for bb in self._tqdm(bins, desc="Images"):
                for inner_image in totals:
                    inner_image.add_image(
                        bb,
                        self.nx_file.root.images.data.signal[
                            layer,
                            chunk[1],
                            chunk[2],
                            bb,
                        ],
                    )
        else:
            for x, y in self._tqdm(xy, desc="Spectra"):
                for image in totals:
                    image.add_spectra(
                        x,
                        y,
                        self.nx_file.root.spectra.data.signal[layer, x, y, :],
                    )

    def accumulated_spectrum(self, accumulator: Accumulator, layer: int) -> np.ndarray:
        match accumulator:
            case Accumulator.TIC:
                return self.nx_file.root.total_spectra.data.signal[0, layer, :].nxdata
            case Accumulator.MAX:
                return self.nx_file.root.total_spectra.data.signal[1, layer, :].nxdata

    def accumulated_image(self, accumulator: Accumulator, layer: int) -> np.ndarray:
        match accumulator:
            case Accumulator.TIC:
                return self.nx_file.root.total_images.data.signal[0, layer, :, :].nxdata
            case Accumulator.MAX:
                return self.nx_file.root.total_images.data.signal[1, layer, :, :].nxdata
