# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

from .bounds import Shape
from .normalisation import Accumulator, IncrementalAccumulator, P2Histogram


class MzFilter(ABC):
    def __init__(self, shape: Shape):
        self._image = np.zeros(shape[0:2])
        self._spectrum = np.zeros((shape[2],))

    def add_image(self, z_bin: int, image: np.ndarray) -> None:
        if self._spectrum[z_bin] == 1:
            return
        self._spectrum[z_bin] = 1

        self._process_image(z_bin, image)

    def add_spectra(self, w: int, h: int, spectrum: np.ndarray) -> None:
        if self._image[w, h] == 1:
            return
        self._image[w, h] = 1

        self._process_spectra(w, h, spectrum)

    def clear(self) -> None:
        self._image[:] = 0
        self._spectrum[:] = 0
        self._clear()

    @abstractmethod
    def _clear(self) -> None:
        pass

    @abstractmethod
    def _process_image(self, z_bin: int, image: np.ndarray) -> None:
        pass

    @abstractmethod
    def _process_spectra(self, w: int, h: int, spectrum: np.ndarray) -> None:
        pass


class TotalImages(MzFilter):
    def __init__(self, shape: Shape):
        super().__init__(shape)
        self.tic_image = np.zeros(shape[0:2])
        self.tic_spectrum = np.zeros((shape[2],))

        self.max_image = np.zeros(shape[0:2])
        self.max_spectrum = np.zeros((shape[2],))

    def _clear(self) -> None:
        self.tic_image = np.zeros(self.tic_image.shape)
        self.tic_spectrum = np.zeros(self.tic_spectrum.shape)

        self.max_image = np.zeros(self.tic_image.shape)
        self.max_spectrum = np.zeros(self.tic_spectrum.shape)

    def _process_image(self, z_bin: int, image: np.ndarray) -> None:
        self.tic_image[:, :] += image[:, :]
        self.tic_spectrum[z_bin] = np.sum(image)

        self.max_image[:, :] = np.maximum(self.max_image, image)
        self.max_spectrum[z_bin] = np.max(image)

    def _process_spectra(self, w: int, h: int, spectrum: np.ndarray) -> None:
        self.tic_image[w, h] = np.sum(spectrum)
        self.tic_spectrum[:] += spectrum[:]

        self.max_image[w, h] = np.max(spectrum)
        self.max_spectrum[:] = np.maximum(self.max_spectrum, spectrum)

    def image(self, accumulator: Accumulator) -> np.ndarray:
        match accumulator:
            case Accumulator.TIC:
                return self.tic_image
            case Accumulator.MAX:
                return self.max_image

    def spectrum(self, accumulator: Accumulator) -> np.ndarray:
        match accumulator:
            case Accumulator.TIC:
                return self.tic_spectrum
            case Accumulator.MAX:
                return self.max_spectrum


class AccumulationDirection(Enum):
    IMAGES = "images"
    SPECTRA = "spectra"
    UNKNOWN = "unknown"


class PercentileImages(MzFilter):
    """
    A class that accumulates a histogram of the data in image and spectra form.
    Depending on the direction of accumulation one of these will be explicit (np.percentile) and one will be approximate (P2Histogram)
    In particular the parallel accumulation is approximate while the orthogonal one is explicit.
    i.e. if accumulating by image, the images will be approximated by the P2 algorithm and the spectra will be explicitly calculated using np.percentile.
    """

    def __init__(self, shape: Shape, b: int):
        super().__init__(shape)
        self.percentiles = P2Histogram.percentiles(b)
        self.images_p2 = P2Histogram(b, shape[0:2])
        self.spectra_p2 = P2Histogram(b, (shape[2],))

        self.images_explicit = np.zeros((*self.images_p2.shape, b + 1))
        self.spectra_explicit = np.zeros((*self.spectra_p2.shape, b + 1))

        self.acc_direction = AccumulationDirection.UNKNOWN

    def _clear(self) -> None:
        self.images_p2 = P2Histogram(self.images_p2.b, self.images_p2.shape)
        self.spectra_p2 = P2Histogram(self.spectra_p2.b, self.spectra_p2.shape)

        self.images_explicit = np.zeros((*self.images_p2.shape, self.images_p2.b + 1))
        self.spectra_explicit = np.zeros(
            (*self.spectra_p2.shape, self.spectra_p2.b + 1),
        )

        self.acc_direction = AccumulationDirection.UNKNOWN

    def _process_image(self, z_bin: int, image: np.ndarray) -> None:
        if self.acc_direction == AccumulationDirection.UNKNOWN:
            self.acc_direction = AccumulationDirection.IMAGES
        elif self.acc_direction != AccumulationDirection.IMAGES:
            raise ValueError(
                f"Only a single direction of accumulation is supported. Previously {self.acc_direction.value} was used, now attempting to use spectra.",
            )

        self.images_p2.add(image)
        self.spectra_explicit[z_bin, :] = np.percentile(image, self.percentiles)

    def _process_spectra(self, w: int, h: int, spectrum: np.ndarray) -> None:
        if self.acc_direction == AccumulationDirection.UNKNOWN:
            self.acc_direction = AccumulationDirection.SPECTRA
        elif self.acc_direction != AccumulationDirection.SPECTRA:
            raise ValueError(
                f"Only a single direction of accumulation is supported. Previously {self.acc_direction.value} was used, now attempting to use spectra.",
            )

        self.spectra_p2.add(spectrum)
        self.images_explicit[w, h, :] = np.percentile(spectrum, self.percentiles)

    def all_percentile_images(self) -> np.ndarray[tuple[int, int, int]]:
        match self.acc_direction:
            case AccumulationDirection.IMAGES:
                return self.images_p2.heights
            case AccumulationDirection.SPECTRA:
                return self.images_explicit
            case _:
                raise LookupError("No data collected, cannot return image.")

    def image(self, percentile_index: int) -> np.ndarray[tuple[int, int]]:
        match self.acc_direction:
            case AccumulationDirection.IMAGES:
                return self.images_p2.heights_for(percentile_index)
            case AccumulationDirection.SPECTRA:
                return self.images_explicit[:, :, percentile_index]
            case _:
                raise LookupError("No data collected, cannot return image.")

    def all_percentile_spectrum(self) -> np.ndarray[tuple[int, int]]:
        match self.acc_direction:
            case AccumulationDirection.IMAGES:
                return self.spectra_explicit
            case AccumulationDirection.SPECTRA:
                return self.spectra_p2.heights
            case _:
                raise LookupError("No data collected, cannot return image.")

    def spectrum(self, percentile_index: int) -> np.ndarray[tuple[int]]:
        match self.acc_direction:
            case AccumulationDirection.IMAGES:
                return self.spectra_explicit[:, percentile_index]
            case AccumulationDirection.SPECTRA:
                return self.spectra_p2.heights_for(percentile_index)
            case _:
                raise LookupError("No data collected, cannot return image.")


class MassRangeTotalImage(MzFilter):
    def __init__(self, shape: Shape, mass_index_start: int, mass_index_end: int):
        super().__init__(shape)
        self._start = mass_index_start
        self._stop = mass_index_end
        self._width = self._stop - self._start
        assert self._width > 0

        self.tic_image = np.zeros(shape[0:2])
        self.tic_spectrum = np.zeros((self._width,))

        self.max_image = np.zeros(shape[0:2])
        self.max_spectrum = np.zeros((self._width,))

    def _clear(self) -> None:
        self.tic_image = np.zeros(self.tic_image.shape)
        self.tic_spectrum = np.zeros(self.tic_spectrum.shape)

        self.max_image = np.zeros(self.tic_image.shape)
        self.max_spectrum = np.zeros(self.tic_spectrum.shape)

    def _process_image(self, z_bin: int, image: np.ndarray) -> None:
        shift_bin = z_bin - self._start
        if shift_bin < 0 or shift_bin >= self._width:
            return
        self.tic_image[:, :] += image[:, :]
        self.tic_spectrum[shift_bin] = np.sum(image)

        self.max_image[:, :] = np.max([self.max_image, image], axis=0)
        self.max_spectrum[shift_bin] = np.max(image)

    def _process_spectra(self, w: int, h: int, spectrum: np.ndarray) -> None:
        self.tic_image[w, h] = np.sum(spectrum[self.slice()])
        self.tic_spectrum[:] += spectrum[self.slice()]

        self.max_image[w, h] = np.max(spectrum[self.slice()])
        self.max_spectrum[:] = np.max(
            [self.max_spectrum, spectrum[self.slice()]],
            axis=0,
        )

    def image(self, accumulator: Accumulator) -> np.ndarray:
        match accumulator:
            case Accumulator.TIC:
                return self.tic_image
            case Accumulator.MAX:
                return self.max_image

    def spectrum(self, accumulator: Accumulator) -> np.ndarray:
        match accumulator:
            case Accumulator.TIC:
                return self.tic_spectrum
            case Accumulator.MAX:
                return self.max_spectrum

    def slice(self) -> slice:
        return slice(self._start, self._stop)

    def range(self) -> range:
        return range(self._start, self._stop)

    def width(self) -> int:
        return self._width

    @staticmethod
    def accumulate_images(
        mass_images: list["MassRangeTotalImage"],
        accumulator,
    ) -> Optional[np.ndarray]:
        image_acc = IncrementalAccumulator(axis=2)

        if len(mass_images) == 0:
            return None

        for mi in mass_images:
            image_acc.add(mi.image(accumulator))

        if image_acc.is_empty(accumulator):
            return None
        return image_acc[accumulator]
