from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

from .bounds import Shape
from .normalisation import Accumulator, IncrementalAccumulator


class Filter(ABC):
    def __init__(self, shape: Shape):
        self._image = np.zeros(shape[0:2])
        self._spectrum = np.zeros((shape[2],))

    def add_image(self, bin: int, image: np.ndarray):
        if self._spectrum[bin] == 1:
            return
        self._spectrum[bin] = 1

        self._process_image(bin, image)

    def add_spectra(self, w: int, h: int, spectrum: np.ndarray):
        if self._image[w, h] == 1:
            return
        self._image[w, h] = 1

        self._process_spectra(w, h, spectrum)

    def clear(self):
        self._image[:] = 0
        self._spectrum[:] = 0
        self._clear()

    @abstractmethod
    def _clear(self):
        pass

    @abstractmethod
    def _process_image(self, bin: int, image: np.ndarray):
        pass

    @abstractmethod
    def _process_spectra(self, w: int, h: int, spectrum: np.ndarray):
        pass


class TotalImages(Filter):
    def __init__(self, shape: Shape):
        super().__init__(shape)
        self.tic_image = np.zeros(shape[0:2])
        self.tic_spectrum = np.zeros((shape[2],))

        self.max_image = np.zeros(shape[0:2])
        self.max_spectrum = np.zeros((shape[2],))

    def _clear(self):
        self.tic_image = np.zeros(self.tic_image.shape)
        self.tic_spectrum = np.zeros(self.tic_spectrum.shape)

        self.max_image = np.zeros(self.tic_image.shape)
        self.max_spectrum = np.zeros(self.tic_spectrum.shape)

    def _process_image(self, bin: int, image: np.ndarray):
        self.tic_image[:, :] += image[:, :]
        self.tic_spectrum[bin] = np.sum(image)

        self.max_image[:, :] = np.max([self.max_image, image], axis=0)
        self.max_spectrum[bin] = np.max(image)

        return image

    def _process_spectra(self, w: int, h: int, spectrum: np.ndarray):
        self.tic_image[w, h] = np.sum(spectrum)
        self.tic_spectrum[:] += spectrum[:]

        self.max_image[w, h] = np.max(spectrum)
        self.max_spectrum[:] = np.max([self.max_spectrum, spectrum], axis=0)

        return spectrum

    def image(self, accumulator: Accumulator):
        match accumulator:
            case Accumulator.TIC:
                return self.tic_image
            case Accumulator.MAX:
                return self.max_image

    def spectrum(self, accumulator: Accumulator):
        match accumulator:
            case Accumulator.TIC:
                return self.tic_spectrum
            case Accumulator.MAX:
                return self.max_spectrum


class MassRangeTotalImage(Filter):
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

    def _clear(self):
        self.tic_image = np.zeros(self.tic_image.shape)
        self.tic_spectrum = np.zeros(self.tic_spectrum.shape)

        self.max_image = np.zeros(self.tic_image.shape)
        self.max_spectrum = np.zeros(self.tic_spectrum.shape)

    def _process_image(self, bin: int, image: np.ndarray):
        shift_bin = bin - self._start
        if shift_bin < 0 or shift_bin >= self._width:
            return
        self.tic_image[:, :] += image[:, :]
        self.tic_spectrum[shift_bin] = np.sum(image)

        self.max_image[:, :] = np.max([self.max_image, image], axis=0)
        self.max_spectrum[shift_bin] = np.max(image)

    def _process_spectra(self, w: int, h: int, spectrum: np.ndarray):
        self.tic_image[w, h] = np.sum(spectrum[self.slice()])
        self.tic_spectrum[:] += spectrum[self.slice()]

        self.max_image[w, h] = np.max(spectrum[self.slice()])
        self.max_spectrum[:] = np.max(
            [self.max_spectrum, spectrum[self.slice()]], axis=0
        )

    def image(self, accumulator: Accumulator):
        match accumulator:
            case Accumulator.TIC:
                return self.tic_image
            case Accumulator.MAX:
                return self.max_image

    def spectrum(self, accumulator: Accumulator):
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
    def accumulate_images(mass_images: list["MassRangeTotalImage"], accumulator):
        image_acc = IncrementalAccumulator(axis=2)

        for mi in mass_images:
            image_acc.add(mi.image(accumulator))

        return image_acc[accumulator]
