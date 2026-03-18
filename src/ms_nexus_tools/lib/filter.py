from abc import ABC, abstractmethod
import numpy as np

from .bounds import Shape


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

    @abstractmethod
    def _process_image(self, bin: int, image: np.ndarray):
        pass

    @abstractmethod
    def _process_spectra(self, w: int, h: int, spectrum: np.ndarray):
        pass


class TotalImages(Filter):
    def __init__(self, shape: Shape):
        super().__init__(shape)
        self.total_image = np.zeros(shape[0:2])
        self.total_spectrum = np.zeros((shape[2],))

    def _process_image(self, bin: int, image: np.ndarray):
        self.total_image[:, :] += image[:, :]
        self.total_spectrum[bin] = np.sum(image)
        return image

    def _process_spectra(self, w: int, h: int, spectrum: np.ndarray):
        self.total_image[w, h] = np.sum(spectrum)
        self.total_spectrum[:] += spectrum[:]
        return spectrum


class MassRangeTotalImage(Filter):
    def __init__(self, shape: Shape, mass_index_start: int, mass_index_end: int):
        super().__init__(shape)
        self._start = mass_index_start
        self._stop = mass_index_end
        self._width = self._stop - self._start
        assert self._width > 0
        self.total_image = np.zeros(shape[0:2])

    def _process_image(self, bin: int, image: np.ndarray):
        shift_bin = bin - self._start
        if shift_bin < 0 or shift_bin >= self._width:
            return
        self.total_image[:, :] += image[:, :]

    def _process_spectra(self, w: int, h: int, spectrum: np.ndarray):
        self.total_image[w, h] = np.sum(spectrum[self._start : self._stop])
