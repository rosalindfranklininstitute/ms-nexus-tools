from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .dtypes import Number1D, Float1D32
from .utils import slice_from_values
from .plotting import Plottable


@dataclass
class SpecSlice:
    start: float
    stop: float

    def slice(self, values: Float1D32) -> slice:
        return slice_from_values(self.start, self.stop, values)

    def mask(self, values: np.ndarray) -> tuple[np.ndarray, ...]:
        return np.nonzero((values >= self.start) & (values < self.stop))


def plot_spectrum(
    ax: plt.Axes,
    counts: Number1D,
    values: Number1D,
    spec_slices: list[Plottable[SpecSlice]],
) -> None:
    ax.plot(values, counts)

    for spec_slice in spec_slices:
        x = spec_slice.value.start
        w = spec_slice.value.stop - spec_slice.value.start
        rect = Rectangle(
            (x, 0),
            width=w,
            height=1,
            transform=ax.get_xaxis_transform(),
            linewidth=2,
            edgecolor=spec_slice.color,
            facecolor=spec_slice.color,
            alpha=0.3,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            1.01,
            spec_slice.title,
            transform=ax.get_xaxis_transform(),
            fontsize=12,
            color=spec_slice.color,
        )

    return
