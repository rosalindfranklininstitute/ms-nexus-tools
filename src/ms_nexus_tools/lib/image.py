# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Generic, Any
from enum import Enum
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .dtypes import Number2D, Number1D, Float1D32, Number
from .utils import slice_from_values
from .plotting import Plottable


class OriginLocation(Enum):
    UPPER_LEFT = "upper left"
    UPPER_RIGHT = "upper right"
    LOWER_LEFT = "lower left"
    LOWER_RIGHT = "lower right"


def adjust_origin(
    a: np.ndarray,
    new: OriginLocation,
    current: OriginLocation = OriginLocation.UPPER_LEFT,
) -> np.ndarray:

    left = (OriginLocation.UPPER_LEFT, OriginLocation.LOWER_LEFT)
    right = (OriginLocation.UPPER_RIGHT, OriginLocation.LOWER_RIGHT)

    upper = (OriginLocation.UPPER_LEFT, OriginLocation.UPPER_RIGHT)
    lower = (OriginLocation.LOWER_LEFT, OriginLocation.LOWER_RIGHT)

    axis = []
    if (current in left and new in right) or (current in right and new in left):
        axis.append(1)
    if (current in upper and new in lower) or (current in lower and new in upper):
        axis.append(0)

    if len(axis) > 0:
        return np.flip(a, axis=tuple(axis))
    else:
        return a


@dataclass
class XYRectangle:
    x_start: float
    x_stop: float
    y_start: float
    y_stop: float

    def x_slice(self, x_values: Float1D32) -> slice:
        return slice_from_values(self.x_start, self.x_stop, x_values)

    def y_slice(self, y_values: Float1D32) -> slice:
        return slice_from_values(self.y_start, self.y_stop, y_values)

    def get_plot_rect(self, **kwargs) -> Rectangle:
        x = self.x_start
        w = self.x_stop - x
        y = self.y_start
        h = self.y_stop - y
        return Rectangle((x, y), w, h, **kwargs)


def plot_image(
    ax: plt.Axes,
    image: Number2D,
    x_values: Number1D | None = None,
    y_values: Number1D | None = None,
    xy_rectangles: list[Plottable[XYRectangle]] = [],
    diff_selector=np.median,
    **kwargs,
) -> tuple[Any, tuple[Number, Number]]:
    im_min, im_max = np.percentile(image, [0, 100])
    if x_values is None and y_values is not None:
        x_values = np.array([ii for ii in range(image.shape[0])])
    elif x_values is not None and y_values is None:
        y_values = np.array([ii for ii in range(image.shape[1])])

    if x_values is None and y_values is None:
        im = ax.imshow(image, **kwargs)
    else:
        assert x_values is not None
        assert y_values is not None

        xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
        mnx = np.min(x_values)
        mxx = np.max(x_values)
        mny = np.min(y_values)
        mxy = np.max(y_values)
        dx = diff_selector(np.diff(x_values))
        dy = diff_selector(np.diff(y_values))
        img, xedges, yedges = np.histogram2d(
            xx.ravel(),
            yy.ravel(),
            weights=image.ravel(),
            bins=[
                np.arange(mnx - dx / 2, mxx + dx / 2, dx),
                (np.arange(mny - dy / 2, mxy + dy / 2, dy)),
            ],
        )
        im = ax.imshow(img.T, extent=(mnx, mxx, mny, mxy), **kwargs)

    for jj, xy_rect in enumerate(xy_rectangles):
        rect = xy_rect.value.get_plot_rect(
            linewidth=2,
            edgecolor=xy_rect.color,
            facecolor=xy_rect.color,
            alpha=0.3,
        )
        ax.add_patch(rect)
        ax.text(
            *rect.get_bbox().max,
            xy_rect.title,
            color=xy_rect.color,
            fontsize=12,
        )
    return im, (im_min, im_max)
