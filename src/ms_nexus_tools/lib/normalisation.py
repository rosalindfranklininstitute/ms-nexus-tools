# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
import numpy as np
from scipy.interpolate import PchipInterpolator

from .bounds import Shape
from .utils import reduce_shape, iterate, indices


class Norm(Enum):
    NONE = "none"
    MAX = "max"
    TIC = "tic"


class Accumulator(Enum):
    TIC = "tic"
    MAX = "max"
    MIN = "min"
    MED = "med"
    P25 = "p25"
    P75 = "p75"


EMPTY = np.full((1,), np.nan)


def normalise(data, norm: Norm):
    match norm:
        case Norm.NONE:
            return data
        case Norm.TIC:
            return data / np.sum(data)
        case Norm.MAX:
            return data / np.max(data)


def norm(data, norm: Norm):
    match norm:
        case Norm.NONE:
            return 1
        case Norm.TIC:
            return np.sum(data)
        case Norm.MAX:
            return np.max(data)


def _should_operate(ndims, axis):
    if axis is None:
        return True
    elif isinstance(axis, int):
        return axis < ndims
    elif len(axis) != 1:
        return True
    else:
        return axis[0] < ndims


def _operate(operation, current, new, axis):
    if _should_operate(len(new.shape), axis):
        inc_value = operation(new, axis=axis)
    else:
        inc_value = new
    if np.isnan(current).all():
        return inc_value
    else:
        return operation([current, inc_value], axis=0)


class P2Histogram:
    """
    This calculates an approximate histogram with b equally probable cells.
    This is an approximation based on a parabolic interpolation.
    i.e. it approaches the true percentiles as the number of samples increases.
    This is a vectorised implementation of the histogram described by

    Jain, Raj, and Imrich Chlamtac.
    'The P2 Algorithm for Dynamic Calculation of Quantiles and Histograms without Storing Observations'.
    Communications of the ACM 28, no. 10 (1985): 1076–85.
    https://doi.org/10.1145/4372.4378.
    """

    def __init__(self, b: int, shape: Shape = (1,)):
        self.b = b
        self.shape = (1,) if shape == tuple() else shape
        self.selectors = (slice(None),) * len(self.shape)
        self._heights = np.zeros((*self.shape, b + 1))
        self.positions = np.zeros((*self.shape, b + 1))

        self.ii = 0

        self._k = np.arange(0, self.b + 1)

    @property
    def heights(self) -> np.ndarray:
        if self.ii == 0:
            return np.full(self._heights.shape, np.nan)
        if self.ii <= self.b:
            return np.moveaxis(
                np.percentile(
                    self._heights[*self.selectors, 0 : self.ii],
                    [0, 25, 50, 75, 100],
                    axis=-1,
                ),
                source=0,
                destination=-1,
            )
        return self._heights

    def heights_for(self, percentile_inx) -> np.ndarray:
        return self.heights[*self.selectors, percentile_inx]

    def add(self, x):

        if np.isscalar(x):
            if self.shape != (1,) and self.shape:
                raise ValueError(
                    f"Provided a scalar, but expected data with shape {self.shape}"
                )
            x = np.array([x])
        elif isinstance(x, np.ndarray):
            if x.shape != self.shape:
                raise ValueError(
                    f"Provided data with shape {x.shape}, but expected a shape of {self.shape}"
                )
        else:
            raise NotImplementedError(
                "Input value must be either a scalar or and array"
            )

        if self.ii <= self.b:
            self._heights[*self.selectors, self.ii] = x
            self.positions[*self.selectors, self.ii] = self.ii
            if self.ii == self.b:
                sort_inx = np.argsort(self._heights, axis=-1)
                self._heights = np.take_along_axis(self._heights, sort_inx, axis=-1)
        else:
            padding_shape = [(0, 0) for _ in range(len(self.shape))]
            padding_shape.append((1, 1))

            min_value = x < self._heights[*self.selectors, 0]
            self._heights[min_value, 0] = x[min_value]

            max_value = x > self._heights[*self.selectors, -1]
            self._heights[max_value, -1] = x[max_value]

            mask = x[*self.selectors, None] < self._heights[*self.selectors, self._k]

            self.positions[mask] += 1

            self.positions[*self.selectors, 0] = 0
            self.positions[*self.selectors, -1] = self.ii

            desired_positions = np.arange(self.b + 1) * self.ii / self.b
            d = (desired_positions - self.positions)[*self.selectors, 1:-1]
            d_minus = d <= -1
            d_plus = d >= 1

            spaces = np.diff(self.positions, axis=-1)
            n_minus = spaces[*self.selectors, :-1]
            n_plus = spaces[*self.selectors, 1:]

            d_mask = ((n_plus > 1) * (d_plus)) + ((n_minus > 1) * (d_minus))
            d_minus *= d_mask
            d_plus *= d_mask

            if np.any(d_mask):
                d[d_minus] = -1
                d[d_plus] = 1
                new_q = P2Histogram._interpolate(
                    n_minus,
                    n_plus,
                    self.selectors,
                    self._heights,
                    d,
                    d_mask,
                    d_minus,
                    d_plus,
                )

                mask = np.full(self._heights.shape, False)
                mask[*self.selectors, 1:-1] = d_mask
                self._heights[mask] = new_q[d_mask]
                self.positions[mask] += d[d_mask]
        self.ii += 1

    @staticmethod
    def _parabola(q_minus, q, q_plus, d, n_minus, n_plus):
        return q + (
            d
            / (n_plus + n_minus)
            * (
                (n_minus + d) * (q_plus) / (n_plus)
                + (n_plus - d) * (q_minus) / (n_minus)
            )
        )

    @staticmethod
    def _linear(q_minus, q, q_plus, d_minus, d_plus, n_minus, n_plus):
        result = np.zeros(q.shape)
        result[d_minus] = q[d_minus] - (q_minus[d_minus]) / (n_minus[d_minus])
        result[d_plus] = q[d_plus] + (q_plus[d_plus]) / (n_plus[d_plus])
        return result

    @staticmethod
    def _interpolate(n_minus, n_plus, selectors, q, d, d_mask, d_minus, d_plus):
        q_mid = q[*selectors, 1:-1]
        q_diff = np.diff(q, axis=-1)
        q_minus = q_diff[*selectors, :-1]
        q_plus = q_diff[*selectors, 1:]

        quad_q = P2Histogram._parabola(
            q_minus,
            q_mid,
            q_plus,
            d,
            n_minus,
            n_plus,
        )

        q_mask = (quad_q > q[*selectors, :-2]) * (q[*selectors, 2:] > quad_q) * d_mask

        l_mask = (q_mask == False) * d_mask
        if np.any(l_mask):
            l_minus = l_mask * d_minus
            l_plus = l_mask * d_plus
            quad_q[l_minus] = q_mid[l_minus] - (q_minus[l_minus]) / (n_minus[l_minus])
            quad_q[l_plus] = q_mid[l_plus] + (q_plus[l_plus]) / (n_plus[l_plus])

        return quad_q


class IncrementalAccumulator:
    def __init__(self, axis=None):
        self.p2: P2Histogram | None = None
        self.tic: np.ndarray = EMPTY
        self.axis: tuple[int, ...] | None = axis

    @property
    def min(self) -> np.ndarray:
        if self.p2 is None:
            return np.array([np.nan])
        return self.p2.heights_for(0)

    @property
    def p25(self) -> np.ndarray:
        if self.p2 is None:
            return np.array([np.nan])
        return self.p2.heights_for(1)

    @property
    def med(self) -> np.ndarray:
        if self.p2 is None:
            return np.array([np.nan])
        return self.p2.heights_for(2)

    @property
    def p75(self) -> np.ndarray:
        if self.p2 is None:
            return np.array([np.nan])
        return self.p2.heights_for(3)

    @property
    def max(self) -> np.ndarray:
        if self.p2 is None:
            return np.array([np.nan])
        return self.p2.heights_for(4)

    def add(self, data, axis=None):

        axis_to_use = axis if axis is not None else self.axis
        if self.p2 is None:
            self.p2 = P2Histogram(b=4, shape=reduce_shape(data.shape, axis_to_use))
        self.tic = _operate(np.nansum, self.tic, data, axis_to_use)
        for chunk in iterate(data, axis_to_use):
            self.p2.add(chunk)

    def __getitem__(self, index: str | Accumulator) -> np.ndarray:
        match index:
            case Accumulator.MIN | Accumulator.MIN.value:
                return self.min
            case Accumulator.P25 | Accumulator.P25.value:
                return self.p25
            case Accumulator.MED | Accumulator.MED.value:
                return self.med
            case Accumulator.P75 | Accumulator.P75.value:
                return self.p75
            case Accumulator.MAX | Accumulator.MAX.value:
                return self.max
            case Accumulator.TIC | Accumulator.TIC.value:
                return self.tic
            case _:
                raise KeyError(f"Could not find specified accumulator ")

    def is_empty(self, index: str | Accumulator) -> bool:
        value = self[index]
        if len(value) == 1:
            return bool(np.isnan(value).all())
        else:
            return False
