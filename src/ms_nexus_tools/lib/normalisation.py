# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
import numpy as np

from .bounds import Shape
from .utils import reduce_shape, iterate


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

    @property
    def heights(self):
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

    def heights_for(self, percentile_inx):
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

            for k in range(self.b, 0, -1):
                mask = x < self._heights[*self.selectors, k]
                self.positions[mask, k] += 1

            self.positions[*self.selectors, -1] = self.ii

            desired_positions = np.tile(
                np.arange(self.b + 1) * self.ii / self.b, self.shape
            ).reshape((*self.shape, self.b + 1))
            d = (desired_positions - self.positions)[*self.selectors, 1:-1]

            n_minus = np.diff(self.positions, axis=-1)[*self.selectors, :-1]
            n_plus = np.diff(self.positions, axis=-1)[*self.selectors, 1:]

            d_mask = ((n_plus > 1) * (d >= 1)) + ((n_minus > 1) * (d <= -1))

            if np.any(d_mask):
                d[d < 0] = -1
                d[d > 0] = 1

                new_q, mask = P2Histogram._interpolate(
                    n_minus,
                    n_plus,
                    self.selectors,
                    self._heights,
                    d,
                    d_mask,
                )

                self._heights[mask] = new_q[mask]

                padded_d_mask = np.pad(
                    d_mask,
                    pad_width=padding_shape,
                    constant_values=np.full((len(padding_shape), 2), False),
                )
                self.positions[padded_d_mask] += d[d_mask]
        self.ii += 1

    @staticmethod
    def _parabola(q_minus, q, q_plus, d, n_minus, n_plus):
        return q + d / (n_plus + n_minus) * (
            (n_minus + d) * (q_plus) / (n_plus) + (n_plus - d) * (q_minus) / (n_minus)
        )

    @staticmethod
    def _linear(q_minus, q, q_plus, d, n_minus, n_plus):
        negative = d < 0
        neutral = d == 0
        positive = d > 0
        result = np.zeros(q.shape)
        result[negative] = q[negative] - (q_minus[negative]) / (n_minus[negative])
        result[positive] = q[positive] + (q_plus[positive]) / (n_plus[positive])
        result[neutral] = q[neutral]
        return result

    @staticmethod
    def _interpolate(n_minus, n_plus, selectors, q, d, d_mask):
        padding = np.full(q.shape, False)
        mask = np.full(q.shape, False)
        new_q = q[*selectors, :]
        q_mid = q[*selectors, 1:-1]
        q_minus = np.diff(q, axis=-1)[*selectors, :-1]
        q_plus = np.diff(q, axis=-1)[*selectors, 1:]

        quad_q = np.zeros(d_mask.shape)
        quad_q[d_mask] = P2Histogram._parabola(
            q_minus[d_mask],
            q_mid[d_mask],
            q_plus[d_mask],
            d[d_mask],
            n_minus[d_mask],
            n_plus[d_mask],
        )

        q_mask = (quad_q > q[*selectors, :-2]) * (q[*selectors, 2:] > quad_q) * d_mask
        padding[*selectors, 1:-1] = q_mask
        mask[*selectors, 1:-1] = q_mask
        new_q[padding] = quad_q[q_mask]

        l_mask = (q_mask == False) * d_mask
        if np.any(l_mask):
            lin_q = np.zeros(d_mask.shape)
            lin_q[l_mask] = P2Histogram._linear(
                q_minus[l_mask],
                q_mid[l_mask],
                q_plus[l_mask],
                d[l_mask],
                n_minus[l_mask],
                n_plus[l_mask],
            )

            padding[*selectors, 1:-1] = l_mask
            new_q[padding] = lin_q[l_mask]
            mask[*selectors, 1:-1] += l_mask
        return new_q, mask


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
                raise TypeError(f"Could not find specified accumulator {index}")

    def is_empty(self, index: str | Accumulator) -> bool:
        value = self[index]
        if len(value) == 1:
            return bool(np.isnan(value).all())
        else:
            return False
