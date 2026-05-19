# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
import sys
import bisect

import random

import pytest
import numpy as np

from ms_nexus_tools import lib as nxlib

from icecream import ic, install

install()


def f2(array):
    return f"[{', '.join(f'{a:.2f}' for a in array)}] {array.shape}"


class P2Histogram:
    def __init__(self, b: int, shape: nxlib.bounds.Shape = (1,)):
        self.b = b
        self.shape = shape
        self.selectors = (slice(None),) * len(shape)
        self.heights = np.zeros((*shape, b + 1))
        self.positions = np.zeros((*shape, b + 1))

        self.ii = 0

    def add(self, x):

        if np.isscalar(x):
            if self.shape != (1,):
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
            self.heights[*self.selectors, self.ii] = x
            self.positions[*self.selectors, self.ii] = self.ii
            if self.ii == self.b:
                sort_inx = np.argsort(self.heights, axis=-1)
                self.heights = np.take_along_axis(self.heights, sort_inx, axis=-1)
        else:
            padding_shape = [(0, 0) for _ in range(len(self.shape))]
            padding_shape.append((1, 1))

            min_value = x < self.heights[*self.selectors, 0]
            self.heights[min_value, 0] = x[min_value]

            max_value = x > self.heights[*self.selectors, -1]
            self.heights[max_value, -1] = x[max_value]

            for k in range(self.b, 0, -1):
                mask = x < self.heights[*self.selectors, k]
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
                q = self.heights[*self.selectors, 1:-1]
                q_minus = np.diff(self.heights, axis=-1)[*self.selectors, :-1]
                q_plus = np.diff(self.heights, axis=-1)[*self.selectors, 1:]

                d[d < 0] = -1
                d[d > 0] = 1

                quad_q = np.zeros(d_mask.shape)
                quad_q[d_mask] = P2Histogram._parabola(
                    q_minus[d_mask],
                    q[d_mask],
                    q_plus[d_mask],
                    d[d_mask],
                    n_minus[d_mask],
                    n_plus[d_mask],
                )

                q_mask = (
                    (quad_q > self.heights[*self.selectors, :-2])
                    * (self.heights[*self.selectors, 2:] > quad_q)
                    * d_mask
                )
                padded_q_mask = np.pad(
                    q_mask,
                    pad_width=padding_shape,
                    constant_values=np.full((len(padding_shape), 2), False),
                )
                self.heights[padded_q_mask] = quad_q[q_mask]

                l_mask = (q_mask == False) * d_mask
                if np.any(l_mask):
                    lin_q = np.zeros(d_mask.shape)
                    lin_q[l_mask] = P2Histogram._linear(
                        q_minus[l_mask],
                        q[l_mask],
                        q_plus[l_mask],
                        d[l_mask],
                        n_minus[l_mask],
                        n_plus[l_mask],
                    )
                    padded_l_mask = np.pad(
                        l_mask,
                        pad_width=padding_shape,
                        constant_values=np.full((len(padding_shape), 2), False),
                    )
                    self.heights[padded_l_mask] = lin_q[l_mask]

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


def test_percentiles():

    # Data from the paper
    data = np.array(
        [
            0.02,
            0.15,
            0.74,
            3.39,
            0.83,
            22.37,
            10.15,
            15.43,
            38.62,
            15.92,
            34.60,
            10.28,
            1.47,
            0.40,
            0.05,
            11.39,
            0.27,
            0.42,
            0.09,
            11.37,
        ]
    )
    expected_results = np.array([0.02, 0.5232, 4.402, 11.29, 38.62])
    percentiles = np.percentile(data, [0, 25, 50, 75, 100])

    p2_perc = P2Histogram(b=4)

    for ii, x in enumerate(data):
        p2_perc.add(x)

    np.testing.assert_allclose(p2_perc.heights[0, :], expected_results, rtol=1e-3)

    # Extra 100 random data points
    new_data = np.random.random((100)) * 40
    for x in new_data:
        p2_perc.add(x)
    data = np.concat([data, new_data])
    percentiles = np.percentile(data, [0, 25, 50, 75, 100])

    error = 1 - np.dot(
        percentiles / np.linalg.norm(percentiles),
        (p2_perc.heights / np.linalg.norm(p2_perc.heights))[0, :],
    )
    r2 = np.sqrt(np.average(np.pow(percentiles - p2_perc.heights[0, :], 2)))
    assert r2 < 1.7
    assert error < 1e-2

    # Extra 1000 random data points
    new_data = np.random.random((1000)) * 40
    for x in new_data:
        p2_perc.add(x)
    data = np.concat([data, new_data])
    percentiles = np.percentile(data, [0, 25, 50, 75, 100])

    error = 1 - np.dot(
        percentiles / np.linalg.norm(percentiles),
        (p2_perc.heights / np.linalg.norm(p2_perc.heights))[0, :],
    )
    r2 = np.sqrt(np.average(np.pow(percentiles - p2_perc.heights[0, :], 2)))
    assert r2 < 0.18
    assert error < 1e-4


def test_percentiles_2d():

    # Data from the paper
    data = np.array(
        [
            0.02,
            0.15,
            0.74,
            3.39,
            0.83,
            22.37,
            10.15,
            15.43,
            38.62,
            15.92,
            34.60,
            10.28,
            1.47,
            0.40,
            0.05,
            11.39,
            0.27,
            0.42,
            0.09,
            11.37,
        ]
    )
    expected_results = np.array([0.02, 0.5232, 4.402, 11.29, 38.62])
    data = np.tile(data, (2, 1)).T
    expected_results = np.tile(expected_results, (2, 1))
    ic(data.shape)
    ic(expected_results.shape)
    percentiles = np.percentile(data, [0, 25, 50, 75, 100], axis=0).T
    ic(percentiles.shape)
    ic(percentiles)

    p2_perc = P2Histogram(
        b=4,
        shape=(2,),
    )

    for ii, x in enumerate(data):
        p2_perc.add(x)

    np.testing.assert_allclose(p2_perc.heights, expected_results, rtol=1e-3)

    # Extra 100 random data points
    new_data = np.random.random((100, 2)) * 40
    for x in new_data:
        p2_perc.add(x)
    data = np.concat([data, new_data])
    percentiles = np.percentile(data, [0, 25, 50, 75, 100], axis=0).T

    ic(percentiles)
    ic(p2_perc.heights)

    r2 = np.sqrt(np.average(np.pow(percentiles - p2_perc.heights, 2)))
    assert r2 < 1.7

    # Extra 1000 random data points
    new_data = np.random.random((1000, 2)) * 40
    for x in new_data:
        p2_perc.add(x)
    data = np.concat([data, new_data])
    percentiles = np.percentile(data, [0, 25, 50, 75, 100], axis=0).T

    r2 = np.sqrt(np.average(np.pow(percentiles - p2_perc.heights, 2)))
    assert r2 < 0.18
