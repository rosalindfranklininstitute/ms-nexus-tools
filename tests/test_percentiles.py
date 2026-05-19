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
    def __init__(self, b: int):
        self.b = b
        self.heights = np.zeros((b + 1))
        self.positions = np.zeros((b + 1))

        self.ii = 0

    def add(self, x):
        if self.ii <= self.b:
            self.heights[self.ii] = x
            self.positions[self.ii] = self.ii
            if self.ii == self.b:
                sort_inx = np.argsort(self.heights)
                heights = np.array([self.heights[inx] for inx in sort_inx])
        else:
            if x < self.heights[0]:
                self.heights[0] = x
                k = 0
            elif self.heights[-1] < x:
                self.heights[-1] = x
                k = self.b - 1
            else:
                k = bisect.bisect_left(self.heights, x) - 1

            self.positions[k + 1 :] += 1

            desired_positions = np.arange(self.b + 1) * self.ii / self.b
            d = (desired_positions - self.positions)[1:-1]

            n_minus = np.diff(self.positions)[:-1]
            n_plus = np.diff(self.positions)[1:]

            d_mask = ((n_plus > 1) * (d >= 1)) + ((n_minus > 1) * (d <= -1))

            if np.any(d_mask):
                q = self.heights[1:-1]
                q_minus = np.diff(self.heights)[:-1]
                q_plus = np.diff(self.heights)[1:]

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
                    (quad_q > self.heights[:-2]) * (self.heights[2:] > quad_q) * d_mask
                )
                self.heights[[False, *q_mask, False]] = quad_q[q_mask]

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
                    self.heights[[False, *l_mask, False]] = lin_q[l_mask]
                self.positions[[False, *d_mask, False]] += d[d_mask]
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
        result[positive] = q[negative] + (q_plus[negative]) / (n_plus[negative])
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
    percentiles = np.percentile(data, [0, 25, 50, 75, 100])
    marker_heights = np.array([0.02, 0.5, 4.44, 17.22, 38.62])

    p2_perc = P2Histogram(b=4)

    for x in data:
        p2_perc.add(x)

    np.testing.assert_allclose(
        p2_perc.heights, np.array([0.02, 0.506, 3.370, 14.61, 38.62]), rtol=1e-3
    )

    # Extra 100 random data points
    new_data = np.random.random((100)) * 40
    for x in new_data:
        p2_perc.add(x)
    data = np.concat([data, new_data])
    percentiles = np.percentile(data, [0, 25, 50, 75, 100])

    error = 1 - np.dot(
        percentiles / np.linalg.norm(percentiles),
        p2_perc.heights / np.linalg.norm(p2_perc.heights),
    )
    r2 = np.sqrt(np.average(np.pow(percentiles - p2_perc.heights, 2)))
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
        p2_perc.heights / np.linalg.norm(p2_perc.heights),
    )
    r2 = np.sqrt(np.average(np.pow(percentiles - p2_perc.heights, 2)))
    assert r2 < 0.17
    assert error < 1e-4
