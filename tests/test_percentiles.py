# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
import sys

import numpy as np

from ms_nexus_tools import lib as nxlib


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

    p2_perc = nxlib.normalisation.P2Histogram(b=4)

    for ii, x in enumerate(data):
        p2_perc.add(x)

    ic(p2_perc.heights)
    ic(percentiles)
    ic(expected_results)

    ic(p2_perc.positions)
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

    p2_perc = nxlib.normalisation.P2Histogram(
        b=4,
        shape=(2,),
    )

    for ii, x in enumerate(data):
        p2_perc.add(x)
        print(f" --- {ii + 1}: {x} --- ", file=sys.stderr)
        ic(p2_perc.positions + 1)
        ic(p2_perc.heights)

    ic(p2_perc.heights)
    ic(p2_perc.positions)

    np.testing.assert_allclose(p2_perc.heights, expected_results, rtol=1e-3)


def test_percentiles_3d():

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
    data = np.tile(data, (2, 2, 1)).T
    expected_results = np.tile(expected_results, (2, 2, 1))
    ic(data.shape)
    ic(expected_results.shape)
    percentiles = np.percentile(data, [0, 25, 50, 75, 100], axis=0).T
    ic(percentiles.shape)
    ic(percentiles)

    p2_perc = nxlib.normalisation.P2Histogram(
        b=4,
        shape=(
            2,
            2,
        ),
    )

    for ii, x in enumerate(data):
        p2_perc.add(x)

    ic(p2_perc.heights)
    ic(p2_perc.positions)

    np.testing.assert_allclose(p2_perc.heights, expected_results, rtol=1e-3)
