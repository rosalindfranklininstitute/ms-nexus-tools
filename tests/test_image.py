# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

from hypothesis import given

from ms_nexus_tools import lib as nxlib

import numpy as np


def get_corner(data, corner: nxlib.OriginLocation):
    match corner:
        case nxlib.OriginLocation.UPPER_LEFT:
            return data[0, 0]
        case nxlib.OriginLocation.UPPER_RIGHT:
            return data[0, -1]
        case nxlib.OriginLocation.LOWER_LEFT:
            return data[-1, 0]
        case nxlib.OriginLocation.LOWER_RIGHT:
            return data[-1, -1]


@given(...)
def test_adjust_origin(new: nxlib.OriginLocation, current: nxlib.OriginLocation):

    data = np.array([[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]])

    adjusted = nxlib.image.adjust_origin(data, new, current)

    assert get_corner(adjusted, current) == get_corner(data, new)
    assert len(data.shape) == len(adjusted.shape)
    assert len(data) == len(adjusted)


@given(...)
def test_adjust_origin_default(new: nxlib.OriginLocation):

    data = np.array([[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]])

    adjusted = nxlib.image.adjust_origin(data, new)

    assert get_corner(adjusted, nxlib.OriginLocation.UPPER_LEFT) == get_corner(
        data, new
    )
    assert len(data.shape) == len(adjusted.shape)
    assert len(data) == len(adjusted)
