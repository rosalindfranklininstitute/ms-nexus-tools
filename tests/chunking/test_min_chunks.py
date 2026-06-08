# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

import math

from hypothesis import given, strategies as st

import numpy as np

from ms_nexus_tools.lib.chunker import Chunker


@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=1000),
)
def test_priority_bounds(x, y, z, count):
    chunker = Chunker.from_min_chunks(
        data_shape=(x, y, z), priorities=(1, 2, 3), chunk_count=count
    )

    # Test priorities
    if chunker.data_shape[0] >= chunker.data_shape[1]:
        assert chunker.chunk_count[0] >= chunker.chunk_count[1]
    if chunker.data_shape[1] >= chunker.data_shape[2]:
        assert chunker.chunk_count[1] >= chunker.chunk_count[2]

    # Test effect of priorities
    if count <= chunker.data_shape[0]:
        assert chunker.chunk_count[0] >= count
        assert chunker.chunk_count[1] == 1
        assert chunker.chunk_count[2] == 1
    elif count < chunker.data_shape[0] + chunker.data_shape[1]:
        assert chunker.chunk_count[0] == chunker.data_shape[0]
        assert chunker.chunk_count[2] == 1
    elif count < np.sum(chunker.data_shape):
        assert chunker.chunk_count[0] == chunker.data_shape[0]


@given(
    st.integers(min_value=1, max_value=3),
    st.integers(min_value=1, max_value=3),
    st.integers(min_value=1, max_value=3),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=1000),
)
def test_bounds(a, b, c, x, y, z, count):
    chunker = Chunker.from_min_chunks(
        data_shape=(x, y, z), priorities=(a, b, c), chunk_count=count
    )

    if np.prod(chunker.data_shape) >= count:
        assert np.prod(chunker.chunk_count) >= count

    for i in range(3):
        # Basic requirement: chunks cover data
        assert chunker.chunk_shape[i] * chunker.chunk_count[i] >= chunker.data_shape[i]

        # Optimality cirteria: chunks are not superfulously large
        assert (
            chunker.chunk_shape[i] * (chunker.chunk_count[i] - 1)
            < chunker.data_shape[i]
        )


@given(
    st.integers(min_value=1, max_value=1000000),
)
def test_2_shared_priorities(count):

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(1, 1, 2), chunk_count=count
    )

    assert chunker.chunk_shape[0] == chunker.chunk_shape[1]
    assert chunker.chunk_count[0] == chunker.chunk_count[1]
    assert chunker.chunk_shape[0] >= max(chunker.data_shape[0] // count, 1)

    if count < 10_000:  # 100*100
        chunker.chunk_shape[2] == 100
    else:
        chunker.chunk_shape[2] >= count / (
            10_000  # 100*100
        )
