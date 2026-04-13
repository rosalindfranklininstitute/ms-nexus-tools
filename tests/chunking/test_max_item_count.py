import math

from hypothesis import given, strategies as st

import numpy as np

from ms_nexus_tools.lib.chunking import Chunker
import ms_nexus_tools.lib.chunking as chunking

from icecream import ic


@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=1000),
)
def test_priority_bounds(x, y, z, count):
    chunker = Chunker.from_max_item_count(
        data_shape=(x, y, z), priorities=(1, 2, 3), items_per_chunk=count
    )

    # Test priorities
    if chunker.data_shape[0] >= chunker.data_shape[1]:
        assert chunker.chunk_shape[0] >= chunker.chunk_shape[1]
    if chunker.data_shape[1] >= chunker.data_shape[2]:
        assert chunker.chunk_shape[1] >= chunker.chunk_shape[2]

    # Test effect of priorities
    if count < chunker.data_shape[0]:
        assert chunker.chunk_shape[0] == count
        assert chunker.chunk_shape[1] == 1
        assert chunker.chunk_shape[2] == 1
    elif count < chunker.data_shape[0] * chunker.data_shape[1]:
        assert chunker.chunk_shape[0] == chunker.data_shape[0]
        assert chunker.chunk_shape[0] * chunker.chunk_shape[1] <= count
        assert chunker.chunk_shape[2] == 1


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
    chunker = Chunker.from_max_item_count(
        data_shape=(x, y, z), priorities=(a, b, c), items_per_chunk=count
    )

    if np.prod(chunker.data_shape) >= count:
        assert np.prod(chunker.chunk_shape) <= count

    for i in range(3):
        # Basic requirement: chunks cover data
        assert chunker.chunk_shape[i] * chunker.chunk_count[i] >= chunker.data_shape[i]

        # Optimality cirteria: chunks are not superfulously large
        assert (
            chunker.chunk_shape[i] * (chunker.chunk_count[i] - 1)
            < chunker.data_shape[i]
        )

        for j in range(i + 1, 3):
            if chunker.priorities[i] < chunker.priorities[j]:
                if chunker.data_shape[i] >= chunker.data_shape[j]:
                    if count > chunker.data_shape[j]:
                        assert chunker.chunk_shape[i] >= chunker.chunk_shape[j]
            elif chunker.priorities[i] > chunker.priorities[j]:
                if chunker.data_shape[i] <= chunker.data_shape[j]:
                    if count > chunker.data_shape[i]:
                        assert chunker.chunk_shape[i] <= chunker.chunk_shape[j]


@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=1000000),
)
def test_2_shared_priorities(n, count):

    chunker = Chunker.from_max_item_count(
        data_shape=(n, n, n), priorities=(1, 1, 2), items_per_chunk=count
    )

    assert chunker.chunk_shape[0] == chunker.chunk_shape[1]
    assert chunker.chunk_count[0] == chunker.chunk_count[1]
    assert chunker.chunk_shape[0] <= min(math.sqrt(count), chunker.data_shape[0])

    if count < n * n:
        chunker.chunk_shape[2] == n
        chunker.chunk_count[2] == 1
    else:
        chunker.chunk_shape[2] >= count / (n * n)
