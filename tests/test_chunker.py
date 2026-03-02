import math

from hypothesis import given, strategies as st

import numpy as np

from ms_nexus_tools.lib.chunking import Chunker

from icecream import ic


def test_overflow():

    chunker = Chunker(data_shape=(100, 100, 100), priorities=(1, 2, 3), count=10)

    assert chunker.chunk_shape == (10, 1, 1)
    assert chunker.chunk_count == (10, 100, 100)

    chunker = Chunker(data_shape=(100, 100, 100), priorities=(1, 2, 3), count=200)

    assert chunker.chunk_shape == (100, 2, 1)
    assert chunker.chunk_count == (1, 50, 100)

    chunker = Chunker(data_shape=(100, 100, 100), priorities=(3, 2, 1), count=10)

    assert chunker.chunk_shape == (1, 1, 10)
    assert chunker.chunk_count == (100, 100, 10)

    chunker = Chunker(data_shape=(100, 100, 100), priorities=(3, 2, 1), count=200)

    assert chunker.chunk_shape == (1, 2, 100)
    assert chunker.chunk_count == (100, 50, 1)


def test_matched_priorities():

    chunker = Chunker(data_shape=(100, 100, 100), priorities=(1, 1, 2), count=10)

    assert chunker.chunk_shape == (4, 4, 1)
    assert chunker.chunk_count == (25, 25, 100)

    chunker = Chunker(data_shape=(100, 100, 100), priorities=(1, 1, 2), count=200)

    assert chunker.chunk_shape == (15, 15, 1)
    assert chunker.chunk_count == (7, 7, 100)

    chunker = Chunker(data_shape=(100, 100, 100), priorities=(1, 1, 1), count=10)

    assert chunker.chunk_shape == (3, 3, 3)
    assert chunker.chunk_count == (34, 34, 34)

    chunker = Chunker(data_shape=(100, 100, 100), priorities=(1, 1, 1), count=200)

    assert chunker.chunk_shape == (6, 6, 6)
    assert chunker.chunk_count == (17, 17, 17)


@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=1000),
)
def test_bounds(x, y, z, count):
    chunker = Chunker(data_shape=(x, y, z), priorities=(1, 2, 3), count=count)

    if np.prod(chunker.data_shape) >= count:
        assert np.prod(chunker.chunk_shape) >= count
        assert np.prod(chunker.chunk_shape) <= 2 * count
    if chunker.data_shape[0] >= chunker.data_shape[1]:
        assert chunker.chunk_shape[0] >= chunker.chunk_shape[1]
    if chunker.data_shape[1] >= chunker.data_shape[2]:
        assert chunker.chunk_shape[1] >= chunker.chunk_shape[2]

    for i in range(3):
        assert chunker.chunk_shape[i] * chunker.chunk_count[i] >= chunker.data_shape[i]

    if count < chunker.data_shape[0]:
        assert chunker.chunk_shape[1] == 1

    if count < chunker.data_shape[0] * chunker.data_shape[1]:
        assert chunker.chunk_shape[2] == 1


@given(
    st.integers(min_value=1, max_value=1000000),
)
def test_2_shared_priorities(count):

    chunker = Chunker(data_shape=(100, 100, 100), priorities=(1, 1, 2), count=count)

    assert chunker.chunk_shape[0] == chunker.chunk_shape[1]
    assert chunker.chunk_count[0] == chunker.chunk_count[1]
    assert chunker.chunk_shape[0] >= min(math.sqrt(count), chunker.data_shape[0])

    if count < 10_000:  # 100*100
        chunker.chunk_shape[2] == 100
    else:
        chunker.chunk_shape[2] >= count / (
            chunker.data_shape[0] * chunker.data_shape[1]
        )
