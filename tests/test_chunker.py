import math

from hypothesis import given, strategies as st

import numpy as np

from ms_nexus_tools.lib.chunking import Chunker
import ms_nexus_tools.lib.chunking as chunking

from icecream import ic


def test_itmes_overflow():

    chunker = Chunker.from_item_count(
        data_shape=(100, 100, 100), priorities=(1, 2, 3), min_items_per_chunk=10
    )

    assert chunker.chunk_shape == (10, 1, 1)
    assert chunker.chunk_count == (10, 100, 100)

    chunker = Chunker.from_item_count(
        data_shape=(100, 100, 100), priorities=(1, 2, 3), min_items_per_chunk=200
    )

    assert chunker.chunk_shape == (100, 2, 1)
    assert chunker.chunk_count == (1, 50, 100)

    chunker = Chunker.from_item_count(
        data_shape=(100, 100, 100), priorities=(3, 2, 1), min_items_per_chunk=10
    )

    assert chunker.chunk_shape == (1, 1, 10)
    assert chunker.chunk_count == (100, 100, 10)

    chunker = Chunker.from_item_count(
        data_shape=(100, 100, 100), priorities=(3, 2, 1), min_items_per_chunk=200
    )

    assert chunker.chunk_shape == (1, 2, 100)
    assert chunker.chunk_count == (100, 50, 1)


def test_itmes_matched_priorities():

    chunker = Chunker.from_item_count(
        data_shape=(100, 100, 100), priorities=(1, 1, 2), min_items_per_chunk=10
    )

    assert chunker.chunk_shape == (4, 4, 1)
    assert chunker.chunk_count == (25, 25, 100)

    chunker = Chunker.from_item_count(
        data_shape=(100, 100, 100), priorities=(1, 1, 2), min_items_per_chunk=200
    )

    assert chunker.chunk_shape == (15, 15, 1)
    assert chunker.chunk_count == (7, 7, 100)

    chunker = Chunker.from_item_count(
        data_shape=(100, 100, 100), priorities=(1, 1, 1), min_items_per_chunk=10
    )

    assert chunker.chunk_shape == (3, 3, 3)
    assert chunker.chunk_count == (34, 34, 34)

    chunker = Chunker.from_item_count(
        data_shape=(100, 100, 100), priorities=(1, 1, 1), min_items_per_chunk=200
    )

    assert chunker.chunk_shape == (6, 6, 6)
    assert chunker.chunk_count == (17, 17, 17)


def test_count_overflow():

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(1, 2, 3), min_chunk_count=10
    )

    assert chunker.chunk_shape == (10, 100, 100)
    assert chunker.chunk_count == (10, 1, 1)

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(1, 2, 3), min_chunk_count=200
    )

    assert chunker.chunk_shape == (1, 50, 100)
    assert chunker.chunk_count == (100, 2, 1)

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(3, 2, 1), min_chunk_count=10
    )

    assert chunker.chunk_shape == (100, 100, 10)
    assert chunker.chunk_count == (1, 1, 10)

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(3, 2, 1), min_chunk_count=200
    )

    assert chunker.chunk_shape == (100, 50, 1)
    assert chunker.chunk_count == (1, 2, 100)


def test_count_matched_priorities():

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(1, 1, 2), min_chunk_count=10
    )

    assert chunker.chunk_shape == (25, 25, 100)
    assert chunker.chunk_count == (4, 4, 1)

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(1, 1, 2), min_chunk_count=200
    )
    ic(chunker)

    assert chunker.chunk_shape == (7, 7, 100)
    assert chunker.chunk_count == (15, 15, 1)

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(1, 1, 1), min_chunk_count=10
    )

    assert chunker.chunk_shape == (34, 34, 34)
    assert chunker.chunk_count == (3, 3, 3)

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(1, 1, 1), min_chunk_count=200
    )

    assert chunker.chunk_shape == (17, 17, 17)
    assert chunker.chunk_count == (6, 6, 6)


def test_chunks():

    chunker = Chunker.from_item_count(
        data_shape=(10, 10, 10), priorities=(1, 1, 1), min_items_per_chunk=10
    )

    assert chunker.chunk_shape == (3, 3, 3)
    assert chunker.chunk_count == (4, 4, 4)

    chunks = [chunk for chunk in chunker.chunks()]
    assert len(chunks) == np.prod(chunker.chunk_count)

    flags = np.zeros(chunker.data_shape)
    assert np.sum(flags) == 0
    for chunk in chunks:
        flags[*chunk] = 1
    assert np.sum(flags) == np.prod(chunker.data_shape)

    chunker = Chunker.from_min_chunks(
        data_shape=(10, 10, 10), priorities=(1, 1, 1), min_chunk_count=10
    )

    assert chunker.chunk_shape == (4, 4, 4)
    assert chunker.chunk_count == (3, 3, 3)

    chunks = [chunk for chunk in chunker.chunks()]
    assert len(chunks) == np.prod(chunker.chunk_count)

    flags = np.zeros(chunker.data_shape)
    assert np.sum(flags) == 0
    for chunk in chunks:
        flags[*chunk] = 1
    assert np.sum(flags) == np.prod(chunker.data_shape)


def test_spectial():
    """
    Test cases that came up from the memory chunking tests.
    Isolated here to simplify debugging.
    """

    chunker = Chunker.from_min_chunks(
        data_shape=(71, 85, 85, 2212), priorities=(1, 2, 2, 3), min_chunk_count=80315
    )

    assert len([c for c in chunker.chunks()]) > 0
    assert chunker.chunk_shape == (1, 2, 2, 2212)
    assert chunker.chunk_count == (71, 43, 43, 1)

    chunker = Chunker.from_min_chunks(
        data_shape=(1, 2, 42, 2085), priorities=(1, 2, 2, 3), min_chunk_count=3
    )

    assert len([c for c in chunker.chunks()]) > 0
    assert chunker.chunk_shape == (1, 1, 21, 2085)
    assert chunker.chunk_count == (1, 2, 2, 1)

    chunker = Chunker.from_min_chunks(
        data_shape=(14, 31, 62, 9999), priorities=(1, 3, 3, 2), min_chunk_count=1003
    )

    assert len([c for c in chunker.chunks()]) > 0
    assert chunker.chunk_shape == (1, 31, 62, 139)
    assert chunker.chunk_count == (14, 1, 1, 72)


@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=1000),
)
def test_items_bounds(x, y, z, count):
    chunker = Chunker.from_item_count(
        data_shape=(x, y, z), priorities=(1, 2, 3), min_items_per_chunk=count
    )

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
        data_shape=(x, y, z), priorities=(a, b, c), min_chunk_count=count
    )

    if np.prod(chunker.data_shape) >= count:
        assert np.prod(chunker.chunk_count) >= count

    for i in range(3):
        # Basic requirement
        assert chunker.chunk_shape[i] * chunker.chunk_count[i] >= chunker.data_shape[i]
        # All chunks withing bounds
        assert (
            chunker.chunk_shape[i] * (chunker.chunk_count[i] - 1)
            < chunker.data_shape[i]
        )

    chunker = Chunker.from_item_count(
        data_shape=(x, y, z), priorities=(a, b, c), min_items_per_chunk=count
    )

    if np.prod(chunker.data_shape) >= count:
        assert np.prod(chunker.chunk_shape) >= count

    for i in range(3):
        # Basic requirement
        assert chunker.chunk_shape[i] * chunker.chunk_count[i] >= chunker.data_shape[i]
        # All chunks withing bounds
        assert (
            chunker.chunk_shape[i] * (chunker.chunk_count[i] - 1)
            < chunker.data_shape[i]
        )


@given(
    st.integers(min_value=1, max_value=1000000),
)
def test_items_2_shared_priorities(count):

    chunker = Chunker.from_item_count(
        data_shape=(100, 100, 100), priorities=(1, 1, 2), min_items_per_chunk=count
    )

    assert chunker.chunk_shape[0] == chunker.chunk_shape[1]
    assert chunker.chunk_count[0] == chunker.chunk_count[1]
    assert chunker.chunk_shape[0] >= min(math.sqrt(count), chunker.data_shape[0])

    if count < 10_000:  # 100*100
        chunker.chunk_shape[2] == 100
    else:
        chunker.chunk_shape[2] >= count / (
            chunker.data_shape[0] * chunker.data_shape[1]
        )


@given(
    st.integers(min_value=1, max_value=1000000),
)
def test_counts_2_shared_priorities(count):

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(1, 1, 2), min_chunk_count=count
    )

    assert chunker.chunk_shape[0] == chunker.chunk_shape[1]
    assert chunker.chunk_count[0] == chunker.chunk_count[1]
    assert chunker.chunk_count[0] >= min(math.sqrt(count), chunker.data_shape[0])

    if count < 10_000:  # 100*100
        chunker.chunk_shape[2] == 100
    else:
        chunker.chunk_shape[2] >= count / (
            chunker.data_shape[0] * chunker.data_shape[1]
        )
