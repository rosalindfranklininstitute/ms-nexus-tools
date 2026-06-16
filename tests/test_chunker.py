# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0

import math

from hypothesis import given, strategies as st

import numpy as np

from ms_nexus_tools.lib.chunker import Chunker

from icecream import ic


def test_itmes_overflow():

    chunker = Chunker.from_max_item_count(
        data_shape=(100, 100, 100), priorities=(1, 2, 3), items_per_chunk=10
    )

    assert chunker.chunk_shape == (10, 1, 1)
    assert chunker.chunk_count == (10, 100, 100)

    chunker = Chunker.from_max_item_count(
        data_shape=(100, 100, 100), priorities=(1, 2, 3), items_per_chunk=200
    )

    assert chunker.chunk_shape == (100, 2, 1)
    assert chunker.chunk_count == (1, 50, 100)

    chunker = Chunker.from_max_item_count(
        data_shape=(100, 100, 100), priorities=(3, 2, 1), items_per_chunk=10
    )

    assert chunker.chunk_shape == (1, 1, 10)
    assert chunker.chunk_count == (100, 100, 10)

    chunker = Chunker.from_max_item_count(
        data_shape=(100, 100, 100), priorities=(3, 2, 1), items_per_chunk=200
    )

    assert chunker.chunk_shape == (1, 2, 100)
    assert chunker.chunk_count == (100, 50, 1)


def test_itmes_matched_priorities():

    chunker = Chunker.from_max_item_count(
        data_shape=(100, 100, 100), priorities=(1, 1, 2), items_per_chunk=10
    )

    assert chunker.chunk_shape == (3, 3, 1)
    assert chunker.chunk_count == (34, 34, 100)

    chunker = Chunker.from_max_item_count(
        data_shape=(100, 100, 100), priorities=(1, 1, 2), items_per_chunk=200
    )

    assert chunker.chunk_shape == (14, 14, 1)
    assert chunker.chunk_count == (8, 8, 100)

    chunker = Chunker.from_max_item_count(
        data_shape=(100, 100, 100), priorities=(1, 1, 1), items_per_chunk=10
    )

    assert chunker.chunk_shape == (2, 2, 2)
    assert chunker.chunk_count == (50, 50, 50)

    chunker = Chunker.from_max_item_count(
        data_shape=(100, 100, 100), priorities=(1, 1, 1), items_per_chunk=200
    )

    assert chunker.chunk_shape == (5, 5, 5)
    assert chunker.chunk_count == (20, 20, 20)


def test_itmes_min_chunk_count():

    chunker = Chunker.from_max_item_count(
        data_shape=(100, 100, 100),
        priorities=(1, 2, 3),
        items_per_chunk=10,
        min_chunk_count=(2, 2, 2),
    )

    assert chunker.chunk_shape == (10, 1, 1)
    assert chunker.chunk_count == (10, 100, 100)

    chunker = Chunker.from_max_item_count(
        data_shape=(100, 100, 100),
        priorities=(1, 2, 3),
        items_per_chunk=200,
        min_chunk_count=(2, 2, 2),
    )

    assert chunker.chunk_shape == (50, 4, 1)
    assert chunker.chunk_count == (2, 25, 100)

    chunker = Chunker.from_max_item_count(
        data_shape=(10, 10, 10),
        priorities=(1, 2, 3),
        items_per_chunk=1000,
        min_chunk_count=(2, 2, 2),
    )

    assert chunker.chunk_shape == (5, 5, 5)
    assert chunker.chunk_count == (2, 2, 2)

    chunker = Chunker.from_max_item_count(
        data_shape=(10, 10, 10),
        priorities=(1, 2, 3),
        items_per_chunk=5,
        min_chunk_count=(2, 2, 2),
    )

    assert chunker.chunk_shape == (5, 1, 1)
    assert chunker.chunk_count == (2, 10, 10)

    chunker = Chunker.from_max_item_count(
        data_shape=(10, 10, 10),
        priorities=(1, 2, 3),
        items_per_chunk=5,
        min_chunk_count=(11, 2, 2),
    )

    assert chunker.chunk_shape == (1, 5, 1)
    assert chunker.chunk_count == (10, 2, 10)


def test_count_overflow():

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(1, 2, 3), chunk_count=10
    )

    assert chunker.chunk_shape == (10, 100, 100)
    assert chunker.chunk_count == (10, 1, 1)

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(1, 2, 3), chunk_count=200
    )

    assert chunker.chunk_shape == (1, 50, 100)
    assert chunker.chunk_count == (100, 2, 1)

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(3, 2, 1), chunk_count=10
    )

    assert chunker.chunk_shape == (100, 100, 10)
    assert chunker.chunk_count == (1, 1, 10)

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(3, 2, 1), chunk_count=200
    )

    assert chunker.chunk_shape == (100, 50, 1)
    assert chunker.chunk_count == (1, 2, 100)


def test_count_matched_priorities():

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(1, 1, 2), chunk_count=10
    )

    assert chunker.chunk_shape == (25, 25, 100)
    assert chunker.chunk_count == (4, 4, 1)

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(1, 1, 2), chunk_count=200
    )

    assert chunker.chunk_shape == (7, 7, 100)
    assert chunker.chunk_count == (15, 15, 1)

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(1, 1, 1), chunk_count=10
    )

    assert chunker.chunk_shape == (34, 34, 34)
    assert chunker.chunk_count == (3, 3, 3)

    chunker = Chunker.from_min_chunks(
        data_shape=(100, 100, 100), priorities=(1, 1, 1), chunk_count=200
    )

    assert chunker.chunk_shape == (17, 17, 17)
    assert chunker.chunk_count == (6, 6, 6)


def test_chunks():

    chunker = Chunker.from_max_item_count(
        data_shape=(10, 10, 10), priorities=(1, 1, 1), items_per_chunk=10
    )

    assert chunker.chunk_shape == (2, 2, 2)
    assert chunker.chunk_count == (5, 5, 5)

    chunks = [chunk for chunk in chunker.chunks()]
    assert len(chunks) == np.prod(chunker.chunk_count)

    flags = np.zeros(chunker.data_shape)
    assert np.sum(flags) == 0
    for chunk in chunks:
        flags[*chunk] += 1
    assert np.sum(flags) == np.prod(chunker.data_shape)

    chunker = Chunker.from_min_chunks(
        data_shape=(10, 10, 10), priorities=(1, 1, 1), chunk_count=10
    )

    assert chunker.chunk_shape == (4, 4, 4)
    assert chunker.chunk_count == (3, 3, 3)

    chunks = [chunk for chunk in chunker.chunks()]
    assert len(chunks) == np.prod(chunker.chunk_count)

    flags = np.zeros(chunker.data_shape)
    assert np.sum(flags) == 0
    for chunk in chunks:
        flags[*chunk] += 1
    assert np.sum(flags) == np.prod(chunker.data_shape)


def test_normalise():

    chunker = Chunker.from_max_item_count(
        data_shape=(10, 10, 10), priorities=(1, 1, 2), items_per_chunk=49
    )

    assert chunker.chunk_shape == (7, 7, 1)
    assert chunker.chunk_count == (2, 2, 10)

    chunks = chunker.normalise()
    assert chunker.chunk_shape == (5, 5, 1)
    assert chunker.chunk_count == (2, 2, 10)


def test_bulk_chunks():

    chunker = Chunker.from_max_item_count(
        data_shape=(10, 10, 10), priorities=(1, 1, 1), items_per_chunk=10
    )

    assert chunker.chunk_shape == (2, 2, 2)
    assert chunker.chunk_count == (5, 5, 5)

    chunks = chunker.bulk_chunks()
    assert len(chunks) == 2 ** len(chunker.data_shape)

    flags = np.zeros(chunker.data_shape)
    assert np.sum(flags) == 0
    for chunk in chunks:
        flags[*chunk] += 1
    assert (flags == 1).all()
    assert np.sum(flags) == np.prod(chunker.data_shape)

    chunker = Chunker.from_max_item_count(
        data_shape=(1, 10, 10), priorities=(1, 1, 1), items_per_chunk=10
    )

    assert chunker.chunk_shape == (1, 3, 3)
    assert chunker.chunk_count == (1, 4, 4)

    chunks = chunker.bulk_chunks()
    assert len(chunks) == 2 ** (len(chunker.data_shape) - 1)

    flags = np.zeros(chunker.data_shape)
    assert np.sum(flags) == 0
    for chunk in chunks:
        flags[*chunk] += 1
    assert (flags == 1).all()
    assert np.sum(flags) == np.prod(chunker.data_shape)


def test_spectial():
    """
    Test cases that came up from the memory chunking tests.
    Isolated here to simplify debugging.
    """

    chunker = Chunker.from_min_chunks(
        data_shape=(71, 85, 85, 2212), priorities=(1, 2, 2, 3), chunk_count=80315
    )

    assert len([c for c in chunker.chunks()]) > 0
    assert chunker.chunk_shape == (1, 2, 2, 2212)
    assert chunker.chunk_count == (71, 43, 43, 1)

    chunker = Chunker.from_min_chunks(
        data_shape=(1, 2, 42, 2085), priorities=(1, 2, 2, 3), chunk_count=3
    )

    assert len([c for c in chunker.chunks()]) > 0
    assert chunker.chunk_shape == (1, 1, 21, 2085)
    assert chunker.chunk_count == (1, 2, 2, 1)

    chunker = Chunker.from_min_chunks(
        data_shape=(14, 31, 62, 9999), priorities=(1, 3, 3, 2), chunk_count=1003
    )

    assert len([c for c in chunker.chunks()]) > 0
    assert chunker.chunk_shape == (1, 31, 62, 139)
    assert chunker.chunk_count == (14, 1, 1, 72)
