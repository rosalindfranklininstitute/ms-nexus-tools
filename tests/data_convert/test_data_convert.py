# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
from ms_nexus_tools.lib.data_source import Axis, AxisDensity

import math
from pathlib import Path
import os

import numpy as np
import h5py

from ms_nexus_tools.lib.chunker import Chunker
from ms_nexus_tools.api import data_convert

from . import man_source

import pytest

from icecream import ic


@pytest.fixture(scope="module")
def man_data():
    return man_source.ManData()


@pytest.fixture(scope="function")
def nx_file():
    filename = Path(__file__).parent / "test.nxs"
    if filename.exists():
        os.remove(filename)
    yield filename
    pass


def check_data_correct(fle, man_data, max_chunk_item_count):
    for data_name in ["images", "spectra"]:
        assert f"/entry/{data_name}/data/signal" in fle
        assert f"/entry/{data_name}/data/x" in fle
        assert f"/entry/{data_name}/data/y" in fle
        assert f"/entry/{data_name}/data/mz" in fle

        assert fle[f"/entry/{data_name}/data/signal"].shape == man_data.shape
        if not np.all(
            fle[f"/entry/{data_name}/data/signal"][:, :, :] == man_data.dense
        ):
            a = ic(fle[f"/entry/{data_name}/data/signal"][2, 4, 0:20])
            b = ic(man_data.dense[2, 4, 0:20])
            ic(a - b)
            ic(np.sum(a), np.sum(b))

        assert np.all(fle[f"/entry/{data_name}/data/signal"][:, :, :] == man_data.dense)

        actual_item_count = np.prod(fle[f"/entry/{data_name}/data/signal"].chunks)
        assert actual_item_count <= max_chunk_item_count

    total_image = np.sum(man_data.dense, axis=2)
    assert fle["/entry/total_image/data/signal"].shape == (2, *total_image.shape)
    assert np.all(fle["/entry/total_image/data/signal"][1, :, :] == total_image)

    total_spectra = np.sum(man_data.dense, axis=(0, 1))
    assert fle["/entry/total_spectra/data/signal"].shape == (
        2,
        *total_spectra.shape,
    )
    assert np.all(fle["/entry/total_spectra/data/signal"][1, :] == total_spectra)


def test_dense_single_axis_single_chunk(nx_file, man_data):

    man_data_source = man_source.ManSource(man_data)

    process_args = data_convert.ProcessArgs(
        in_path=Path(__file__).parent / "Man1.txt",
        out_path=nx_file,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})

    assert nx_file.exists()

    with h5py.File(nx_file, "r") as fle:
        check_data_correct(fle, man_data, process_args.chunk_max_byte_count / 2)


def test_dense_single_axis_multi_chunk(nx_file, man_data):
    man_data_source = man_source.ManSource(man_data)

    process_args = data_convert.ProcessArgs(
        in_path=Path(__file__).parent / "Man1.txt",
        out_path=nx_file,
        chunk_max_byte_count=8 * 8 * 20 * 2,
        memory_max_byte_count=8 * 8 * 20 * 4,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})

    assert nx_file.exists()

    with h5py.File(nx_file, "r") as fle:
        check_data_correct(fle, man_data, process_args.chunk_max_byte_count / 2)


def test_sparse_single_axis_single_chunk(nx_file, man_data):
    man_data_source = man_source.ManSource(
        man_data,
        supplimentary_axes=[Axis("mz", 2, [0, 1], AxisDensity.SPARSE, np.int16, "mz")],
    )

    process_args = data_convert.ProcessArgs(
        in_path=Path(__file__).parent / "Man1.txt",
        out_path=nx_file,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})

    assert nx_file.exists()

    with h5py.File(nx_file, "r") as fle:
        check_data_correct(fle, man_data, process_args.chunk_max_byte_count / 2)


def test_sparse_single_axis_multi_chunk(nx_file, man_data):
    man_data_source = man_source.ManSource(
        man_data,
        supplimentary_axes=[Axis("mz", 2, [0, 1], AxisDensity.SPARSE, np.int16, "mz")],
    )

    process_args = data_convert.ProcessArgs(
        in_path=Path(__file__).parent / "Man1.txt",
        out_path=nx_file,
        chunk_max_byte_count=8 * 8 * 20 * 2,
        memory_max_byte_count=8 * 8 * 20 * 4,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})

    assert nx_file.exists()

    with h5py.File(nx_file, "r") as fle:
        check_data_correct(fle, man_data, process_args.chunk_max_byte_count / 2)


def test_dense_multi_axis_single_chunk():
    pass


def test_sparse_multi_continuous_axis_single_chunk():
    pass


def test_sparse_multi_sparse_axis_single_chunk():
    pass
