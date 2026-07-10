# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
from functools import reduce
from ms_nexus_tools.lib.data_source import Axis, AxisDensity

from pathlib import Path

import numpy as np
import h5py

from ms_nexus_tools.lib.chunker import count_chunks_to_cover
from ms_nexus_tools.api import data_convert

from . import man_source

import pytest


@pytest.fixture(scope="module")
def man_data():
    return man_source.ManData()


@pytest.fixture
def nx_file():
    filename = Path(__file__).parent / "test.nxs"
    if filename.exists():
        filename.unlink()
    yield filename
    filename.unlink()


def check_data_correct(fle, man_data, max_chunk_item_count):
    for data_name in ["images", "spectra"]:
        assert f"/entry/{data_name}/data/signal" in fle
        assert f"/entry/{data_name}/data/x" in fle
        assert f"/entry/{data_name}/data/y" in fle
        assert f"/entry/{data_name}/data/mz" in fle

        assert "x_indices" in fle[f"/entry/{data_name}/data/"].attrs
        assert "y_indices" in fle[f"/entry/{data_name}/data/"].attrs
        assert "mz_indices" in fle[f"/entry/{data_name}/data/"].attrs

        assert fle[f"/entry/{data_name}/data"].attrs["x_indices"] == 0
        assert fle[f"/entry/{data_name}/data"].attrs["y_indices"] == 1
        assert fle[f"/entry/{data_name}/data"].attrs["mz_indices"] == 2
        assert np.all(fle[f"/entry/{data_name}/data"].attrs["axes"] == ["x", "y", "mz"])

        assert fle[f"/entry/{data_name}/data/signal"].shape == man_data.shape

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
        chunk_max_byte_count=240 * 2,
        memory_max_byte_count=8 * 8 * 20 * 2,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})

    assert nx_file.exists()

    with h5py.File(nx_file, "r") as fle:
        check_data_correct(fle, man_data, process_args.chunk_max_byte_count / 2)


def get_dataset_total_and_used_chunks(fle, name):
    shape = fle[name].shape
    chunks = fle[name].chunks
    dsid = fle[name].id
    n = dsid.get_num_chunks()
    count = count_chunks_to_cover(shape, chunks)
    total_chunks = reduce(lambda x, y: x * y, count)
    return total_chunks, n


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

        for data_name in ["images", "spectra"]:
            assert f"/entry/{data_name}/data/mz_exact" in fle
            assert "mz_exact_indices" in fle[f"/entry/{data_name}/data/"].attrs

            name = f"/entry/{data_name}/data/signal"

            assert fle[f"/entry/{data_name}/data/mz_exact"].shape == fle[name].shape

            total_chunks, n = get_dataset_total_and_used_chunks(fle, name)

            assert f"/entry/{data_name}/data/mz_exact" in fle

            assert n == total_chunks
            assert n == 1

        assert "/entry/item_counts" in fle
        assert np.all(fle["/entry/item_counts/data/signal"][...] <= 1)

        assert "/entry/item_counts_total_spectra" in fle
        assert "/entry/item_counts_total_image" not in fle


def test_sparse_single_axis_multi_chunk(nx_file, man_data):
    man_data_source = man_source.ManSource(
        man_data,
        supplimentary_axes=[Axis("mz", 2, [0, 1], AxisDensity.SPARSE, np.int16, "mz")],
    )

    process_args = data_convert.ProcessArgs(
        in_path=Path(__file__).parent / "Man1.txt",
        out_path=nx_file,
        chunk_max_byte_count=240 * 2,
        memory_max_byte_count=8 * 8 * 20 * 4,
        data_source=man_data_source,
    )
    data_convert.process(process_args, {})

    assert nx_file.exists()

    with h5py.File(nx_file, "r") as fle:
        check_data_correct(fle, man_data, process_args.chunk_max_byte_count / 2)

        has_some_sparsity = False
        for data_name in ["images", "spectra"]:
            assert f"/entry/{data_name}/data/mz_exact" in fle
            assert "mz_exact_indices" in fle[f"/entry/{data_name}/data/"].attrs

            name = f"/entry/{data_name}/data/signal"

            assert fle[f"/entry/{data_name}/data/mz_exact"].shape == fle[name].shape

            total_chunks, n = get_dataset_total_and_used_chunks(fle, name)

            has_some_sparsity |= n < total_chunks
        assert has_some_sparsity

        assert "/entry/item_counts" in fle
        assert np.all(fle["/entry/item_counts/data/signal"][...] <= 1)

        assert "/entry/item_counts_total_spectra" in fle
        assert "/entry/item_counts_total_image" not in fle


def test_sparse_single_axis_single_chunk_with_mz_bin_2(nx_file, man_data):
    man_data_source = man_source.ManSource(
        man_data,
        supplimentary_axes=[Axis("mz", 2, [0, 1], AxisDensity.SPARSE, np.int16, "mz")],
        mz_binning=2,
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
        for data_name in ["images", "spectra"]:
            data_part = fle[f"/entry/{data_name}/data/signal"]
            mz_values = fle[f"/entry/{data_name}/data/mz"]
            assert np.min(mz_values[:]) == 2
            assert np.max(mz_values[:]) == 240
            assert mz_values.shape == (120,)
            assert np.all(np.sum(data_part, axis=2) == np.sum(man_data.dense, axis=2))
            for ii in range(4):
                assert np.all(
                    np.sum(data_part[:, :, 30 * ii : 30 * (ii + 1)], axis=2)
                    == np.sum(man_data.dense[:, :, 60 * ii : 60 * (ii + 1)], axis=2),
                )

        assert "/entry/item_counts" in fle
        count_data = fle["/entry/item_counts/data/signal"][...]
        correct_value = np.full(count_data.shape, False)
        correct_value[count_data >= 0] = True
        correct_value[count_data <= 2] = True
        assert np.all(correct_value)


def test_dense_multi_axis_single_chunk(nx_file, man_data):
    man_data_source = man_source.ManSource(
        man_data,
        supplimentary_axes=[
            Axis("time", 0, [], AxisDensity.CONTINUOUS, np.int16, "s"),
            Axis("error", 2, [], AxisDensity.CONTINUOUS, np.int16, ""),
        ],
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
        for data_name in ["images", "spectra"]:
            assert f"/entry/{data_name}/data/signal" in fle
            assert f"/entry/{data_name}/data/x" in fle
            assert f"/entry/{data_name}/data/y" in fle
            assert f"/entry/{data_name}/data/mz" in fle
            assert f"/entry/{data_name}/data/time" in fle
            assert f"/entry/{data_name}/data/error" in fle

            assert "x_indices" in fle[f"/entry/{data_name}/data/"].attrs
            assert "y_indices" in fle[f"/entry/{data_name}/data/"].attrs
            assert "mz_indices" in fle[f"/entry/{data_name}/data/"].attrs
            assert "time_indices" in fle[f"/entry/{data_name}/data/"].attrs
            assert "error_indices" in fle[f"/entry/{data_name}/data/"].attrs

            assert np.all(
                fle[f"/entry/{data_name}/data"].attrs["axes"] == ["x", "y", "mz"],
            )
            assert fle[f"/entry/{data_name}/data/"].attrs["x_indices"] == 0
            assert fle[f"/entry/{data_name}/data/"].attrs["time_indices"] == 0
            assert fle[f"/entry/{data_name}/data/"].attrs["y_indices"] == 1
            assert fle[f"/entry/{data_name}/data/"].attrs["mz_indices"] == 2
            assert fle[f"/entry/{data_name}/data/"].attrs["error_indices"] == 2


def test_sparse_multi_continuous_axis_single_chunk(nx_file, man_data):
    man_data_source = man_source.ManSource(
        man_data,
        supplimentary_axes=[
            Axis("mz", 2, [1, 2], AxisDensity.SPARSE, np.int16, "s"),
            Axis("time", 0, [], AxisDensity.CONTINUOUS, np.int16, "s"),
        ],
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
        for data_name in ["images", "spectra"]:
            assert f"/entry/{data_name}/data/signal" in fle
            assert f"/entry/{data_name}/data/x" in fle
            assert f"/entry/{data_name}/data/y" in fle
            assert f"/entry/{data_name}/data/mz" in fle
            assert f"/entry/{data_name}/data/time" in fle

            assert "x_indices" in fle[f"/entry/{data_name}/data/"].attrs
            assert "time_indices" in fle[f"/entry/{data_name}/data/"].attrs
            assert "y_indices" in fle[f"/entry/{data_name}/data/"].attrs
            assert "mz_indices" in fle[f"/entry/{data_name}/data/"].attrs

            assert np.all(
                fle[f"/entry/{data_name}/data"].attrs["axes"] == ["x", "y", "mz"],
            )
            assert fle[f"/entry/{data_name}/data/"].attrs["x_indices"] == 0
            assert fle[f"/entry/{data_name}/data/"].attrs["time_indices"] == 0
            assert fle[f"/entry/{data_name}/data/"].attrs["y_indices"] == 1
            assert fle[f"/entry/{data_name}/data/"].attrs["mz_indices"] == 2


@pytest.mark.skip(
    reason=""" 
    Currently only 1 axis is enforced, even though the API suggest 
    multiple are possible. 
    It is easy to see how nultiple sparse axis covering the same 
    dimension might work (like error and mz.) But it is not entierly 
    clear how it should work with multiple dimensions being sparse. 
    For example, given (x,y,mz), how do we handle y and mz being
    sparse? The exact values of y do need to be propogated over
    mz, only x and y. But, that would require a seperate coords
    for each primary dimension. 
    In addition what if there was an exis (like y above) but where 
    the secondary axis were not strictly those lower than its primary, 
    axis. 
    Does that even make sense? 
    Or are its secondary axis always the full set of axis less than 
    its prmary axis? 
    """,
)
def test_sparse_multi_sparse_axis_single_chunk():
    pass
