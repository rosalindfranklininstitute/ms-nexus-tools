# SPDX-FileCopyrightText: 2026 Duncan McDougall <duncan.mcdougall@rfi.ac.uk>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import h5py

from ms_nexus_tools.api import data_convert, imzml
from ms_nexus_tools.lib.data_source import Axis, AxisDensity

from pyimzml.ImzMLParser import ImzMLParser

from .data import man_source

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


@pytest.fixture
def imzml_files():
    filename = Path(__file__).parent / "test.imzML"
    dbname = Path(__file__).parent / "test.ibd"
    if filename.exists():
        filename.unlink()
    if dbname.exists():
        dbname.unlink()
    yield filename, dbname
    if filename.exists():
        filename.unlink()
    if dbname.exists():
        dbname.unlink()


def test_full_convert(nx_file, imzml_files, man_data):

    man_data_source = man_source.ManSource(man_data)

    convert_args = data_convert.ProcessArgs(
        in_path=Path(__file__).parent / "data" / "Man1.txt",
        out_path=nx_file,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(convert_args, {})
    assert nx_file.exists()

    imzml_args = imzml.ProcessArgs(
        in_path=nx_file,
        out_path=imzml_files[0],
        entry_name="images",
        signal="signal",
        mass="mz",
        x_axis=0,
        y_axis=1,
        z_axis=-1,
        mz_axis=2,
        one_indexed=False,
    )
    imzml.process(imzml_args, {})
    assert imzml_files[0].exists()
    assert imzml_files[1].exists()

    with ImzMLParser(filename=imzml_files[0]) as imzml_data:
        for ii, coords in enumerate(imzml_data.coordinates):
            mz_values, int_values = imzml_data.getspectrum(ii)
            assert np.all(man_data.dense[*coords[0:2], :] == int_values[:])


def test_swap_x_y(nx_file, imzml_files, man_data):

    man_data_source = man_source.ManSource(man_data)

    convert_args = data_convert.ProcessArgs(
        in_path=Path(__file__).parent / "data" / "Man1.txt",
        out_path=nx_file,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(convert_args, {})
    assert nx_file.exists()

    imzml_args = imzml.ProcessArgs(
        in_path=nx_file,
        out_path=imzml_files[0],
        entry_name="images",
        signal="signal",
        mass="mz",
        x_axis=1,
        y_axis=0,
        z_axis=-1,
        mz_axis=2,
        one_indexed=False,
    )
    imzml.process(imzml_args, {})
    assert imzml_files[0].exists()
    assert imzml_files[1].exists()

    with ImzMLParser(filename=imzml_files[0]) as imzml_data:
        for ii, coords in enumerate(imzml_data.coordinates):
            mz_values, int_values = imzml_data.getspectrum(ii)
            assert np.all(man_data.dense[coords[1], coords[0], :] == int_values[:])


def test_total_image(nx_file, imzml_files, man_data):

    man_data_source = man_source.ManSource(man_data)

    convert_args = data_convert.ProcessArgs(
        in_path=Path(__file__).parent / "data" / "Man1.txt",
        out_path=nx_file,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(convert_args, {})
    assert nx_file.exists()

    imzml_args = imzml.ProcessArgs(
        in_path=nx_file,
        out_path=imzml_files[0],
        entry_name="total_image",
        signal="signal",
        mass="y",
        x_axis=1,
        y_axis=-1,
        z_axis=0,
        mz_axis=2,
        one_indexed=False,
    )
    imzml.process(imzml_args, {})
    assert imzml_files[0].exists()
    assert imzml_files[1].exists()

    sum_image = np.sum(man_data.dense, axis=2)
    max_image = np.max(man_data.dense, axis=2)

    with ImzMLParser(filename=imzml_files[0]) as imzml_data:
        for ii, coords in enumerate(imzml_data.coordinates):
            mz_values, int_values = imzml_data.getspectrum(ii)
            if coords[2] == 1:
                assert np.all(sum_image[coords[0], :] == int_values[:])
            elif coords[2] == 0:
                assert np.all(max_image[coords[0], :] == int_values[:])


def test_binned_axis(nx_file, imzml_files, man_data):

    man_data_source = man_source.ManSource(
        man_data,
        supplimentary_axes=[Axis("mz", 2, AxisDensity.BINNED, np.int16, "mz")],
    )

    convert_args = data_convert.ProcessArgs(
        in_path=Path(__file__).parent / "data" / "Man1.txt",
        out_path=nx_file,
        chunk_max_byte_count=1024 * 1024,
        memory_max_byte_count=1024 * 1024 * 1024,
        data_source=man_data_source,
    )
    data_convert.process(convert_args, {})
    assert nx_file.exists()

    imzml_args = imzml.ProcessArgs(
        in_path=nx_file,
        out_path=imzml_files[0],
        entry_name="images",
        signal="signal",
        mass="mz_exact",
        x_axis=0,
        y_axis=1,
        z_axis=-1,
        mz_axis=2,
        one_indexed=False,
    )
    imzml.process(imzml_args, {})
    assert imzml_files[0].exists()
    assert imzml_files[1].exists()

    with ImzMLParser(filename=imzml_files[0]) as imzml_data:
        for ii, coords in enumerate(imzml_data.coordinates):
            mz_values, int_values = imzml_data.getspectrum(ii)
            assert np.all(man_data.dense[*coords[0:2], :] == int_values[:])
