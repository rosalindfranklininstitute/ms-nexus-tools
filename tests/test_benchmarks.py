import pytest

from dataclasses import dataclass
import json
import numpy as np
from pathlib import Path
import shutil
import datetime as dt

import multiprocessing as mp

from nexusformat.nexus import NXfield, nxload
from nexusformat.nexus.tree import NXinstrument

import h5py as h5

from ms_nexus_tools import api as nxapi, lib as nxlib

from icecream import ic

subprocess = True


def create_files(dir: str, layers: int, width: int, height: int, spectrum: int):

    path = Path("tmp_test_data") / Path(dir)
    hdf_path: Path = path.joinpath("hdf.h5")
    vds_path: Path = path.joinpath("vds.h5")
    nxs_path: Path = path.joinpath("nxs.h5")

    if not path.exists():
        path.mkdir()
    else:
        identical = True
        _, bounds, axis = nxapi.ion.read_metadata(hdf_path)
        identical &= bounds.layer_count == layers
        identical &= bounds.layer_width == width
        identical &= bounds.layer_height == height
        identical &= bounds.spectrum_length == spectrum

        if identical:
            with h5.File(vds_path, "r") as vds:
                identical &= "spectra" in vds
                if identical:
                    identical &= vds["spectra"].shape == bounds.shape
                identical &= "images" in vds
                if identical:
                    identical &= vds["images"].shape == bounds.shape

        if identical:
            data_root = nxload(nxs_path)

            identical &= data_root["entry"]["data"]["signal"].shape == bounds.shape
            raw_data = data_root["entry"]["data"]["signal"]

        if not identical:
            shutil.rmtree(path)
            path.mkdir()
        else:
            return dir, raw_data.shape, path, hdf_path, vds_path, nxs_path

    with nxlib.Timer(dir):
        bounds = nxapi.ion.IONImageBounds(
            layer_count=layers,
            layer_width=width,
            layer_height=height,
            spectrum_length=spectrum,
        )
        x_microns = 2
        y_microns = 2

        raw_data = np.zeros(shape=bounds.shape)

        for ll in range(bounds.layer_count):
            for ww in range(bounds.layer_width):
                for hh in range(bounds.layer_height):
                    raw_data[ll, ww, hh] = [
                        (ll * width * height * spectrum)
                        + (ww * width * height * spectrum)
                        + (hh * spectrum)
                        + ss
                        for ss in range(bounds.spectrum_length)
                    ]

        image_axis = nxapi.ion.ImageAxis(
            layer_axis=NXfield(np.arange(1, bounds.layer_count + 1, 1.0), name="layer"),
            x_axis=NXfield(
                np.arange(0, bounds.layer_width, 1.0) * x_microns,
                name="x",
                unit="micron",
            ),
            y_axis=NXfield(
                np.arange(0, bounds.layer_height, 1.0) * y_microns,
                name="y",
                unit="micron",
            ),
            mass_axis=NXfield(
                [ss**2 for ss in range(bounds.spectrum_length)], name="mass", unit="m/z"
            ),
        )

        nxapi.ion.write_metadata(hdf_path, NXinstrument(), bounds, image_axis)
        nxapi.ion.write_spectrum(hdf_path, bounds, raw_data, append=True)
        nxapi.ion.write_image(hdf_path, bounds, raw_data, append=True)

        nxapi.ion.create_spectra_vds(hdf_path, vds_path, bounds, append=False)
        nxapi.ion.create_image_vds(hdf_path, vds_path, bounds, append=True)

        nxlib.nxs.write_nxs(
            nxs_path, raw_data, x_microns, y_microns, np.array(image_axis.mass_axis)
        )

        return dir, raw_data.shape, path, hdf_path, vds_path, nxs_path


@pytest.fixture(scope="module")
def create_small_files():
    return create_files("small", 1, 50, 50, 1_000)


@pytest.fixture(scope="module")
def create_medium_files():
    return create_files("med", 5, 50, 50, 10_000)


@pytest.fixture(scope="module")
def create_large_files():
    return create_files("large", 10, 50, 50, 100_000)


def generate_parameterization():
    side_sizes = [1, 5, 25]
    file_fixture_names = [
        "create_small_files",
        "create_medium_files",
        "create_large_files",
    ]
    # file_fixture_names = ["create_small_files", "create_medium_files"]
    # file_fixture_names = ["create_small_files"]
    functions = [("spec", run_spectrum_image, 1, 1), ("mass", run_mass_image, 10, 1)]
    return "test_files, n, side_inc, name, function", [
        (fixture_name, n, side * mult, name, func)
        for fixture_name in file_fixture_names
        for side in side_sizes
        for name, func, mult, n in functions
    ]


def run_spectrum_image(
    n: int,
    side_inc: int,
    shape: tuple[int, int, int, int],
    in_path: Path,
    out_path: Path,
    filetype: nxlib.filetypes.DataType,
    iterations,
):
    count = 0
    for _ in range(n):
        for ii in range(10):
            if (ii + 1) * side_inc > shape[1]:
                continue

            if (ii + 1) * side_inc > shape[2]:
                continue

            args = nxapi.spectrum_image.ProcessArgs(
                hdf_in_path=in_path,
                img_out_path=out_path,
                layer=0,
                start_width=ii * side_inc,
                end_width=(ii + 1) * side_inc,
                start_height=ii * side_inc,
                end_height=(ii + 1) * side_inc,
                filetype=filetype,
            )
            nxapi.spectrum_image.process(args)
            assert out_path.exists()
            count += 1
    iterations.value = count


def run_mass_image(
    n: int,
    side_inc: int,
    shape: tuple[int, int, int, int],
    in_path: Path,
    out_path: Path,
    filetype: nxlib.filetypes.DataType,
    iterations,
):
    count = 0
    for _ in range(n):
        for ss in range(10):
            if (ss + 1) * side_inc > shape[3]:
                continue

            args = nxapi.mass_image.ProcessArgs(
                hdf_in_path=in_path,
                img_out_path=out_path,
                layer=0,
                start=ss * side_inc,
                end=(ss + 1) * side_inc,
                filetype=filetype,
            )
            nxapi.mass_image.process(args)
            assert out_path.exists()
            count += 1
    iterations.value = count


@pytest.mark.parametrize(*generate_parameterization())
def test_slices(test_files, n, side_inc, name, function, request):
    dir, shape, path, hdf_path, vds_path, nxs_path = request.getfixturevalue(test_files)
    out_path = path.joinpath(f"{name}.csv")

    files_to_test = [
        (hdf_path, nxlib.filetypes.DataType.ION_H5),
    ]
    if dir == "small":
        files_to_test.append((vds_path, nxlib.filetypes.DataType.ION_VDS))
    files_to_test.append((nxs_path, nxlib.filetypes.DataType.NEXUS))

    for in_path, filetype in files_to_test:
        with nxlib.JSONTimer(
            path.joinpath("times.json"),
            (name, str(filetype), str(side_inc), "forward"),
        ) as tmr:
            count = mp.Value("d", 0)
            if subprocess:
                p = mp.Process(
                    target=function,
                    args=(n, side_inc, shape, in_path, out_path, filetype, count),
                )
                p.start()
                p.join()
            else:
                function(n, side_inc, shape, in_path, out_path, filetype, count)
            tmr.add_user_data(count=count.value)

    for in_path, filetype in reversed(files_to_test):
        with nxlib.JSONTimer(
            path.joinpath("times.json"),
            (name, str(filetype), str(side_inc), "backward"),
        ) as tmr:
            count = mp.Value("d", 0)
            if subprocess:
                p = mp.Process(
                    target=function,
                    args=(n, side_inc, shape, in_path, out_path, filetype, count),
                )
                p.start()
                p.join()
            else:
                function(n, side_inc, shape, in_path, out_path, filetype, count)
            tmr.add_user_data(count=count.value)
