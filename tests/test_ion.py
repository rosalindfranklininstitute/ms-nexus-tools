import pytest

import numpy as np
from pathlib import Path
import shutil

from nexusformat.nexus import NXfield, nxload
from nexusformat.nexus.tree import NXinstrument

import h5py as h5

from ms_nexus_tools import api as nxapi, lib as nxlib

from icecream import ic


@pytest.fixture(scope="module")
def create_file():

    bounds = nxapi.ion.IONImageBounds(
        layer_count=5, layer_width=10, layer_height=10, spectrum_length=100
    )
    x_microns = 2
    y_microns = 2

    raw_data = np.zeros(shape=bounds.shape)

    for ll in range(bounds.layer_count):
        for ww in range(bounds.layer_width):
            for hh in range(bounds.layer_height):
                raw_data[ll, ww, hh] = [
                    ll * 100000 + ww * 1000 + hh * 100 + ss
                    for ss in range(bounds.spectrum_length)
                ]

    image_axis = nxapi.ion.ImageAxis(
        layer_axis=NXfield(np.arange(1, bounds.layer_count + 1, 1.0), name="layer"),
        x_axis=NXfield(
            np.arange(0, bounds.layer_width, 1.0) * x_microns, name="x", unit="micron"
        ),
        y_axis=NXfield(
            np.arange(0, bounds.layer_height, 1.0) * y_microns, name="y", unit="micron"
        ),
        mass_axis=NXfield(
            [ss**2 for ss in range(bounds.spectrum_length)], name="mass", unit="m/z"
        ),
    )

    path = Path("tmp_test_data")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    hdf_in_path: Path = path.joinpath("hdf_in.h5")

    nxapi.ion.write_metadata(hdf_in_path, NXinstrument(), bounds, image_axis)
    nxapi.ion.write_spectrum(hdf_in_path, bounds, raw_data, append=True)
    nxapi.ion.write_image(hdf_in_path, bounds, raw_data, append=True)

    return raw_data, hdf_in_path, path


def test_spectra_vds(create_file):
    raw_data, hdf_in_path, path = create_file
    _, bounds, axis = nxapi.ion.read_metadata(hdf_in_path)
    vds_path = path.joinpath("spectrum_vds.h5")
    nxapi.ion.create_spectra_vds(hdf_in_path, vds_path, bounds, append=False)

    with h5.File(vds_path, "r") as vds:
        assert np.equal(vds["spectra"][:, :, :, :], raw_data).all()


def test_images_vds(create_file):
    raw_data, hdf_in_path, path = create_file
    _, bounds, axis = nxapi.ion.read_metadata(hdf_in_path)
    vds_path = path.joinpath("spectrum_vds.h5")
    nxapi.ion.create_image_vds(hdf_in_path, vds_path, bounds, append=False)

    with h5.File(vds_path, "r") as vds:
        assert np.equal(vds["images"][:, :, :, :], raw_data).all()


def test_ion_nxs(create_file):
    raw_data, hdf_in_path, path = create_file
    _, bounds, axis = nxapi.ion.read_metadata(hdf_in_path)

    hdf_out_path = path.joinpath("rfi.nxs")
    ic(hdf_in_path, hdf_out_path)
    tmp_path = path.joinpath("tmp")
    args = nxapi.ion.ProcessArgs(
        hdf_in_path=hdf_in_path, hdf_out_path=hdf_out_path, tmp_data_path=tmp_path
    )
    nxapi.ion.process(args)

    data_root = nxload(hdf_out_path)

    assert np.equal(data_root["entry"]["spectra"]["data"]["signal"], raw_data).all()
    assert np.equal(data_root["entry"]["images"]["data"]["signal"], raw_data).all()
    assert np.equal(data_root["entry"]["data"]["signal"], raw_data).all()
