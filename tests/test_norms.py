import random

import pytest
import numpy as np

from ms_nexus_tools import lib as nxlib

from icecream import ic, install

install()


def test_norms_shape():

    ones = np.ones((2, 3, 4))

    norms = dict(
        x=nxlib.IncrementalNorm(axis=0),
        y=nxlib.IncrementalNorm(axis=1),
        z=nxlib.IncrementalNorm(axis=2),
        xy=nxlib.IncrementalNorm(axis=(0, 1)),
        yz=nxlib.IncrementalNorm(axis=(1, 2)),
        xyz=nxlib.IncrementalNorm(axis=(0, 1, 2)),
    )

    for n in norms.values():
        n.add(ones)
        n.add(ones * 2)

    assert norms["x"].min.shape == (3, 4)
    assert norms["y"].min.shape == (2, 4)
    assert norms["z"].min.shape == (2, 3)
    assert norms["xy"].min.shape == (4,)
    assert norms["yz"].min.shape == (2,)
    assert norms["xyz"].min.shape == ()

    for n in norms.values():
        assert n.min is not None
        if n.min.shape == ():
            ic(n)
            assert n.min == 1.0
        else:
            assert (n.min == np.ones(n.min.shape)).all()
        assert n.max is not None
        if n.max.shape == ():
            assert n.max == 2.0
        else:
            assert (n.max == np.ones(n.max.shape) * 2).all()

    assert (norms["x"].tic == 3 * 2).all()
    assert (norms["y"].tic == 3 * 3).all()
    assert (norms["z"].tic == 3 * 4).all()
    assert (norms["xy"].tic == 3 * 6).all()
    assert (norms["yz"].tic == 3 * 12).all()
    assert (norms["xyz"].tic == 3 * 24).all()


def test_norms_values():
    random_values = np.array([random.randrange(1000) for _ in range(1000)])

    norm = nxlib.IncrementalNorm(axis=None)

    for r in random_values:
        norm.add(r)

    assert norm.min == np.min(random_values)
    assert norm.max == np.max(random_values)
    assert norm.tic == np.sum(random_values)
