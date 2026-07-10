import numpy as np
from hypothesis import given, strategies as st

from ms_nexus_tools import lib as nxlib


def test_iterage():

    data = np.random.random((10, 10, 10))

    ii = 0
    for blob in nxlib.utils.iterate(data):
        np.testing.assert_array_equal(blob, data)
        ii += 1
    assert ii == 1


@given(
    st.integers(min_value=-3, max_value=2),
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=10),
)
def test_iterage_1d(axis, a, b, c):

    data = np.random.random((a, b, c))

    for ii, blob in enumerate(nxlib.utils.iterate(data, axis=axis)):
        match axis:
            case 0 | -3:
                expected_blob = data[ii, :, :]
            case 1 | -2:
                expected_blob = data[:, ii, :]
            case 2 | -1:
                expected_blob = data[:, :, ii]
            case _:
                raise ValueError

        np.testing.assert_array_equal(blob, expected_blob)


@given(
    st.integers(min_value=-2, max_value=2),
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=10),
)
def test_indices_1d(axis, a, b, c):

    data = np.zeros((a, b, c))

    for slc in nxlib.utils.indices(data.shape, axis=axis):
        data[*slc] = 1

    np.testing.assert_array_equal(data, np.ones((a, b, c)))


@given(
    st.integers(min_value=0, max_value=2),
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=10),
)
def test_indices_2d(not_axis, a, b, c):

    data = np.zeros((a, b, c))

    axis = tuple(ii for ii in range(3) if ii != not_axis)
    for slc in nxlib.utils.indices(data.shape, axis=axis):
        data[*slc] = 1

    np.testing.assert_array_equal(data, np.ones((a, b, c)))
