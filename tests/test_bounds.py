from hypothesis import strategies as st, given
from pytest import raises

from ms_nexus_tools.lib.bounds import ContainedBounds, Chunk


@st.composite
def oute_inner_offset(draw):
    n1 = draw(st.integers(min_value=0, max_value=100))
    n2 = draw(st.integers(min_value=0, max_value=n1))
    n3 = draw(st.integers(min_value=0, max_value=n1 - n2))
    return (n1, n2, n3)


@st.composite
def bounds(draw):
    w = draw(oute_inner_offset())  # type: ignore
    h = draw(oute_inner_offset())  # type: ignore
    d = draw(oute_inner_offset())  # type: ignore
    return ContainedBounds(
        outer_shape=(w[0], h[0], d[0]),
        inner_shape=(w[1], h[1], d[1]),
        offset=(w[2], h[2], d[2]),
    )


@given(
    bounds(),  # type: ignore
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=0, max_value=100),
)
def test_indexes(bounds, iw, ih, id):

    inner_slices = bounds.inner_slices()

    if (
        inner_slices[0].start <= iw < inner_slices[0].stop
        and inner_slices[1].start <= ih < inner_slices[1].stop
        and inner_slices[2].start <= id < inner_slices[2].stop
    ):
        inner_inx = bounds.inner_index(iw, ih, id)
        outer_inx = bounds.outer_index(*inner_inx)

        assert outer_inx == [iw, ih, id]
    else:
        with raises(IndexError):
            bounds.inner_index(iw, ih, id)


@st.composite
def st_slice(draw):
    start = draw(st.integers(min_value=0, max_value=100))
    stop = draw(st.integers(min_value=start, max_value=100))
    return slice(start, stop)


@given(
    bounds(),  # type: ignore
    st_slice(),  # type: ignore
    st_slice(),  # type: ignore
    st_slice(),  # type: ignore
)
def test_slices(bounds, w, h, d):

    inner_slices = bounds.inner_slices()

    if (
        inner_slices[0].start <= w.start < inner_slices[0].stop
        and w.stop < inner_slices[0].stop
        and inner_slices[1].start <= h.start < inner_slices[1].stop
        and h.stop < inner_slices[1].stop
        and inner_slices[2].start <= d.start < inner_slices[2].stop
        and d.stop < inner_slices[2].stop
    ):
        inner_chunk = bounds.inner_chunk(Chunk((w, h, d)))
        outer_chunk = bounds.outer_chunk(inner_chunk)

        assert outer_chunk == Chunk([w, h, d])
    else:
        with raises(IndexError):
            bounds.inner_chunk(Chunk((w, h, d)))
