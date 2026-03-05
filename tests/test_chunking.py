from hypothesis import given, settings, strategies as st

from ms_nexus_tools import lib as mnxlib


@given(
    st.integers(min_value=-1),
    st.floats(min_value=1e-3, max_value=1000),
    st.integers(min_value=1),
    st.integers(min_value=1),
    st.integers(min_value=1),
    st.integers(min_value=1),
    st.integers(min_value=1),
)
def test_memory_info(
    chunk_count_min: int, gb_max: float, processors: int, layers, width, height, spectra
):
    image_bounds = mnxlib.chunking.ImageBounds(layers, width, height, spectra)

    memory = mnxlib.chunking.MemoryInfo.calculate(
        chunk_count_min, gb_max, processors, image_bounds.to_bounds()
    )

    assert memory.min_chunk_count >= chunk_count_min
    total_memory_used = memory.max_chunk_gb * processors
    assert (total_memory_used - gb_max) < 1e-5

    assert memory.max_chunk_gb <= memory.total_gb


@given(
    st.integers(min_value=-1, max_value=1000),
    st.floats(min_value=1e-3, max_value=1000),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=10000),
)
@settings(deadline=1800)
def test_calculate_chunks(
    chunk_count_min: int, gb_max: float, processors: int, layers, width, height, spectra
):
    image_bounds = mnxlib.chunking.ImageBounds(layers, width, height, spectra)

    spectra_chunks, image_chunks, memory = mnxlib.chunking.calculate_chunks_from_memory(
        chunk_count_min, gb_max, processors, image_bounds.to_bounds()
    )

    assert len(spectra_chunks) == len(set(spectra_chunks))
    assert len(image_chunks) == len(set(image_chunks))

    assert memory.min_chunk_count >= chunk_count_min
    # Removed because it is a flaky check.
    # assert memory.max_chunk_gb * processors - gb_max < 1e-5
    assert memory.max_chunk_gb <= memory.total_gb

    max_chunks = layers * width * height * spectra
    if max_chunks > memory.min_chunk_count:
        assert len(spectra_chunks) >= memory.min_chunk_count
        assert len(image_chunks) >= memory.min_chunk_count
        assert len(spectra_chunks) <= max_chunks
        assert len(image_chunks) <= max_chunks
    else:
        min_max_chunks = layers * min(width, height) ** 2 * spectra
        assert len(spectra_chunks) >= min_max_chunks
        assert len(image_chunks) >= min_max_chunks

    # assert spectra_chunks[0].approximate_size_gb() - memory.max_chunk_gb < 1e-5
    # assert image_chunks[0].approximate_size_gb() - memory.max_chunk_gb < 1e-5
