"""
Regression test for C++ pybind vis_ffa_transform_basep memory ownership
and native period array return.

Verifies:
1. Returned trial periods array owns its memory buffer and does not suffer
   from use-after-free corruption following Python garbage collection.
2. Trial periods match the analytical FFA shift formula to within machine
   precision (1e-12).
3. Monotonicity: periods strictly increase across shift rows starting at
   P(0) = tsamp * bins.
4. Downsampling support (both downsample_factor == 1.0 and > 1.0).
5. Robustness under repeated memory stress and garbage collection cycles.
"""

import gc
import numpy as np
import pytest

from riptide import libcpp
from riptide.vis_search import vis_ffa_search_basep


def analytical_ffa_periods(nsamp, bins, tsamp, rows_eval=0, downsample_factor=1.0):
    """
    Compute expected FFA trial periods from the analytical shift formula:
    P(s) = (tau * bins^2) / (bins - s / (rows - 1.0))
    """
    n = int(nsamp / downsample_factor) if downsample_factor > 1.0 else nsamp
    rows = n // bins
    if rows < 2:
        raise ValueError(f"Too few rows ({rows}) for bins={bins}.")
    actual_rows_eval = rows_eval if (0 < rows_eval < rows) else rows
    s_shifts = np.arange(actual_rows_eval, dtype=np.float64)
    periods = (tsamp * bins * bins) / (bins - s_shifts / (rows - 1.0))
    return periods, actual_rows_eval


def generate_synthetic_visibilities(n_uv=4, nsamp=2048, seed=42):
    """
    Initialise reproducible complex visibility time series for testing.
    """
    rng = np.random.default_rng(seed)
    real_part = rng.standard_normal((n_uv, nsamp), dtype=np.float32)
    imag_part = rng.standard_normal((n_uv, nsamp), dtype=np.float32)
    return (real_part + 1j * imag_part).astype(np.complex64)


@pytest.mark.parametrize("downsample_factor", [1.0, 2.0, 4.0])
@pytest.mark.parametrize("rows_eval", [0, 8])
def test_vis_ffa_transform_basep_analytical_precision(downsample_factor, rows_eval):
    """
    Verify returned periods match analytical FFA formulation within 1e-12.
    """
    n_uv = 3
    nsamp = 4096
    bins = 32
    tsamp = 0.110

    vis_data = generate_synthetic_visibilities(n_uv=n_uv, nsamp=nsamp, seed=123)
    expected_periods, expected_rows = analytical_ffa_periods(
        nsamp=nsamp,
        bins=bins,
        tsamp=tsamp,
        rows_eval=rows_eval,
        downsample_factor=downsample_factor,
    )

    out_cube, periods = libcpp.vis_ffa_transform_basep(
        vis_data,
        bins,
        tsamp,
        rows_eval,
        downsample_factor,
    )

    # Validate output cube geometry and data type
    assert out_cube.shape == (n_uv, expected_rows, bins)
    assert out_cube.dtype == np.complex64

    # Validate periods geometry, data type, and boundary behaviour
    assert periods.shape == (expected_rows,)
    assert periods.dtype == np.float64
    assert np.isclose(periods[0], tsamp * bins, rtol=1e-12, atol=1e-12)

    # Strictly monotonic increase across shift rows
    if expected_rows > 1:
        diffs = np.diff(periods)
        assert np.all(diffs > 0), "Trial periods must strictly increase across shift rows"

    # Analytical agreement within machine precision
    np.testing.assert_allclose(periods, expected_periods, rtol=1e-12, atol=1e-12)


def _call_and_isolate_periods(vis_data, bins, tsamp, rows_eval, downsample_factor):
    """
    Helper function to invoke transform and let output cube fall out of scope.
    """
    out_cube, periods = libcpp.vis_ffa_transform_basep(
        vis_data,
        bins,
        tsamp,
        rows_eval,
        downsample_factor,
    )
    # Explicitly delete output cube and return only the periods array
    del out_cube
    return periods


def test_vis_ffa_periods_memory_ownership_under_gc():
    """
    Verify periods array retains valid data across multiple aggressive GC cycles
    and heap allocations, guarding against dangling pointers and use-after-free bugs.
    """
    n_uv = 2
    nsamp = 2048
    bins = 16
    tsamp = 0.05
    downsample_factor = 1.0

    vis_data = generate_synthetic_visibilities(n_uv=n_uv, nsamp=nsamp, seed=999)
    expected_periods, _ = analytical_ffa_periods(
        nsamp=nsamp,
        bins=bins,
        tsamp=tsamp,
        rows_eval=0,
        downsample_factor=downsample_factor,
    )

    # Obtain periods from isolated call where stack buffers and out_cube are discarded
    periods = _call_and_isolate_periods(
        vis_data,
        bins,
        tsamp,
        0,
        downsample_factor,
    )

    # Aggressive memory churn to overwrite freed heap memory if periods is dangling
    for _ in range(50):
        junk = [np.ones((512, 512), dtype=np.float64) * 3.14159 for _ in range(5)]
        del junk
        gc.collect()

    # The periods array must still be intact and match expected analytical values
    assert periods.dtype == np.float64
    assert np.isclose(periods[0], tsamp * bins, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(periods, expected_periods, rtol=1e-12, atol=1e-12)


def test_vis_ffa_search_basep_integration():
    """
    End-to-end integration test verifying that vis_ffa_search_basep unpacks
    and provides the native periods array without Python recalculation.
    """
    n_uv = 4
    nsamp = 2048
    bins = 32
    tsamp = 0.110
    downsample_factor = 2.0

    vis_data = generate_synthetic_visibilities(n_uv=n_uv, nsamp=nsamp, seed=777)
    uv_inds = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)

    expected_periods, expected_rows = analytical_ffa_periods(
        nsamp=nsamp,
        bins=bins,
        tsamp=tsamp,
        rows_eval=0,
        downsample_factor=downsample_factor,
    )

    res = vis_ffa_search_basep(
        vis_data,
        bins,
        uv_inds,
        tsamp,
        rows_eval=0,
        downsample_factor=downsample_factor,
    )

    assert len(res.periods) == expected_rows
    assert res.periods.dtype == np.float64
    assert np.all(np.diff(res.periods) > 0)
    np.testing.assert_allclose(res.periods, expected_periods, rtol=1e-12, atol=1e-12)


def test_vis_ffa_transform_basep_error_handling():
    """
    Verify invalid input dimensions or parameters fail loudly with exceptions.
    """
    tsamp = 0.110
    bins = 16

    # 1D array must be rejected (must be 2D N_uv, nsamp)
    bad_vis_1d = np.zeros(1024, dtype=np.complex64)
    with pytest.raises(RuntimeError):
        libcpp.vis_ffa_transform_basep(bad_vis_1d, bins, tsamp)

    # Insufficient samples for specified bins (rows < 2)
    insufficient_samples = np.zeros((2, 10), dtype=np.complex64)
    with pytest.raises(RuntimeError):
        libcpp.vis_ffa_transform_basep(insufficient_samples, bins, tsamp)

    # Invalid downsampling factor (< 1.0)
    vis_data = np.zeros((2, 512), dtype=np.complex64)
    with pytest.raises((ValueError, RuntimeError)):
        libcpp.vis_ffa_transform_basep(vis_data, bins, tsamp, 0, 0.5)
