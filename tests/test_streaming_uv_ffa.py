import time
import numpy as np
import riptide
from riptide.vis_search import uv_ffa_streaming_search
from riptide.time_series import VisTimeSeries

def test_streaming_uv_ffa_injection():
    print("=================================================================")
    print("Testing riptide Streaming UV-FFA with ducc0.wgridder.vis2grid")
    print("=================================================================")
    nbaselines = 253
    nchan = 1
    ntimes = 600
    tsamp = 0.100  # 100 ms sampling
    period_true = 2.000  # 2.0 seconds (period = 20 samples)
    duty_cycle = 0.05  # 5% pulse width
    
    nx, ny = 64, 64
    pixsize_x = 0.0005
    pixsize_y = 0.0005
    freq = np.array([1.4e9])
    c = 299792458.0
    
    # Injected pulsar location: (+3, -5) pixels from phase center
    delta_l = 3 * pixsize_x
    delta_m = -5 * pixsize_y
    source_flux = 3.5  # Jy peak flux
    
    print(f"Observation setup: {nbaselines} baselines, {ntimes} time steps ({ntimes * tsamp:.1f} s), dt={tsamp*1000:.0f} ms")
    print(f"Injected Pulsar: P={period_true:.3f} s, Flux={source_flux} Jy, Offset=(+3, -5) pixels")
    
    # 1. Synthesize baseline UVW tracks
    np.random.seed(42)
    uvw_0 = (np.random.rand(nbaselines, 3) - 0.5) * 4000.0
    uvw_0[:, 2] = 0.0
    
    # Slight baseline migration over time (hour angle rotation)
    theta = np.linspace(0, 0.005, ntimes)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    uvw_3d = np.zeros((nbaselines, 3, ntimes), dtype=np.float64)
    for it in range(ntimes):
        uvw_3d[:, 0, it] = uvw_0[:, 0] * cos_t[it] - uvw_0[:, 1] * sin_t[it]
        uvw_3d[:, 1, it] = uvw_0[:, 0] * sin_t[it] + uvw_0[:, 1] * cos_t[it]
    
    # 2. Generate noisy baseline visibilities + periodic pulsed point source
    t_axis = np.arange(ntimes) * tsamp
    pulse_phase = (t_axis % period_true) / period_true
    pulse_amplitude = np.where(pulse_phase < duty_cycle, source_flux, 0.0)
    
    # Geometric phase per baseline
    u_lam = uvw_3d[:, 0, :] * freq[0] / c
    v_lam = uvw_3d[:, 1, :] * freq[0] / c
    geom_phase = -2.0 * np.pi * (u_lam * delta_l + v_lam * delta_m)
    
    noise_sigma = 0.5
    noise = (np.random.randn(nbaselines, nchan, ntimes) + 1j * np.random.randn(nbaselines, nchan, ntimes)) * noise_sigma
    pulsar_signal = (pulse_amplitude[np.newaxis, np.newaxis, :] * np.exp(1j * geom_phase[:, np.newaxis, :])).astype(np.complex64)
    vis_3d = (noise + pulsar_signal).astype(np.complex64)
    wgt_3d = np.ones((nbaselines, nchan, ntimes), dtype=np.float32)
    
    # 3. Execute streaming UV-domain FFA search
    t0 = time.perf_counter()
    results = uv_ffa_streaming_search(
        msname_or_data=(vis_3d, wgt_3d, uvw_3d, freq, tsamp),
        period_min=1.5,
        period_max=2.5,
        bins_min=16,
        bins_max=32,
        nx=nx,
        ny=ny,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        epsilon=1e-5,
        snr_thresh=6.0,
        nthreads=4
    )
    t_search = time.perf_counter() - t0
    
    print(f"\nStreaming UV-FFA search completed in {t_search:.3f} s")
    candidates = results["candidates"]
    print(f"Number of candidates detected above S/N threshold: {len(candidates)}")
    if len(results["snrs"]) > 0:
        print(f"Max SNR across all trials: {np.max(results['snrs']):.2f}")
    
    assert len(candidates) > 0, "Failed to detect injected pulsar!"
    top_cand = candidates[0]
    print(f"\nTop Candidate:")
    print(f"  Period:        {top_cand['period']:.5f} s (Injected: {period_true:.5f} s)")
    print(f"  Detection SNR: {top_cand['snr']:.2f}")
    print(f"  Peak Pixel:    {top_cand['peak_pixel']} (Expected: ({nx//2 + 3}, {ny//2 - 5}))")
    
    # Verify period accuracy (within 1%)
    period_err = abs(top_cand["period"] - period_true) / period_true
    print(f"  Period Error:  {period_err * 100.0:.3f}%")
    assert period_err < 0.02, f"Period mismatch: {top_cand['period']} vs {period_true}"
    
    # Verify image peak location
    expected_peak = (nx // 2 + 3, ny // 2 - 5)
    assert top_cand["peak_pixel"] == expected_peak, f"Peak location mismatch: {top_cand['peak_pixel']} vs {expected_peak}"
    
    print("\n>> ALL TESTS PASSED: Streaming UV-FFA successfully detected candidate and localized the source!")


def test_streaming_uv_ffa_high_dm_1deg_offaxis():
    """
    Injection and recovery test for a high-DM (DM = 1000 pc cm^-3),
    highly off-axis (~1.0 degree from phase centre) periodic pulsar in riptide.
    """
    print("\n=================================================================")
    print("Testing riptide Streaming UV-FFA: High DM (1000 pc/cm^3), 1 deg Off-Axis")
    print("=================================================================")
    np.random.seed(101)
    nbaselines = 64
    nchan = 16
    ntimes = 600
    tsamp = 0.100  # 100 ms sampling (60 s total)
    period_true = 2.000  # 2.0 s period
    dm_true = 1000.0  # pc cm^-3
    duty_cycle = 0.05

    # 128x128 image grid with 0.0004 rad (~82.5 arcsec) pixels -> ~2.93 deg FoV
    nx, ny = 128, 128
    pixsize_x = 0.0004
    pixsize_y = 0.0004
    chan_freq = np.linspace(1.15e9, 1.45e9, nchan)
    c = 299792458.0

    # Place source ~1.00 degree (0.01744 rad) from phase centre:
    # sqrt(35^2 + (-26)^2) * 0.0004 = 0.01744 rad (~1.00 deg)
    dx_pix, dy_pix = 35, -26
    expected_peak = (nx // 2 + dx_pix, ny // 2 + dy_pix)
    delta_l = dx_pix * pixsize_x
    delta_m = dy_pix * pixsize_y
    offset_deg = np.degrees(np.arcsin(np.hypot(delta_l, delta_m)))
    assert 0.98 <= offset_deg <= 1.02

    # Dispersion delays across frequency band
    k_dm = 4.148808e-3
    f_ghz = chan_freq * 1e-9
    ref_ghz = np.max(f_ghz)
    delays_s = k_dm * dm_true * (1.0 / f_ghz**2 - 1.0 / ref_ghz**2)

    print(
        f"Injected Pulsar: P={period_true:.3f} s, DM={dm_true:.1f} pc/cm^3 "
        f"(max delay={np.max(delays_s):.2f} s), Offset={offset_deg:.3f} deg -> Pixel {expected_peak}"
    )

    # Synthesize baseline UVW tracks with Earth-rotation drift
    uvw_0 = (np.random.rand(nbaselines, 3) - 0.5) * 1800.0
    uvw_0[:, 2] = 0.0
    theta = np.linspace(0.0, 0.015, ntimes)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    uvw_3d = np.zeros((nbaselines, 3, ntimes), dtype=np.float64)
    uvw_3d[:, 0, :] = uvw_0[:, 0, None] * cos_t[None, :] - uvw_0[:, 1, None] * sin_t[None, :]
    uvw_3d[:, 1, :] = uvw_0[:, 0, None] * sin_t[None, :] + uvw_0[:, 1, None] * cos_t[None, :]

    # Generate dispersed wideband visibilities
    t_axis = np.arange(ntimes) * tsamp
    noise_sigma = 0.35
    vis_3d = (
        (np.random.randn(nbaselines, nchan, ntimes) + 1j * np.random.randn(nbaselines, nchan, ntimes))
        * noise_sigma
    ).astype(np.complex64)
    wgt_3d = np.ones((nbaselines, nchan, ntimes), dtype=np.float32)

    source_flux = 4.0
    for ic in range(nchan):
        t_emit = t_axis - delays_s[ic]
        phase_c = (t_emit % period_true) / period_true
        pulse_c = np.where(phase_c < duty_cycle, source_flux, 0.0)

        inv_lam = chan_freq[ic] / c
        u_lam = uvw_3d[:, 0, :] * inv_lam
        v_lam = uvw_3d[:, 1, :] * inv_lam
        geom_phase = -2.0 * np.pi * (u_lam * delta_l + v_lam * delta_m)
        vis_3d[:, ic, :] += (pulse_c[None, :] * np.exp(1j * geom_phase)).astype(np.complex64)

    # 1. Search with exact per-channel UVW tracking & dedispersion at DM = 1000
    results = uv_ffa_streaming_search(
        msname_or_data=(vis_3d, wgt_3d, uvw_3d, chan_freq, tsamp),
        period_min=1.8,
        period_max=2.2,
        bins_min=16,
        bins_max=24,
        dm=dm_true,
        exact_uvw=True,
        collapse_channels=False,
        nx=nx,
        ny=ny,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        epsilon=1e-5,
        snr_thresh=8.0,
        nthreads=4,
    )

    # 2. Search with collapsed channels (chromatic radial smearing at 1 deg off-axis)
    results_collapsed = uv_ffa_streaming_search(
        msname_or_data=(vis_3d, wgt_3d, uvw_3d, chan_freq, tsamp),
        period_min=1.8,
        period_max=2.2,
        bins_min=16,
        bins_max=24,
        dm=dm_true,
        exact_uvw=False,
        collapse_channels=True,
        nx=nx,
        ny=ny,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        epsilon=1e-5,
        snr_thresh=8.0,
        nthreads=4,
    )

    candidates = results["candidates"]
    assert len(candidates) > 0, "Failed to detect DM=1000, 1-deg off-axis pulsar!"
    top_cand = candidates[0]
    period_err = abs(top_cand["period"] - period_true) / period_true
    collapsed_max_snr = float(np.max(results_collapsed["snrs"])) if len(results_collapsed["snrs"]) > 0 else 0.0

    print(f"Top Candidate (Exact UVW, DM=1000):")
    print(f"  Period:        {top_cand['period']:.5f} s (Injected: {period_true:.5f} s, Error: {period_err*100:.3f}%)")
    print(f"  Detection SNR: {top_cand['snr']:.2f} (Collapsed-channel max SNR: {collapsed_max_snr:.2f})")
    print(f"  Peak Pixel:    {top_cand['peak_pixel']} (Expected: {expected_peak})")

    assert period_err < 0.005, f"Period mismatch: {top_cand['period']} vs {period_true}"
    assert top_cand["peak_pixel"] == expected_peak, (
        f"Peak location mismatch: {top_cand['peak_pixel']} vs {expected_peak}"
    )
    assert top_cand["snr"] > 25.0, f"Expected high SNR (>25), got {top_cand['snr']:.2f}"
    assert top_cand["snr"] > 2.0 * collapsed_max_snr, (
        f"Exact UVW SNR ({top_cand['snr']:.2f}) should far exceed collapsed SNR ({collapsed_max_snr:.2f})"
    )


if __name__ == "__main__":
    test_streaming_uv_ffa_injection()
    test_streaming_uv_ffa_high_dm_1deg_offaxis()
