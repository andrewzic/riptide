import os
import time
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastducc.imaging import load_chunk_data
from fastducc.baseline_ffa import prepare_baseline_series
from riptide.vis_search import uv_ffa_streaming_search


def test_craco_option_c_dedisp_streaming():
    print("=========================================================================")
    print("VALIDATION: EXACT UVW TRACKING & DEDISPERSION STREAMING UV-FFA")
    print("=========================================================================")

    msname = os.path.join(
        os.path.dirname(__file__),
        "data",
        "cracoData.LTR_1733-2344.SB77974.beam01.20251015072402.calG9.uvsub.ms",
    )
    start_time_idx = 100
    end_time_idx = 700
    npix_x, npix_y = 384, 384
    pixsize_x = 22.0 / 206265.0
    pixsize_y = 22.0 / 206265.0
    epsilon = 1e-5
    nthreads = 4

    # 1. Load real CRACO data chunk
    t_load_0 = time.perf_counter()
    times, chan_freq, dt, vis_3d, wgt_3d, uvw_3d = load_chunk_data(
        msname,
        start_time_idx=start_time_idx,
        end_time_idx=end_time_idx,
        corr_mode="average",
        average_correlations=True
    )
    t_load = time.perf_counter() - t_load_0
    nbl, nchan, nt = vis_3d.shape
    total_obs_s = nt * dt
    print(f"Loaded CRACO MS: {nbl} baselines, {nchan} channels, {nt} integrations ({total_obs_s:.2f} s, dt={dt*1000:.1f} ms) in {t_load:.2f} s")
    print(f"Image Grid: {npix_x}x{npix_y} pixels, resolution: {pixsize_x*206265:.1f} arcsec/pixel")

    # 2. Inject dispersed periodic pulsar signal into 3D visibilities
    dm_inj = 75.0         # Injected DM in pc/cm^3
    p_inj = 2.00000       # Injected period in seconds
    duty_cycle = 0.05     # 5% duty cycle
    amp_inj = 0.40        # Peak flux in Jy
    dx_pix = +12          # Pixel offset from center in X
    dy_pix = -8           # Pixel offset from center in Y
    target_x = npix_x // 2 + dx_pix
    target_y = npix_y // 2 + dy_pix
    dl = dx_pix * pixsize_x
    dm = dy_pix * pixsize_y

    print(f"\nInjected Dispersed Pulsar Signal:")
    print(f"  Period:       {p_inj:.5f} s")
    print(f"  Dispersion:   DM = {dm_inj:.1f} pc/cm^3")
    print(f"  Peak Flux:    {amp_inj:.2f} Jy")
    print(f"  Pixel Offset: ({dx_pix:+d}, {dy_pix:+d}) -> Target Pixel: ({target_x}, {target_y})")

    # Compute physical dispersion delays across frequency channels
    K_DM = 4.148808e3 * 1e-6
    f_ghz = chan_freq * 1e-9
    ref_ghz = np.max(f_ghz)
    delays_s = K_DM * dm_inj * (1.0 / f_ghz**2 - 1.0 / ref_ghz**2)
    max_delay_s = np.max(delays_s)
    max_delay_samp = int(np.round(max_delay_s / dt))
    print(f"  Max Delay:    {max_delay_s*1000:.1f} ms ({max_delay_samp} time samples) across band [{np.min(f_ghz)*1e3:.1f} - {ref_ghz*1e3:.1f}] MHz")

    t_axis = np.arange(nt) * dt
    c_light = 299792458.0

    vis_injected = vis_3d.copy()
    for c in range(nchan):
        t_c = t_axis - delays_s[c]
        phase_c = (t_c % p_inj) / p_inj
        pulse_c = np.where(phase_c < duty_cycle, amp_inj, 0.0)

        lam_c = c_light / chan_freq[c]
        u_c = uvw_3d[:, 0, :] / lam_c
        v_c = uvw_3d[:, 1, :] / lam_c
        geom_phase_c = -2.0 * np.pi * (u_c * dl + v_c * dm)

        vis_injected[:, c, :] += (pulse_c[None, :] * np.exp(1j * geom_phase_c)).astype(np.complex64)

    # 3. Execute streaming UV-FFA search with Option C
    print("\n-------------------------------------------------------------------------")
    print("EXECUTING OPTION C STREAMING UV-FFA SEARCH")
    print("-------------------------------------------------------------------------")
    t_search_0 = time.perf_counter()
    results = uv_ffa_streaming_search(
        msname_or_data=(vis_injected, wgt_3d, uvw_3d, chan_freq, dt),
        period_min=1.8,
        period_max=2.2,
        bins_min=16,
        bins_max=24,
        dm=dm_inj,
        exact_uvw=True,
        collapse_channels=False,
        nx=npix_x,
        ny=npix_y,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        epsilon=epsilon,
        threshold=1e-4,
        nthreads=nthreads,
        snr_thresh=8.0,
        top_candidates=5
    )
    t_search_tot = time.perf_counter() - t_search_0

    vis_ts = results["vis_timeseries"]
    dga_shape = vis_ts.dga.shape
    unique_cells = vis_ts.unique_uv.shape[0]
    oversampled_grid = (vis_ts.nu, vis_ts.nv)

    dense_cube_gb = (vis_ts.nu * vis_ts.nv * nt * 8) / (1024 ** 3)
    dga_mb = (unique_cells * nt * 8) / (1024 ** 2)

    print("\n-------------------------------------------------------------------------")
    print("OPTION C PERFORMANCE & MEMORY METRICS")
    print("-------------------------------------------------------------------------")
    print(f"Pseudo-Baselines per Snapshot: {nbl * nchan} (253 bl x 288 chan)")
    print(f"Oversampled UV Grid:           {oversampled_grid[0]} x {oversampled_grid[1]}")
    print(f"Active Non-Zero UV Cells:      {unique_cells} ({unique_cells / (vis_ts.nu * vis_ts.nv) * 100:.2f}% occupancy)")
    print(f"Dense 3D Cube Memory:          {dense_cube_gb:.2f} GB (Eliminated entirely)")
    print(f"Dynamic Grid Array Memory:     {dga_mb:.1f} MB (In-memory storage)")
    print(f"Memory Reduction Factor:       {dense_cube_gb * 1024 / dga_mb:.1f}x")
    print(f"Total Search Runtime:          {t_search_tot:.3f} s ({t_search_tot / len(results['periods']):.2f} ms per period trial)")

    # 4. Candidate verification
    candidates = results["candidates"]
    print("\n-------------------------------------------------------------------------")
    print(f"DETECTION RESULTS: {len(candidates)} candidates above threshold")
    print("-------------------------------------------------------------------------")
    assert len(candidates) > 0, "Dispersed pulsar was not detected!"

    top = candidates[0]
    p_det = top["period"]
    snr_det = top["snr"]
    pix_det = top["peak_pixel"]
    p_err = abs(p_det - p_inj) / p_inj * 100.0

    print(f"Top Candidate:")
    print(f"  Detected Period:   {p_det:.5f} s (Injected: {p_inj:.5f} s, Error: {p_err:.3f}%)")
    print(f"  Detection SNR:      {snr_det:.1f}")
    print(f"  Detected Pixel:    {pix_det} (Expected: ({target_x}, {target_y}))")
    print(f"  Fold Bins:         {top['bins']}")

    # Assertions
    assert p_err < 0.25, f"Period error {p_err:.3f}% exceeds tolerance 0.25%"
    assert pix_det == (target_x, target_y), f"Pixel mismatch: {pix_det} vs ({target_x}, {target_y})"
    assert snr_det > 50.0, f"Detection SNR {snr_det:.1f} is unexpectedly low"

    # 5. Diagnostic Plotting (4-Panel View)
    plot_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "craco_option_c_dedisp_candidate.png")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # Panel 1: Full-field dirty image with non-obstructing open target ring
    im0 = axes[0].imshow(top["image"].T, origin="lower", cmap="inferno")
    target_circle = plt.Circle((target_x, target_y), radius=16, fill=False, edgecolor="cyan", lw=1.8, ls="--")
    axes[0].add_patch(target_circle)
    axes[0].text(target_x + 20, target_y + 20, f"({target_x}, {target_y})", color="cyan", fontsize=11, fontweight="bold")
    axes[0].set_title(f"Full Field ({npix_x}x{npix_y}, 22''/pix)\nP = {p_det:.4f} s, SNR = {snr_det:.1f}")
    axes[0].set_xlabel("X Pixel (RA)")
    axes[0].set_ylabel("Y Pixel (Dec)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Intensity (a.u.)")

    # Panel 2: High-resolution zoom cutout (40x40 pixels) centered on candidate
    cut_rad = 20
    x_min, x_max = target_x - cut_rad, target_x + cut_rad
    y_min, y_max = target_y - cut_rad, target_y + cut_rad
    cutout = top["image"].T[y_min:y_max, x_min:x_max]
    im1 = axes[1].imshow(cutout, origin="lower", cmap="inferno", extent=[x_min, x_max, y_min, y_max])
    axes[1].plot([x_min + 3, x_min + 8], [target_y, target_y], color="cyan", lw=1.5)
    axes[1].plot([x_max - 8, x_max - 3], [target_y, target_y], color="cyan", lw=1.5)
    axes[1].plot([target_x, target_x], [y_min + 3, y_min + 8], color="cyan", lw=1.5)
    axes[1].plot([target_x, target_x], [y_max - 8, y_max - 3], color="cyan", lw=1.5)
    axes[1].set_title(f"Option C Zoom Cutout\nExact UVW & DM={dm_inj} Coherence")
    axes[1].set_xlabel("X Pixel (RA)")
    axes[1].set_ylabel("Y Pixel (Dec)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Intensity (a.u.)")

    # Panel 3: Folded Pulse Profile at candidate pixel
    phase_axis = np.linspace(0, 1, top["bins"], endpoint=False)
    axes[2].step(phase_axis, top["profile"], where="mid", color="navy", lw=2)
    axes[2].set_title(f"Dedispersed Pulse Profile ({top['bins']} bins)\nPixel ({target_x}, {target_y})")
    axes[2].set_xlabel("Pulse Phase")
    axes[2].set_ylabel("Intensity")
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Periodogram around target period
    sort_idx = np.argsort(results["periods"])
    axes[3].plot(results["periods"][sort_idx], results["snrs"][sort_idx], color="firebrick", lw=1.5)
    axes[3].axvline(p_inj, color="black", linestyle="--", alpha=0.7, label=f"True P = {p_inj:.3f} s")
    axes[3].set_title(f"Option C Periodogram\nPeak SNR = {snr_det:.1f}")
    axes[3].set_xlabel("Trial Period (s)")
    axes[3].set_ylabel("Profile SNR (MAD)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"\nDiagnostic plot saved to: {plot_file}")
    print("\n>> OPTION C CRACO DEDISPERSION & STREAMING UV-FFA TEST PASSED!")


def test_craco_high_dm_1000_1deg_offaxis_streaming():
    """
    Real CRACO MS injection and recovery test for an extreme high-DM (DM = 1000 pc cm^-3),
    highly off-axis (~1.0 degree from phase centre) periodic pulsar.
    """
    print("\n=========================================================================")
    print("CRACO HIGH-DM (1000 pc/cm^3) & 1-DEG OFF-AXIS STREAMING UV-FFA VALIDATION")
    print("=========================================================================")

    msname = os.path.join(
        os.path.dirname(__file__),
        "data",
        "cracoData.LTR_1733-2344.SB77974.beam01.20251015072402.calG9.uvsub.ms",
    )
    start_time_idx = 100
    end_time_idx = 700
    npix_x, npix_y = 384, 384
    pixsize_x = 22.0 / 206265.0
    pixsize_y = 22.0 / 206265.0
    epsilon = 1e-5
    nthreads = 4

    times, chan_freq, dt, vis_3d, wgt_3d, uvw_3d = load_chunk_data(
        msname,
        start_time_idx=start_time_idx,
        end_time_idx=end_time_idx,
        corr_mode="average",
        average_correlations=True,
    )
    nbl, nchan, nt = vis_3d.shape

    # Inject dispersed periodic pulsar at DM = 1000 pc/cm^3 and ~1.00 degree from phase centre:
    # sqrt(130^2 + (-98)^2) * 22'' = 162.80 * 22'' = 3581.6'' = 0.995 deg (~1.00 deg)
    dm_inj = 1000.0
    p_inj = 2.00000
    duty_cycle = 0.05
    amp_inj = 0.45
    dx_pix = +130
    dy_pix = -98
    target_x = npix_x // 2 + dx_pix
    target_y = npix_y // 2 + dy_pix
    dl = dx_pix * pixsize_x
    dm_coord = dy_pix * pixsize_y
    offset_deg = math.degrees(math.asin(math.hypot(dl, dm_coord)))
    assert 0.98 <= offset_deg <= 1.02

    K_DM = 4.148808e3 * 1e-6
    f_ghz = chan_freq * 1e-9
    ref_ghz = np.max(f_ghz)
    delays_s = K_DM * dm_inj * (1.0 / f_ghz**2 - 1.0 / ref_ghz**2)
    max_delay_s = float(np.max(delays_s))
    max_delay_samp = int(np.round(max_delay_s / dt))

    print(
        f"Injected Pulsar: P={p_inj:.5f} s, DM={dm_inj:.1f} pc/cm^3 "
        f"(max delay={max_delay_s:.2f} s = {max_delay_samp} samples), "
        f"Offset={offset_deg:.3f} deg -> Target Pixel ({target_x}, {target_y})"
    )

    t_axis = np.arange(nt) * dt
    c_light = 299792458.0

    vis_injected = vis_3d.copy()
    for c in range(nchan):
        t_c = t_axis - delays_s[c]
        phase_c = (t_c % p_inj) / p_inj
        pulse_c = np.where(phase_c < duty_cycle, amp_inj, 0.0)

        lam_c = c_light / chan_freq[c]
        u_c = uvw_3d[:, 0, :] / lam_c
        v_c = uvw_3d[:, 1, :] / lam_c
        geom_phase_c = -2.0 * np.pi * (u_c * dl + v_c * dm_coord)

        vis_injected[:, c, :] += (pulse_c[None, :] * np.exp(1j * geom_phase_c)).astype(np.complex64)

    results = uv_ffa_streaming_search(
        msname_or_data=(vis_injected, wgt_3d, uvw_3d, chan_freq, dt),
        period_min=1.8,
        period_max=2.2,
        bins_min=16,
        bins_max=24,
        dm=dm_inj,
        exact_uvw=True,
        collapse_channels=False,
        nx=npix_x,
        ny=npix_y,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        epsilon=epsilon,
        threshold=1e-4,
        nthreads=nthreads,
        snr_thresh=8.0,
        top_candidates=5,
    )

    candidates = results["candidates"]
    assert len(candidates) > 0, "DM=1000, 1-deg off-axis pulsar was not detected in CRACO MS!"

    top = candidates[0]
    p_det = top["period"]
    snr_det = top["snr"]
    pix_det = top["peak_pixel"]
    p_err = abs(p_det - p_inj) / p_inj * 100.0

    print(f"Top Candidate (DM=1000, 1 deg off-axis):")
    print(f"  Detected Period: {p_det:.5f} s (Injected: {p_inj:.5f} s, Error: {p_err:.3f}%)")
    print(f"  Detection SNR:   {snr_det:.1f}")
    print(f"  Detected Pixel:  {pix_det} (Expected: ({target_x}, {target_y}))")

    assert p_err < 0.25, f"Period error {p_err:.3f}% exceeds tolerance 0.25%"
    assert pix_det == (target_x, target_y), f"Pixel mismatch: {pix_det} vs ({target_x}, {target_y})"
    assert snr_det > 50.0, f"Detection SNR {snr_det:.1f} is unexpectedly low"


if __name__ == "__main__":
    test_craco_option_c_dedisp_streaming()
    test_craco_high_dm_1000_1deg_offaxis_streaming()
