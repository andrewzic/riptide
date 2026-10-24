#!/usr/bin/env python3
"""
search_craco_ozstar.py

Execute Visibility FFA search (P = 1.0s to 5.0s) on dedispersed CRACO UV cubes on OzSTAR.
Generates candidate data products (plots, FITS maps, JSON summaries) for the highest S/N
candidate detected in each DM trial, as well as forced extraction at target coordinates.

Supports:
  1. Single DM cube search (--real-cube / --imag-cube)
  2. Batch search across all DM trials up to DM=500 (--data-dir, --dm-max 500)
  3. Slurm Job Array execution via --array-id ${SLURM_ARRAY_TASK_ID}
  4. Inspection and dry-run listing (--list-only)
  5. Post-run collation of parallel array results (--collate)
"""

import os
import sys
import gc
import re
import json
import glob
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

import riptide
from riptide.ffautils import plan_ffa
import riptide.libcpp as _libcpp


def parse_args():
    parser = argparse.ArgumentParser(description="Visibility FFA 1.0 - 5.0s period search on OzSTAR")
    parser.add_argument(
        "--data-dir",
        default="/fred/oz451/azic/data/SB77974",
        help="Directory containing CRACO UV cubes (default: /fred/oz451/azic/data/SB77974)",
    )
    parser.add_argument(
        "--real-cube",
        default=None,
        help="Path to specific real UV cube FITS file (overrides --data-dir)",
    )
    parser.add_argument(
        "--imag-cube",
        default=None,
        help="Path to specific imag UV cube FITS file",
    )
    parser.add_argument("--dm-min", type=float, default=0.0, help="Minimum DM to process (default: 0.0)")
    parser.add_argument("--dm-max", type=float, default=500.0, help="Maximum DM to process (default: 500.0)")
    parser.add_argument("--pmin", type=float, default=1.0, help="Minimum period to search in seconds (default: 1.0)")
    parser.add_argument("--pmax", type=float, default=5.0, help="Maximum period to search in seconds (default: 5.0)")
    parser.add_argument("--bins-min", type=int, default=10, help="Minimum phase bins (default: 10)")
    parser.add_argument("--bins-max", type=int, default=64, help="Maximum phase bins (default: 64)")
    parser.add_argument("--cand-x", type=int, default=274, help="Target Candidate X pixel coordinate (default: 274)")
    parser.add_argument("--cand-y", type=int, default=10, help="Target Candidate Y pixel coordinate (default: 10)")
    parser.add_argument("--out-prefix", default="craco_ffa_search", help="Prefix for outputs")
    parser.add_argument("--save-cube", action="store_true", default=False, help="Save 3D SNR cube to FITS")
    parser.add_argument("--skip-existing", action="store_true", default=False, help="Skip DM trials with existing plots")
    parser.add_argument(
        "--array-id",
        "--array-index",
        dest="array_id",
        type=int,
        default=None,
        help="Index of DM trial to process (or automatically reads $SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--list-only",
        "--dry-run",
        action="store_true",
        default=False,
        help="Scan and print matching DM trials and Slurm array configuration without running",
    )
    parser.add_argument(
        "--collate",
        action="store_true",
        default=False,
        help="Collate results from all completed array tasks into master CSV and S/N vs DM curve",
    )
    return parser.parse_args()


def find_dm_pairs(data_dir, dm_min=0.0, dm_max=500.0):
    """
    Search data_dir for CRACO UV cube pairs matching real/imag and dm within [dm_min, dm_max].
    Returns list of dicts: {'dm': float, 'real': path, 'imag': path} sorted ascending by dm.
    """
    patterns = [
        os.path.join(data_dir, "*dirty_cube-uv-real.cube.fits"),
        os.path.join(data_dir, "*", "*dirty_cube-uv-real.cube.fits"),
        os.path.join(data_dir, "*real.cube.fits"),
    ]
    real_files = []
    for pat in patterns:
        found = glob.glob(pat)
        if found:
            real_files.extend(found)

    real_files = sorted(list(set(real_files)))
    pairs = []
    for rf in real_files:
        m = re.search(r"dm([0-9.]+)", os.path.basename(rf))
        if not m:
            continue
        try:
            dm = float(m.group(1))
        except ValueError:
            continue

        if not (dm_min <= dm <= dm_max):
            continue

        # Look for matching imaginary cube
        imag_f = rf.replace("-real.cube.fits", "-imag.cube.fits")
        if not os.path.isfile(imag_f):
            imag_f = rf.replace("uv-real", "uv-imag")
        if not os.path.isfile(imag_f):
            imag_f = rf.replace("real", "imag")

        if os.path.isfile(imag_f):
            pairs.append({"dm": dm, "real": rf, "imag": imag_f})

    pairs.sort(key=lambda x: x["dm"])
    return pairs


def make_sky_wcs(hdr):
    """
    Construct or extract a 2D RA/Dec celestial WCS projection for synthesized image frames.
    """
    w = None
    try:
        w_cand = WCS(hdr).celestial
        if w_cand.has_celestial and w_cand.pixel_n_dim == 2:
            w = w_cand
    except Exception:
        pass

    if w is None:
        w = WCS(naxis=2)
        w.wcs.crpix = [193.0, 193.0]
        w.wcs.cdelt = [-22.0 / 3600.0, 22.0 / 3600.0]
        ra_center = 264.238783892833
        dec_center = -23.582287754619
        if "LATPOLE" in hdr and abs(float(hdr["LATPOLE"])) > 0:
            dec_center = float(hdr["LATPOLE"])
        w.wcs.crval = [ra_center, dec_center]
        w.wcs.ctype = ["RA---SIN", "DEC--SIN"]

    return w


def plot_candidate_product(
    detection_frame,
    cand_pgram,
    pgram_center,
    periods_arr,
    cand_x,
    cand_y,
    cand_p,
    cand_snr,
    dm_val,
    w,
    coord_cand,
    nx,
    ny,
    nsamp,
    tsamp,
    plot_path,
    title_prefix="Top Candidate",
    target_info=None,
):
    """
    Generate a publication-quality 4-panel Candidate Product sheet.
    """
    fig = plt.figure(figsize=(15, 11), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.95], hspace=0.28, wspace=0.24)

    # 1. Top Left: Full Sky S/N Map
    ax_full = fig.add_subplot(gs[0, 0])
    vmax = max(8.0, float(cand_snr) * 1.08)
    im_full = ax_full.imshow(detection_frame, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    ax_full.plot(192, 192, "w+", markersize=14, markeredgewidth=2, label="Phase Center (192, 192)")

    box_r = 22
    x0, x1 = max(0, cand_x - box_r), min(nx, cand_x + box_r)
    y0, y1 = max(0, cand_y - box_r), min(ny, cand_y + box_r)
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=1.8, linestyle="--")
    ax_full.add_patch(rect)
    ax_full.plot(cand_x, cand_y, "ro", markersize=10, fillstyle="none", markeredgewidth=2, label=f"Candidate ({cand_x}, {cand_y})")

    ax_full.set_title(
        f"Full S/N Sky Frame | P = {cand_p:.5f} s (DM = {dm_val:.1f} pc cm$^{{-3}}$)\nPeak S/N = {cand_snr:.2f} at Pixel ({cand_x}, {cand_y})",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_full.set_xlabel("RA Pixel (x)", fontsize=10.5)
    ax_full.set_ylabel("Dec Pixel (y)", fontsize=10.5)
    ax_full.legend(loc="upper right", fontsize=9.5)
    cb_full = plt.colorbar(im_full, ax=ax_full, fraction=0.046, pad=0.04)
    cb_full.set_label("Hybrid Boxcar S/N", fontsize=10)

    # 2. Top Right: High-Resolution Zoomed Cutout
    ax_zoom = fig.add_subplot(gs[0, 1])
    cutout = detection_frame[y0:y1, x0:x1]
    im_zoom = ax_zoom.imshow(cutout, origin="lower", extent=[x0, x1, y0, y1], cmap="viridis", vmin=0, vmax=vmax)
    ax_zoom.plot(cand_x, cand_y, "ro", markersize=14, markeredgewidth=2.5, fillstyle="none", label=f"Peak ({cand_x}, {cand_y})\nS/N = {cand_snr:.2f}")

    ra_str = coord_cand.ra.to_string(unit=u.hour, sep=':', precision=2)
    dec_str = coord_cand.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True)

    ax_zoom.set_title(
        f"Zoomed Cutout around Candidate ({cand_x}, {cand_y})\nRA (J2000): {ra_str} | Dec: {dec_str}",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_zoom.set_xlabel("RA Pixel (x)", fontsize=10.5)
    ax_zoom.set_ylabel("Dec Pixel (y)", fontsize=10.5)
    ax_zoom.legend(loc="upper right", fontsize=10)
    cb_zoom = plt.colorbar(im_zoom, ax=ax_zoom, fraction=0.046, pad=0.04)
    cb_zoom.set_label("Hybrid Boxcar S/N", fontsize=10)

    # 3. Bottom Left: Candidate Periodogram
    ax_pgram = fig.add_subplot(gs[1, 0])
    ax_pgram.plot(periods_arr, cand_pgram, color="#1f77b4", linewidth=1.3, label=f"Candidate Pixel ({cand_x}, {cand_y})")
    ax_pgram.plot(periods_arr, pgram_center, color="gray", linestyle=":", alpha=0.6, label="Phase Center Baseline")
    ax_pgram.axvline(cand_p, color="crimson", linestyle="--", linewidth=1.8, label=f"Peak: P = {cand_p:.5f} s (S/N = {cand_snr:.2f})")
    ax_pgram.set_title(f"Visibility FFA Periodogram at Pixel ({cand_x}, {cand_y})", fontsize=11.5, fontweight="bold")
    ax_pgram.set_xlabel("Trial Period (seconds)", fontsize=10.5, fontweight="bold")
    ax_pgram.set_ylabel("Hybrid Boxcar S/N", fontsize=10.5, fontweight="bold")
    ax_pgram.set_xlim(periods_arr.min(), periods_arr.max())
    ax_pgram.set_ylim(0.5, max(8.5, cand_snr * 1.15))
    ax_pgram.grid(True, linestyle=":", alpha=0.6)
    ax_pgram.legend(loc="upper right", fontsize=9.5)

    # 4. Bottom Right: Metadata Card
    ax_meta = fig.add_subplot(gs[1, 1])
    ax_meta.axis("off")

    gal = coord_cand.galactic
    obs_dur = nsamp * tsamp

    meta_text = (
        f"====================================================\n"
        f"        CRACO FFA CANDIDATE PRODUCT SUMMARY         \n"
        f"====================================================\n"
        f" Classification     : {title_prefix.upper()}\n"
        f" Dispersion Measure : {dm_val:7.2f} pc cm^-3\n"
        f" Detection S/N      : {cand_snr:7.2f}\n"
        f" Best Trial Period  : {cand_p:9.6f} s\n"
        f" Spin Frequency     : {1.0/cand_p:9.6f} Hz\n"
        f"----------------------------------------------------\n"
        f" Pixel Coordinates  : (X = {cand_x:3d}, Y = {cand_y:3d})\n"
        f" Celestial RA(J2000): {ra_str}\n"
        f" Celestial DEC(2000): {dec_str}\n"
        f" Galactic (l, b)    : l = {gal.l.deg:6.2f}°, b = {gal.b.deg:6.2f}°\n"
        f"----------------------------------------------------\n"
        f" Integration Steps  : {nsamp} snapshots\n"
        f" Sampling Interval  : {tsamp*1000.0:.2f} ms ({tsamp:.6f} s)\n"
        f" Total Observation  : {obs_dur:.1f} s ({obs_dur/60.0:.2f} min)\n"
        f" Search Bounds      : [{periods_arr.min():.2f}s, {periods_arr.max():.2f}s]\n"
    )
    if target_info:
        meta_text += (
            f"----------------------------------------------------\n"
            f" Forced Target Pixel: ({target_info['x']}, {target_info['y']})\n"
            f" Target S/N         : {target_info['snr']:.2f}\n"
            f" Target Best Period : {target_info['p']:.5f} s\n"
        )
    meta_text += f"===================================================="

    ax_meta.text(
        0.02,
        0.98,
        meta_text,
        transform=ax_meta.transAxes,
        fontsize=9.5,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1.5),
    )

    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def search_single_dm(real_file, imag_file, dm, args, plans=None, total_trials=None):
    """
    Execute Visibility FFA search on a single DM real/imag cube pair.
    Generates Candidate Products for the highest S/N candidate in this DM trial.
    """
    t0_dm = time.time()
    top_cand_plot = f"{args.out_prefix}_dm{dm:06.1f}_candidate_product.png"
    top_cand_json = f"{args.out_prefix}_dm{dm:06.1f}_candidate_product.json"
    top_cand_fits = f"{args.out_prefix}_dm{dm:06.1f}_candidate_map.fits"

    if args.skip_existing and os.path.isfile(top_cand_plot) and os.path.isfile(top_cand_json):
        print(f"Skipping DM={dm:.1f} (Candidate products already exist).")
        try:
            with open(top_cand_json, "r") as f:
                return json.load(f)
        except Exception:
            pass

    print(f"\n=================================================================")
    print(f"  Processing DM = {dm:6.1f} pc cm^-3")
    print(f"  Real: {os.path.basename(real_file)}")
    print(f"  Imag: {os.path.basename(imag_file)}")
    print(f"=================================================================")

    # 1. Load FITS files
    with fits.open(real_file, memmap=True) as hr, fits.open(imag_file, memmap=True) as hi:
        hdr = hr[0].header
        r = np.nan_to_num(hr[0].data.squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
        i = np.nan_to_num(hi[0].data.squeeze(), nan=0.0, posinf=0.0, neginf=0.0)

    vis = (r + 1j * i).astype(np.complex64)
    nsamp, ny, nx = vis.shape
    del r, i

    # Read metadata
    tsamp = float(hdr.get("CDELT3", hdr.get("CDELT5", 0.11056)))
    dm_val = float(hdr.get("DM", dm))

    # 2. Extract active UV tracks
    uv_amp = np.abs(vis).mean(axis=0)
    active = uv_amp > 0
    u_idx, v_idx = np.nonzero(active)
    uv_indices = np.column_stack([u_idx, v_idx]).astype(np.uint64)
    M = len(u_idx)
    print(f"  Active UV cells: {M} / {nx*ny} ({100.0*M/(nx*ny):.1f}% coverage) | nsamp={nsamp}")

    if M == 0:
        print(f"  [WARNING] Cube for DM={dm:.1f} contains 0 active UV cells. Skipping.")
        del vis
        gc.collect()
        return None

    dga = np.ascontiguousarray(vis[:, u_idx, v_idx].T, dtype=np.complex64)
    del vis
    gc.collect()

    # 3. Plan FFA if not provided
    if plans is None or total_trials is None:
        plans = plan_ffa(nsamp, tsamp, args.pmin, args.pmax, args.bins_min, args.bins_max)
        total_trials = sum(p["rows_eval"] for p in plans)

    all_periods = []
    all_snr_imgs = []
    all_best_snrs = []

    t0_ffa = time.time()
    for idx, p in enumerate(plans, 1):
        bins = p["bins"]
        tau = p["tau"]
        rows_eval = p["rows_eval"]
        ds = p["downsample_factor"]

        ffa_data, periods = _libcpp.vis_ffa_transform_basep(dga, bins, tau, rows_eval, ds)
        tp_f32 = periods.astype(np.float32)

        res = _libcpp.image_ffa_test_img(ffa_data, uv_indices, tp_f32, nx, ny)

        for r_idx in range(rows_eval):
            all_periods.append(float(tp_f32[r_idx]))
            all_snr_imgs.append(res["images"][r_idx])
            all_best_snrs.append(float(res["snrs"][r_idx]))

    del dga
    gc.collect()

    periods_arr = np.array(all_periods, dtype=np.float32)
    snr_stack = np.array(all_snr_imgs, dtype=np.float32)
    del all_snr_imgs
    gc.collect()

    # Sort strictly by period
    sort_idx = np.argsort(periods_arr)
    periods_arr = periods_arr[sort_idx]
    snr_stack = snr_stack[sort_idx]

    # 4. Target Candidate Forced Measurement
    cx, cy = args.cand_x, args.cand_y
    forced_pgram = snr_stack[:, cy, cx]
    forced_best_idx = int(np.argmax(forced_pgram))
    forced_best_p = float(periods_arr[forced_best_idx])
    forced_best_snr = float(forced_pgram[forced_best_idx])

    # 5. Global Highest S/N Candidate Detection in this DM
    frame_maxes = np.array([img.max() for img in snr_stack])
    blind_best_idx = int(np.argmax(frame_maxes))
    blind_best_p = float(periods_arr[blind_best_idx])
    blind_best_frame = snr_stack[blind_best_idx]
    blind_best_snr = float(blind_best_frame.max())
    blind_y, blind_x = np.unravel_index(np.argmax(blind_best_frame), (ny, nx))
    blind_x, blind_y = int(blind_x), int(blind_y)
    blind_pgram = snr_stack[:, blind_y, blind_x]

    # Baseline periodogram at phase center
    pgram_center = snr_stack[:, 192, 192]

    # Synthesized sky WCS
    w = make_sky_wcs(hdr)
    coord_cand = w.pixel_to_world(cx, cy)
    coord_blind = w.pixel_to_world(blind_x, blind_y)

    elapsed_ffa = time.time() - t0_ffa
    rate = len(periods_arr) / elapsed_ffa if elapsed_ffa > 0 else 0
    print(f"  FFA Execution: {len(periods_arr)} periods in {elapsed_ffa:.1f}s ({rate:.1f} trials/s)")
    print(f"  >>> HIGHEST S/N CANDIDATE in DM={dm_val:.1f}: Pixel ({blind_x}, {blind_y}) | S/N = {blind_best_snr:5.2f} | P = {blind_best_p:7.5f} s")
    print(f"      Coordinates: RA = {coord_blind.ra.to_string(unit=u.hour, sep=':', precision=2)}, Dec = {coord_blind.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True)}")
    print(f"  Target Pixel ({cx}, {cy}): S/N = {forced_best_snr:5.2f} at P = {forced_best_p:7.5f} s")

    # 6. Generate Candidate Product for Highest S/N Candidate
    target_info = {"x": cx, "y": cy, "snr": forced_best_snr, "p": forced_best_p}
    plot_candidate_product(
        detection_frame=blind_best_frame,
        cand_pgram=blind_pgram,
        pgram_center=pgram_center,
        periods_arr=periods_arr,
        cand_x=blind_x,
        cand_y=blind_y,
        cand_p=blind_best_p,
        cand_snr=blind_best_snr,
        dm_val=dm_val,
        w=w,
        coord_cand=coord_blind,
        nx=nx,
        ny=ny,
        nsamp=nsamp,
        tsamp=tsamp,
        plot_path=top_cand_plot,
        title_prefix="Top Candidate",
        target_info=target_info if (blind_x != cx or blind_y != cy) else None,
    )
    print(f"  [PRODUCED] Candidate Plot : {top_cand_plot}")

    # Save 2D FITS Map of Candidate Detection Frame
    try:
        h_cand = w.to_header()
        h_cand["DM"] = dm_val
        h_cand["PERIOD"] = blind_best_p
        h_cand["PEAK_SNR"] = blind_best_snr
        h_cand["PIX_X"] = blind_x
        h_cand["PIX_Y"] = blind_y
        fits.writeto(top_cand_fits, blind_best_frame, header=h_cand, overwrite=True)
        print(f"  [PRODUCED] Candidate FITS : {top_cand_fits}")
    except Exception as e:
        print(f"  Warning saving candidate FITS: {e}")

    # If target pixel is distinct from top peak, also produce target candidate product
    if (blind_x != cx or blind_y != cy):
        target_plot = f"{args.out_prefix}_dm{dm:06.1f}_target_product.png"
        plot_candidate_product(
            detection_frame=snr_stack[forced_best_idx],
            cand_pgram=forced_pgram,
            pgram_center=pgram_center,
            periods_arr=periods_arr,
            cand_x=cx,
            cand_y=cy,
            cand_p=forced_best_p,
            cand_snr=forced_best_snr,
            dm_val=dm_val,
            w=w,
            coord_cand=coord_cand,
            nx=nx,
            ny=ny,
            nsamp=nsamp,
            tsamp=tsamp,
            plot_path=target_plot,
            title_prefix=f"Forced Target ({cx}, {cy})",
            target_info=None,
        )
        print(f"  [PRODUCED] Target Plot    : {target_plot}")

    if args.save_cube:
        cube_file = f"{args.out_prefix}_dm{dm:06.1f}_snr_cube.fits"
        print(f"  Saving full 3D SNR cube to {cube_file}...")
        w_cube = WCS(naxis=2)
        w_cube.wcs.crpix = [193.0, 193.0]
        w_cube.wcs.cdelt = [-22.0 / 3600.0, 22.0 / 3600.0]
        w_cube.wcs.crval = [264.238783892833, -23.582287754619]
        w_cube.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        h_cube = w_cube.to_header()
        h_cube["NAXIS"] = 3
        h_cube["NAXIS1"] = nx
        h_cube["NAXIS2"] = ny
        h_cube["NAXIS3"] = len(periods_arr)
        h_cube["DM"] = dm_val
        primary_hdu = fits.PrimaryHDU(data=snr_stack, header=h_cube)
        col_p = fits.Column(name="PERIODS", format="1E", array=periods_arr)
        table_hdu = fits.BinTableHDU.from_columns([col_p], name="PERIODS")
        fits.HDUList([primary_hdu, table_hdu]).writeto(cube_file, overwrite=True)

    del snr_stack, forced_pgram, blind_pgram, blind_best_frame
    gc.collect()

    result_dict = {
        "dm": dm_val,
        "cand_snr": forced_best_snr,
        "cand_period": forced_best_p,
        "blind_snr": blind_best_snr,
        "blind_period": blind_best_p,
        "blind_x": blind_x,
        "blind_y": blind_y,
        "blind_ra": coord_blind.ra.to_string(unit=u.hour, sep=':', precision=2),
        "blind_dec": coord_blind.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True),
        "cand_ra": coord_cand.ra.to_string(unit=u.hour, sep=':', precision=2),
        "cand_dec": coord_cand.dec.to_string(unit=u.deg, sep=':', precision=2, alwayssign=True),
        "candidate_plot": top_cand_plot,
        "candidate_fits": top_cand_fits,
        "duration_sec": time.time() - t0_dm,
    }

    # Save JSON summary
    try:
        with open(top_cand_json, "w") as jf:
            json.dump(result_dict, jf, indent=2)
        print(f"  [PRODUCED] Candidate JSON : {top_cand_json}")
    except Exception as e:
        print(f"  Warning saving JSON: {e}")

    return result_dict


def collate_results(out_prefix, cand_x, cand_y):
    """
    Search for all per-DM JSON candidate products and produce master table and multi-DM curve.
    """
    json_files = sorted(glob.glob(f"{out_prefix}_dm*_candidate_product.json"))
    if not json_files:
        json_files = sorted(glob.glob(f"{out_prefix}_dm*_result.json"))

    if not json_files:
        print(f"[COLLATE ERROR] No results found matching '{out_prefix}_dm*_candidate_product.json'")
        return

    print(f"\nCollating {len(json_files)} DM candidate products matching '{out_prefix}_dm*...'")
    all_results = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                all_results.append(json.load(f))
        except Exception as e:
            print(f"  Warning reading {jf}: {e}")

    if not all_results:
        return

    df_summary = pd.DataFrame(all_results).sort_values("dm")
    csv_file = f"{out_prefix}_top_candidates.csv"
    df_summary.to_csv(csv_file, index=False)
    print(f"\n==========================================================================================")
    print(f"                   TOP S/N CANDIDATES ACROSS ALL DM TRIALS                                ")
    print(f"==========================================================================================")
    cols = ["dm", "blind_snr", "blind_period", "blind_x", "blind_y", "blind_ra", "blind_dec", "cand_snr"]
    print(df_summary[cols].to_string(index=False))
    print(f"\nSaved master candidate table to: {csv_file}")

    plot_dm_curve(all_results, out_prefix, cand_x, cand_y)

    best_global = df_summary.loc[df_summary["blind_snr"].idxmax()]
    print(f"\n==========================================================================================")
    print(f"               GLOBAL CHAMPION CANDIDATE ACROSS ALL DM TRIALS                             ")
    print(f"==========================================================================================")
    print(f"  Dispersion Measure : {best_global['dm']:.1f} pc cm^-3")
    print(f"  Highest Peak S/N   : {best_global['blind_snr']:.2f}")
    print(f"  Period             : {best_global['blind_period']:.6f} s ({1.0/best_global['blind_period']:.6f} Hz)")
    print(f"  Pixel Position     : ({int(best_global['blind_x'])}, {int(best_global['blind_y'])})")
    print(f"  Sky Coordinates    : RA = {best_global['blind_ra']}, Dec = {best_global['blind_dec']}")
    print(f"==========================================================================================")


def plot_dm_curve(results, out_prefix, cand_x, cand_y):
    """
    Generate master S/N vs DM curve across all evaluated DM trials.
    """
    if len(results) < 2:
        return

    df = pd.DataFrame(results).sort_values("dm")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=150)

    # 1. Candidate S/N vs DM
    ax1.plot(df["dm"], df["cand_snr"], "o-", color="#1f77b4", linewidth=2, markersize=6, label=f"Target Candidate Pixel ({cand_x}, {cand_y})")
    best_cand_row = df.loc[df["cand_snr"].idxmax()]
    ax1.axvline(best_cand_row["dm"], color="crimson", linestyle="--", alpha=0.8,
                label=f"Peak DM = {best_cand_row['dm']:.1f} pc cm^-3 (S/N = {best_cand_row['cand_snr']:.2f}, P = {best_cand_row['cand_period']:.5f}s)")
    ax1.set_ylabel("Hybrid Boxcar S/N", fontsize=11, fontweight="bold")
    ax1.set_title(f"Visibility FFA: Target Candidate Pixel ({cand_x}, {cand_y}) S/N vs DM", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=10)

    # 2. Global Blind Peak S/N vs DM
    ax2.plot(df["dm"], df["blind_snr"], "s-", color="#ff7f0e", linewidth=1.8, markersize=5, label="Highest S/N Candidate in Each DM")
    best_blind_row = df.loc[df["blind_snr"].idxmax()]
    ax2.axvline(best_blind_row["dm"], color="red", linestyle=":", alpha=0.7,
                label=f"Global Champion DM = {best_blind_row['dm']:.1f} pc cm^-3 (S/N = {best_blind_row['blind_snr']:.2f} at pixel {int(best_blind_row['blind_x'])},{int(best_blind_row['blind_y'])})")
    ax2.set_xlabel("Dispersion Measure (pc cm^-3)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Highest Candidate S/N", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=10)

    curve_file = f"{out_prefix}_snr_vs_dm_curve.png"
    plt.savefig(curve_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multi-DM detection curve: {curve_file}")


def main():
    args = parse_args()
    t_start = time.time()

    if args.collate:
        collate_results(args.out_prefix, args.cand_x, args.cand_y)
        return

    print("=================================================================")
    print("      CRACO Visibility FFA Search (1.0s - 5.0s) on OzSTAR        ")
    print("=================================================================")
    print(f"Period Range: [{args.pmin:.2f}s, {args.pmax:.2f}s]")
    print(f"Target Candidate Pixel: ({args.cand_x}, {args.cand_y})")

    if args.real_cube:
        if not args.imag_cube:
            imag_f = args.real_cube.replace("-real.cube.fits", "-imag.cube.fits")
            if not os.path.isfile(imag_f):
                imag_f = args.real_cube.replace("uv-real", "uv-imag")
            args.imag_cube = imag_f

        m = re.search(r"dm([0-9.]+)", os.path.basename(args.real_cube))
        dm_val = float(m.group(1)) if m else 0.0
        tasks = [{"dm": dm_val, "real": args.real_cube, "imag": args.imag_cube}]
    else:
        tasks = find_dm_pairs(args.data_dir, dm_min=args.dm_min, dm_max=args.dm_max)

    if not tasks:
        print(f"[ERROR] No matching CRACO UV cube pairs found in {args.data_dir} with DM in [{args.dm_min}, {args.dm_max}].")
        sys.exit(1)

    if args.list_only:
        print(f"\nDiscovered {len(tasks)} DM trial(s) in {args.data_dir} [DM <= {args.dm_max}]:")
        for i, t in enumerate(tasks):
            print(f"  Task ID {i:3d} : DM = {t['dm']:6.1f} pc cm^-3 | {os.path.basename(t['real'])}")
        print("\n=================================================================")
        print(f"Recommended Slurm Array Directive: #SBATCH --array=0-{len(tasks)-1}%10")
        print("=================================================================")
        return

    array_idx = args.array_id
    if array_idx is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        try:
            array_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        except ValueError:
            pass

    is_array_job = (array_idx is not None)

    if is_array_job:
        if 0 <= array_idx < len(tasks):
            assigned_task = tasks[array_idx]
            print("\n=================================================================")
            print(f"  [SLURM ARRAY TASK ID: {array_idx}] Assigned 1 of {len(tasks)} DM trials")
            print(f"  Target DM  : {assigned_task['dm']:.1f} pc cm^-3")
            print(f"  Input File : {os.path.basename(assigned_task['real'])}")
            print("=================================================================")
            tasks = [assigned_task]
        else:
            print(f"[ERROR] Array index {array_idx} out of range (valid: 0 to {len(tasks)-1})")
            sys.exit(1)
    else:
        print(f"\nBatch Mode: Found {len(tasks)} DM trial(s) to search sequentially:")
        for i, t in enumerate(tasks, 1):
            print(f"  [{i:2d}/{len(tasks):2d}] DM = {t['dm']:6.1f} pc cm^-3 | {os.path.basename(t['real'])}")

    # Pre-plan FFA using the first cube's dimensions
    with fits.open(tasks[0]["real"], memmap=True) as h0:
        nsamp = h0[0].header.get("NAXIS3", 5498)
        tsamp = float(h0[0].header.get("CDELT3", h0[0].header.get("CDELT5", 0.11056)))

    plans = plan_ffa(nsamp, tsamp, args.pmin, args.pmax, args.bins_min, args.bins_max)
    total_trials = sum(p["rows_eval"] for p in plans)
    print(f"\nPre-computed FFA Grid: {len(plans)} Base Plans | {total_trials} Period Trials")

    all_results = []
    for idx, task in enumerate(tasks, 1):
        if not is_array_job:
            print(f"\n>>> Running trial {idx}/{len(tasks)} (DM = {task['dm']:.1f} pc cm^-3) <<<")
        res = search_single_dm(task["real"], task["imag"], task["dm"], args, plans, total_trials)
        if res:
            all_results.append(res)

    t_total = time.time() - t_start

    if not is_array_job and all_results:
        collate_results(args.out_prefix, args.cand_x, args.cand_y)

    print(f"\nTask completed in {t_total:.1f}s ({t_total/60:.2f} min).")


if __name__ == "__main__":
    main()
