import numpy as np

### Local module imports
import riptide.libcpp as libcpp
from .ffautils import generate_width_trials
from .periodogram import Periodogram, UV_FFA, UV_FFA_basep
from .timing import timing

@timing
def vis_ffa_search(
    tseries,
    uv_ind, #tuple to hold (u, v) index
    period_min=1.0,
    period_max=300.0,
    fpmin=8,
    bins_min=6,
    bins_max=256,
    ducy_max=0.5,
    wtsp=1.5,
    deredden=False,
    rmed_width=4.0,
    rmed_minpts=101,
    already_normalised=True,
):
    """
    Run a FFA search of a single visibility TimeSeries object (one uv cell), producing FFA blocks.

    Parameters
    ----------
    tseries : TimeSeries
        The visibility time series object to search
    period_min : float
        Minimum period to search in seconds
    period_max : float
        Maximum period to search in seconds
    fpmin : int
        Minimum number of signal periods that must fit in the data. In other
        words, place a cap on period_max equal to DATA_LENGTH / fpmin
    bins_min : int
        Minimum number of phase bins in the folded data. Higher values
        provide better duty cycle resolution. As the code searches longer trial
        periods, the data are iteratively downsampled so that the number of
        phase bins remains between bins_min and bins_max
    bins_max : int
        Maximum number of phase bins in the folded data. Must be strictly
        larger than bins_min, approx. 10% larger is a good choice
    wtsp : float
        Multiplicative factor between consecutive boxcar width trials. The
        smallest width is always 1 phase bin, and the sequence of width
        trials is generated with the formula:
        W(n+1) = max( floor(wtsp x W(n)), W(n) + 1 )
        wtsp = 1.5 gives the following sequence of width trials (in number of
        phase bins): 1, 2, 3, 4, 6, 9, 13, 19 ...
    ducy_max : float
        Maximum duty cycle to optimally search. Limits the maximum width of the
        boxcar matched filters applied to any given profile.
        Example: on a 300 phase bin profile, ducy_max = 0.2 means that no
        boxcar filter of width > 60 bins will be applied
    deredden : bool
        Subtract red noise from the time series before searching
    rmed_width : float
        The width of the running median filter to subtract from the input data
        before processing, in seconds
    rmed_minpts : int
        The running median is calculated of a time scrunched version of the
        input data to save time: rmed_minpts is the minimum number of
        scrunched samples that must fit in the running median window
        Lower values make the running median calculation less accurate but
        faster, due to allowing a higher scrunching factor
    already_normalised : bool
        Assume that the data are already normalised to zero mean and unit
        standard deviation

    Returns
    -------
    ts : TimeSeries
        The de-reddened and normalised time series that was actually searched
    ffa_result : UV_FFA
        The output of the search, which contains among other things a list of 2D arrays
        the input visibilities folded at different base periods
    """
    ### Prepare data: deredden then normalise IN THAT ORDER
    #shouldn't do this on visibility data hence deredden = False and already_normalised = True by default
    if deredden:
        tseries = tseries.deredden(rmed_width, minpts=rmed_minpts)
    if not already_normalised:
        tseries = tseries.normalise()
    
    #below is akin to periodogram but preserves pulse phase information, 
    # keeping FFA results in butterfly blocks
    periods, foldbins, base_periods, tsamps, vis_ffa_list = libcpp.vis_ffa_transform(
        tseries.data, tseries.tsamp, period_min, period_max, bins_min, bins_max
    )
    ffa_result = UV_FFA(uv_ind, periods, foldbins, base_periods, tsamps, vis_ffa_list, metadata=tseries.metadata)
    return tseries, ffa_result

@timing
def vis_ffa_search_basep(
    dynamic_grid_array,
    base_period, #in bins
    uv_inds,
    tsamp,
    rows_eval=0,
    downsample_factor=1.0
):  
    """
    Run a FFA search of a multiple visibility timeseries to produce ffa blocks for a given base period

    Parameters
    ----------
    vis_data : np array
        The dense visibility time series array to search - should be N_uv x N_t
    base_period:
        The (integer) base period to fold at in bins
    rows_eval : int, optional
        Number of shift rows to evaluate (0 means all rows)
    downsample_factor : float, optional
        Downsampling factor (>= 1.0) to apply before folding

    Returns
    -------
    ffa_result : UV_FFA_basep
        The output of the search, which contains the 3D array of folded visibilities
    """
    ffa_grid_array, periods = libcpp.vis_ffa_transform_basep(
        dynamic_grid_array,
        base_period,
        tsamp,
        int(rows_eval),
        float(downsample_factor)
    )

    foldbins = np.full(len(periods), base_period, dtype=np.int64)

    ffa_result = UV_FFA_basep(uv_inds, periods, foldbins, base_period, tsamp, ffa_grid_array)
    return ffa_result


@timing
def vis_ffa_image_candidates(
    ffa_blocks,
    uv_indices,
    periods,
    nx,
    ny,
    snr_thresh=8.0,
    ducy_max=0.5,
    wtsp=1.5,
    max_candidates_width=100,
    max_candidates_all=1000,
    mask_radius=2
):
    """
    Run C++-based imaging + candidate search on a set of FFA blocks.

    Parameters
    ----------
    ffa_blocks : list of np.ndarray (complex64)
        Each block is a (nperiod, nphase) folded visibility from FFA.
    uv_indices : np.ndarray, shape (M, 2), int
        Integer (u, v) indices for each block.
    nx, ny : int
        Output image dimensions.
    snr_thresh : float
        Minimum SNR for a detection to be considered a candidate.
    ducy_max, wtsp : float
        Parameters for width trial generation.
    max_candidates_width : int
        Max candidates to keep for each width trial.
    max_candidates_all : int
        Max total candidates to keep across all widths.
    mask_radius : int
        Radius (in pixels) to mask around each found candidate.

    Returns
    -------
    candidates : list of dict
        Each dict has { 'x', 'y', 'snr', 'width_bins' }.
    """
    uv_indices = np.asarray(uv_indices)
    if uv_indices.shape != (len(ffa_blocks), 2):
        raise ValueError("uv_indices must have shape (M, 2) matching ffa_blocks length")
    
    # Generate widths in bins
    nbins = ffa_blocks[0].shape[1]
    widths = generate_width_trials(nbins, ducy_max=ducy_max, wtsp=wtsp).astype(np.uint64)

    # Call C++ candidate search
    candidates = libcpp.image_ffa_candidates(
        ffa_blocks,
        uv_indices.astype(np.uint64, copy=False),
        widths,
        np.array(periods, dtype=np.float32),
        nx,
        ny,
        snr_thresh,
        max_candidates_width,
        max_candidates_all,
        mask_radius
    )   
    return candidates

@timing
def test_vis_ffa_image(
    ffa_blocks,
    uv_indices,
    nx,
    ny
):
    """
    Run C++-based imaging + candidate search on a set of FFA blocks.

    Parameters
    ----------
    ffa_blocks : list of np.ndarray (complex64)
        Each block is a (phase, freq) folded visibility from FFA.
    uv_indices : np.ndarray, shape (M, 2), int
        Integer (u, v) indices for each block.
    nx, ny : int
        Output image dimensions.

    Returns
    -------
    results : dict
    """
    uv_indices = np.asarray(uv_indices)
    if uv_indices.shape != (len(ffa_blocks), 2):
        raise ValueError("uv_indices must have shape (M, 2) matching ffa_blocks length")

    # Call C++ test
    results = libcpp.image_ffa_test(
        ffa_blocks,
        uv_indices.astype(np.uint64, copy=False),
        nx,
        ny
    )   
    return results

@timing
def test_vis_ffa_image_imgs(
    ffa_blocks,
    uv_indices,
    periods,
    nx,
    ny
):
    """
    Run C++-based imaging + candidate search on a set of FFA blocks.

    Parameters
    ----------
    ffa_blocks : list of np.ndarray (complex64)
        Each block is a (phase, freq) folded visibility from FFA.
    uv_indices : np.ndarray, shape (M, 2), int
        Integer (u, v) indices for each block.
    periods : list or np.ndarray
        Trial periods.
    nx, ny : int
        Output image dimensions.

    Returns
    -------
    results : list of dict
    """
    uv_indices = np.asarray(uv_indices)
    if uv_indices.shape != (len(ffa_blocks), 2):
        raise ValueError("uv_indices must have shape (M, 2) matching ffa_blocks length")

    # Call C++ test
    results = libcpp.image_ffa_test_img(
        ffa_blocks,
        uv_indices.astype(np.uint64, copy=False),
        periods,
        nx,
        ny
    )   
    return results


@timing
def baseline_ffa_search(
    msname_or_data,
    period_min=1.0,
    period_max=300.0,
    bins_min=16,
    bins_max=64,
    dm=0.0,
    collapse_channels=False,
    exact_uvw=True,
    segment_duration_s=None,
    nx=384,
    ny=384,
    pixsize_x=22.0 / 206265.0,
    pixsize_y=22.0 / 206265.0,
    snr_thresh=8.0,
    **kwargs
):
    """
    Run baseline-domain FFA search and direct ducc0.wgridder imaging without early UV gridding.

    Parameters
    ----------
    msname_or_data : str or tuple
        Path to Measurement Set or tuple of (vis_3d, wgt_3d, uvw_3d, chan_freq, dt).
    period_min, period_max : float
        Trial period search range in seconds.
    bins_min, bins_max : int
        Phase folding bin range (e.g. 16 to 64).
    dm : float
        Dispersion measure in pc/cm^3.
    collapse_channels : bool
        If True, sum across channels before folding (standard mode).
        If False (default), fold pseudo-baselines with exact wavelength coordinates.
    exact_uvw : bool
        If True (default) and collapse_channels is False, compute exact per-channel UVW delay tracks.
    segment_duration_s : float, optional
        Stationary segment duration in seconds. If provided, partitions time series into
        segments where UV tracks are approximately stationary, then synthesizes across segments.
        If None, runs full-observation baseline-folded imaging.
    nx, ny : int
        Output image dimensions.
    pixsize_x, pixsize_y : float
        Pixel size in radians.
    snr_thresh : float
        Candidate detection SNR threshold.

    Returns
    -------
    result : BaselineFFAResult
    """
    try:
        from fastducc.baseline_ffa import search_baseline_ffa
    except ImportError:
        raise ImportError("fastducc is required for baseline_ffa_search.")

    return search_baseline_ffa(
        msname_or_data,
        period_min=period_min,
        period_max=period_max,
        bins_min=bins_min,
        bins_max=bins_max,
        dm=dm,
        collapse_channels=collapse_channels,
        exact_uvw=exact_uvw,
        segment_duration_s=segment_duration_s,
        npix_x=nx,
        npix_y=ny,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        snr_thresh=snr_thresh,
        **kwargs
    )


@timing
def uv_ffa_streaming_search(
    msname_or_data,
    period_min=1.0,
    period_max=300.0,
    bins_min=16,
    bins_max=64,
    dm=0.0,
    exact_uvw=True,
    collapse_channels=False,
    nx=384,
    ny=384,
    pixsize_x=22.0 / 206265.0,
    pixsize_y=22.0 / 206265.0,
    epsilon=1e-5,
    do_wgridding=True,
    threshold=1e-4,
    nthreads=1,
    snr_thresh=6.0,
    top_candidates=5,
    **kwargs
):
    """
    Direct visibility streaming UV-domain FFA search.
    Supports exact wideband visibility dedispersion and per-channel baseline migration tracking.
    Streams dedispersed visibility timestamps into oversampled 2D UV grids via ducc0.wgridder.vis2grid,
    builds the Dynamic Grid Array (DGA) in memory without intermediate FITS cubes,
    executes the C++ Fast Folding Algorithm on each active UV cell, and reconstructs
    candidate dirty images with kernel deconvolution for detected periodic signals.

    Parameters
    ----------
    msname_or_data : str, tuple, or list
        Path to Measurement Set or tuple of (vis_3d, wgt_3d, uvw_3d, chan_freq, dt).
    period_min, period_max : float
        Trial period search range in seconds.
    bins_min, bins_max : int
        Phase folding bin range (e.g. 16 to 64).
    dm : float
        Dispersion measure in pc/cm^3 (default: 0.0).
    exact_uvw : bool
        If True, computes exact per-channel (u, v, w) delay tracks and dimensionless
        wavelength scaling, preserving maximum spatial frequency fidelity.
    collapse_channels : bool
        If True, collapses channels into 1 frequency before gridding.
        If False (default), grids all pseudo-baselines with exact wavelength coordinates.
    nx, ny : int
        Dirty image dimensions.
    pixsize_x, pixsize_y : float
        Pixel size in radians.
    epsilon : float
        DUCC gridding accuracy parameter.
    threshold : float
        UV cell threshold for sparse retention.
    nthreads : int
        Number of CPU threads.
    snr_thresh : float
        Candidate detection SNR threshold.
    top_candidates : int
        Maximum number of candidate images to synthesize.

    Returns
    -------
    results : dict
        Contains detected candidates, periodograms, folded profiles, and reconstructed dirty images.
    """
    import ducc0
    from .time_series import VisTimeSeries
    from .ffautils import plan_ffa

    # Wideband Dedispersion & Exact UVW baseline migration preparation
    if isinstance(msname_or_data, (tuple, list)) and len(msname_or_data) >= 4:
        vis_raw = msname_or_data[0]
        if vis_raw.ndim == 3 and vis_raw.shape[1] > 1:
            try:
                from fastducc.baseline_ffa import prepare_baseline_series
                vis_3d = msname_or_data[0]
                wgt_3d = msname_or_data[1] if len(msname_or_data) == 5 else None
                uvw_3d = msname_or_data[2] if len(msname_or_data) == 5 else msname_or_data[1]
                chan_freq = msname_or_data[3] if len(msname_or_data) == 5 else msname_or_data[2]
                dt = msname_or_data[4] if len(msname_or_data) == 5 else msname_or_data[3]

                vis_2d, wgt_2d, uvw_track, eff_freq, dt = prepare_baseline_series(
                    vis_3d, wgt_3d, uvw_3d, chan_freq, dt,
                    dm=dm, collapse_channels=collapse_channels, exact_uvw=exact_uvw
                )
                msname_or_data = (vis_2d, wgt_2d, uvw_track, eff_freq, dt)
            except ImportError as exc:
                if dm != 0.0:
                    raise ImportError("fastducc is required for wideband visibility dedispersion (dm != 0).") from exc
    elif isinstance(msname_or_data, str) and (dm != 0.0 or exact_uvw):
        try:
            from fastducc.imaging import load_chunk_data
            from fastducc.baseline_ffa import prepare_baseline_series
            start_idx = kwargs.pop("start_time_idx", None)
            end_idx = kwargs.pop("end_time_idx", None)
            times, chan_freq, dt, vis_3d, wgt_3d, uvw_3d = load_chunk_data(
                msname_or_data,
                start_time_idx=start_idx,
                end_time_idx=end_idx,
                corr_mode="average",
                average_correlations=True
            )
            vis_2d, wgt_2d, uvw_track, eff_freq, dt = prepare_baseline_series(
                vis_3d, wgt_3d, uvw_3d, chan_freq, dt,
                dm=dm, collapse_channels=collapse_channels, exact_uvw=exact_uvw
            )
            msname_or_data = (vis_2d, wgt_2d, uvw_track, eff_freq, dt)
        except ImportError as exc:
            if dm != 0.0:
                raise ImportError("fastducc is required for wideband visibility dedispersion (dm != 0).") from exc

    # 1. Stream visibilities directly into VisTimeSeries
    vis_ts = VisTimeSeries.from_visibility_stream(
        msname_or_data,
        npix_x=nx,
        npix_y=ny,
        pixsize_x=pixsize_x,
        pixsize_y=pixsize_y,
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        threshold=threshold,
        nthreads=nthreads,
        direct_dga=True
    )

    dga = vis_ts.dga
    if dga is None or dga.shape[0] == 0:
        return {
            "candidates": [],
            "periods": np.array([], dtype=np.float64),
            "snrs": np.array([], dtype=np.float32),
            "vis_timeseries": vis_ts,
        }

    nsamp = dga.shape[1]
    tsamp = vis_ts.tsamp
    unique_uv = vis_ts.unique_uv
    nu, nv = vis_ts.nu, vis_ts.nv
    cfu, cfv = vis_ts.cfu, vis_ts.cfv

    # Precompute sparse UV indices, conjugate Hartley indices, and 2D kernel correction factor
    u_idx = unique_uv[:, 0]
    v_idx = unique_uv[:, 1]
    xu_idx = (-u_idx) % nu
    xv_idx = (-v_idx) % nv

    icfu = np.abs(nx // 2 - np.arange(nx))
    icfv = np.abs(ny // 2 - np.arange(ny))
    cf_2d = (cfu[icfu, None] * cfv[None, icfv]).astype(np.float32)
    hx, hy = nx // 2, ny // 2
    rx, ry = nx - hx, ny - hy

    # 2. Plan FFA searches
    ffa_plans = plan_ffa(nsamp, tsamp, period_min, period_max, bins_min, bins_max)

    candidates = []
    all_periods = []
    all_snrs = []

    # 3. Execute FFA over each base period block
    for plan in ffa_plans:
        downsample_fac = plan["downsample_factor"]
        base_period = plan["base_period"]
        bins = int(plan["bins"])
        rows_eval = plan["rows_eval"]
        tau = plan["tau"]

        ffa_res = vis_ffa_search_basep(
            dynamic_grid_array=dga,
            base_period=bins,
            uv_inds=unique_uv,
            tsamp=tau,
            rows_eval=rows_eval,
            downsample_factor=downsample_fac
        )

        ffa_arr = ffa_res.ffa_array  # (N_active_uv, N_periods, bins)

        # Pre-allocate real Hartley grid and cropped dirty image cube per base period block
        rgrid_cube = np.zeros((bins, nu, nv), dtype=np.float32)
        dirty_cube = np.empty((bins, nx, ny), dtype=np.float32)

        # Stream trial periods with direct sparse conjugate scatter into Hartley grid
        for ip, p_val in enumerate(ffa_res.periods):
            vis_p = ffa_arr[:, ip, :].T  # (bins, M) complex64
            rgrid_cube.fill(0.0)
            rgrid_cube[:, u_idx, v_idx] = 0.5 * (vis_p.real - vis_p.imag)
            rgrid_cube[:, xu_idx, xv_idx] += 0.5 * (vis_p.real + vis_p.imag)

            # 2D Fast Hartley Transform across spatial dimensions (-2, -1) in-place
            ducc0.fft.genuine_fht(rgrid_cube, axes=(-2, -1), nthreads=nthreads, out=rgrid_cube)

            # Contiguous 4-quadrant fftshift crop and gridding kernel correction
            dirty_cube[:, :hx, :hy] = rgrid_cube[:, nu - hx : nu, nv - hy : nv]
            dirty_cube[:, :hx, hy:] = rgrid_cube[:, nu - hx : nu, :ry]
            dirty_cube[:, hx:, :hy] = rgrid_cube[:, :rx, nv - hy : nv]
            dirty_cube[:, hx:, hy:] = rgrid_cube[:, :rx, :ry]
            dirty_cube *= cf_2d

            # Robust pulse profile SNR per pixel across phase bins (off-pulse MAD)
            p_med = np.median(dirty_cube, axis=0)  # (nx, ny)
            p_mad = np.median(np.abs(dirty_cube - p_med[np.newaxis, :, :]), axis=0) * 1.4826 + 1e-12
            p_max = np.max(dirty_cube, axis=0)
            snr_plane = (p_max - p_med) / p_mad  # (nx, ny)

            peak_pix = np.unravel_index(np.argmax(snr_plane), snr_plane.shape)
            max_p_snr = float(snr_plane[peak_pix])

            all_periods.append(float(p_val))
            all_snrs.append(max_p_snr)

            if max_p_snr >= snr_thresh:
                best_bin = int(np.argmax(dirty_cube[:, peak_pix[0], peak_pix[1]]))
                candidates.append({
                    "period": float(p_val),
                    "snr": max_p_snr,
                    "base_period": float(base_period),
                    "bins": bins,
                    "best_bin": best_bin,
                    "peak_pixel": peak_pix,
                    "image": dirty_cube[best_bin].copy(),
                    "snr_image": snr_plane.copy(),
                    "profile": dirty_cube[:, peak_pix[0], peak_pix[1]].copy(),
                })

    # Sort candidates by descending SNR
    candidates = sorted(candidates, key=lambda c: c["snr"], reverse=True)[:top_candidates]

    return {
        "candidates": candidates,
        "periods": np.asarray(all_periods),
        "snrs": np.asarray(all_snrs),
        "vis_timeseries": vis_ts,
    }


