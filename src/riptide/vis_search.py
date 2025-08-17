import numpy as np

### Local module imports
import riptide.libcpp as libcpp
from .ffautils import generate_width_trials
from .periodogram import Periodogram, UV_FFA
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
def vis_ffa_image_candidates(
    ffa_blocks,
    uv_indices,
    psf_image,
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
        Each block is a (phase, freq) folded visibility from FFA.
    uv_indices : np.ndarray, shape (M, 2), int
        Integer (u, v) indices for each block.
    psf_image : np.ndarray, shape (ny, nx), float
        Real-valued PSF image in image space.
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
    # Validate input types/shapes
    if not isinstance(ffa_blocks, list) or not all(isinstance(b, np.ndarray) for b in ffa_blocks):
        raise ValueError("ffa_blocks must be a list of numpy arrays (complex64)")
    uv_indices = np.asarray(uv_indices)
    if uv_indices.shape != (len(ffa_blocks), 2):
        raise ValueError("uv_indices must have shape (M, 2) matching ffa_blocks length")
    if psf_image.shape != (ny, nx):
        raise ValueError(f"psf_image must have shape ({ny}, {nx})")
    
    # Generate widths in bins
    nbins = ffa_blocks[0].shape[0]
    widths = generate_width_trials(nbins, ducy_max=ducy_max, wtsp=wtsp).astype(np.uint64)

    # Call C++ candidate search
    candidates = libcpp.image_ffa_candidates(
        ffa_blocks,
        uv_indices.astype(np.uint64, copy=False),
        psf_image.astype(np.float32, copy=False),
        widths,
        np.array(periods, dtype=np.float32),
        nx,
        ny,
        snr_thresh,
        max_candidates_width,
        max_candidates_all,
        mask_radius
    )   
        # py::dict d;
        # d["x"] = c.x;
        # d["y"] = c.y;
        # d["snr"] = c.snr;
        # d["width"] = c.width_bins;  // width in bins
        # d["period"]

    return candidates
