import numpy as np


def generate_width_trials(nbins, ducy_max=0.20, wtsp=1.5):
    widths = []
    w = 1
    wmax = int(max(1, ducy_max * nbins))
    while w <= wmax:
        widths.append(w)
        w = int(max(w + 1, wtsp * w))
    return np.asarray(widths)

def downsampled_size(n, f):
    return int(np.floor(n / f))

def ceilshift(rows: int, cols: int, pmax: float) -> int:
    """
    Python port of the C++ ceilshift function.
    
    Parameters
    ----------
    rows : int
        Number of rows in the FFA folding matrix
    cols : int
        Number of bins (foldbins)
    pmax : float
        Maximum number of periods to evaluate (in bins)
    
    Returns
    -------
    int
        ceilshift value
    """
    return int(np.ceil(cols * (rows - 1.0) * (1.0 - cols / pmax)))


def plan_ffa(nsamp, tsamp, period_min, period_max, bins_min, bins_max):
    """
    Generate plan of base periods and downsampling factors for FFA search.
    
    Parameters
    ----------
    nsamp : int
        Number of time samples in original series
    tsamp : float
        Native sampling time (seconds)
    period_min, period_max : float
        Minimum and maximum trial period to search (seconds)
    bins_min, bins_max : int
        Minimum and maximum number of fold bins

    Returns
    -------
    plan : list of dict
        Each dict contains:
            - 'downsample_factor'
            - 'tau' (effective sample time)
            - 'bins'
            - 'base_period'
            - 'rows_eval' (rows used in FFA transform)
    """

    # Initial downsampling factor
    ds_ini = period_min / (tsamp * bins_min)

    # Geometric growth factor
    ds_geo = (bins_max + 1.0) / bins_min

    # Number of required downsampling cycles
    num_downsamplings = int(np.ceil(np.log(period_max / period_min) / np.log(ds_geo)))

    plan = []

    for ids in range(num_downsamplings):
        f = ds_ini * (ds_geo ** ids)      # current downsampling factor
        tau = f * tsamp                   # effective sample time
        n = downsampled_size(nsamp, f)     # downsampled input nsamp
        period_max_samples = period_max / tau

        bstart = bins_min
        bstop = min(bins_max, n, int(period_max_samples))

        for bins in range(bstart, bstop + 1):
            rows = n // bins
            if rows <= 1:
                continue

            period_ceil = min(period_max_samples, bins + 1.0)
            rows_eval = min(rows, ceilshift(rows, bins, period_ceil))

            base_period = tau * bins

            plan.append({
                "downsample_factor": f,
                "tau": tau,
                "bins": bins,
                "base_period": base_period,
                "rows_eval": rows_eval
            })

    return plan