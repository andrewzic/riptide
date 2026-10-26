import os
import logging
import multiprocessing

from riptide import TimeSeries, plan_ffa, ffa_search, find_peaks, libffa
from riptide import VisTimeSeries, vis_ffa_search, vis_ffa_search_basep, vis_ffa_image_candidates, test_vis_ffa_image, test_vis_ffa_image_imgs

from astropy.wcs import WCS
from astropy.io import fits

import pandas as pd
import numpy as np
import math

from itertools import product
from dask import delayed, compute
import zarr

from tqdm import tqdm

log = logging.getLogger("riptide.worker_pool")


class WorkerPool(object):
    """
    deredden_params : dict
    range_confs : list of dicts
        List of dicts from the 'ranges' section of the YAML config file
    loader : func
        Function that takes a file path as its only argument, and returns
        a TimeSeries object
    processes : int
        Number of parallel processes
    fmt : str
        TimeSeries file format
    """

    TIMESERIES_LOADERS = {
        "sigproc": TimeSeries.from_sigproc,
        "presto": TimeSeries.from_presto_inf,
    }

    def __init__(self, deredden_params, range_confs, processes=1, fmt="presto"):
        self.deredden_params = deredden_params
        self.range_confs = range_confs
        self.loader = self.TIMESERIES_LOADERS[fmt]
        self.processes = int(processes)

    def process_fname_list(self, fnames):
        pool = multiprocessing.Pool(processes=self.processes)
        # results is a list of lists of Detections
        results = pool.map(self.process_fname, fnames)
        # NOTE: don't forget to close the pool to free up RAM
        # NOTE: and don't forget to join, otherwise the coverage module
        # does not properly report coverage for sub-processes spawned by
        # the pool
        pool.close()
        pool.join()
        return [det for dlist in results for det in dlist]

    def process_fname(self, fname):
        allpeaks = []
        ts = self.loader(fname)
        dm = ts.metadata["dm"]
        log.debug("Searching DM = {:.3f}".format(dm))

        # Make pre-processing common to all ranges to save time
        ts = ts.deredden(
            self.deredden_params["rmed_width"],
            minpts=self.deredden_params["rmed_minpts"],
        )
        ts = ts.normalise()

        for conf in self.range_confs:
            kw_search = dict(conf["ffa_search"])
            kw_search.update({"deredden": False, "already_normalised": True})
            tsdr, pgram = ffa_search(ts, **kw_search)
            peaks, polycos = find_peaks(pgram, **conf["find_peaks"])
            allpeaks.extend(peaks)
            del tsdr, pgram, peaks, polycos  # Free RAM ASAP
        log.debug(f"Done searching DM = {dm:.3f}, peaks found: {len(allpeaks)}")
        return allpeaks
    

def _uv_slices(nu, nv, u_chunks, v_chunks):
    u_size = math.ceil(nu / u_chunks)
    v_size = math.ceil(nv / v_chunks)
    u_slices = [slice(i, min(i + u_size, nu)) for i in range(0, nu, u_size)]
    v_slices = [slice(j, min(j + v_size, nv)) for j in range(0, nv, v_size)]
    return u_slices, v_slices

def _paths_for(plan_idx, u_sl, v_sl, outdir):
    tag = f"p{plan_idx}_u{u_sl.start}-{u_sl.stop}_v{v_sl.start}-{v_sl.stop}"
    return (os.path.join(outdir, f"ffa_{tag}.npy"),
            os.path.join(outdir, f"uv_{tag}.npy"))

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class VisWorkerPool(object):
    """
    deredden_params : dict
    range_confs : list of dicts
        List of dicts from the 'ranges' section of the YAML config file
    loader : func
        Function that takes a file path as its only argument, and returns
        a TimeSeries object
    processes : int
        Number of parallel processes
    fmt : str
        TimeSeries file format
    """

    TIMESERIES_LOADERS = {
        "img": VisTimeSeries.from_img_cube,
        "uv": VisTimeSeries.from_real_imag_cube
    }

    def __init__(self, range_confs, psf_img_file=None, processes=1, fmt="uv"):
        self.range_confs = range_confs
        self.loader = self.TIMESERIES_LOADERS[fmt]
        self.processes = int(processes)
        self.psf_file = psf_img_file
        #self.load_psf_img()

    def load_psf_img(self, psf_img_file=None):

        if psf_img_file is not None:
            self.psf_file = psf_img_file

        #we load in one psf image (usually just the one made for the continuum map) 
        # to approximate as the point source matched filter for folded
        psf = fits.getdata(self.psf_file).squeeze()

        #now get the phase centre
        header = fits.getheader(self.psf_file)
        ny, nx = psf.shape
        self.nx = nx
        self.ny = ny
        wcs = WCS(header)
        wcs = wcs.dropaxis(3).dropaxis(2)
        world_coords = wcs.pixel_to_world((nx - 1)/2.0, (ny - 1)/2.0)
        self.ra_deg = world_coords.ra.deg
        self.dec_deg = world_coords.dec.deg
        self.psf = psf
        self.psf_header = header


    def process_fname_list(self, fnames):
        pool = multiprocessing.Pool(processes=self.processes)
        # results is a list of lists of Detections
        results = pool.map(self.process_fname, fnames)
        # NOTE: don't forget to close the pool to free up RAM
        # NOTE: and don't forget to join, otherwise the coverage module
        # does not properly report coverage for sub-processes spawned by
        # the pool
        pool.close()
        pool.join()
        return [det for dlist in results for det in dlist]
    
    def process_fname(self, fname):
        allpeaks = []
        ts = self.loader(fname)
        dm = ts.metadata["dm"]
        log.debug("Searching DM = {:.3f}".format(dm))

        # Make pre-processing common to all ranges to save time
        ts = ts.deredden(
            self.deredden_params["rmed_width"],
            minpts=self.deredden_params["rmed_minpts"],
        )
        ts = ts.normalise()

        for conf in self.range_confs:
            kw_search = dict(conf["ffa_search"])
            kw_search.update({"deredden": False, "already_normalised": True})
            tsdr, pgram = ffa_search(ts, **kw_search)
            peaks, polycos = find_peaks(pgram, **conf["find_peaks"])
            allpeaks.extend(peaks)
            del tsdr, pgram, peaks, polycos  # Free RAM ASAP
        log.debug(f"Done searching DM = {dm:.3f}, peaks found: {len(allpeaks)}")
        return allpeaks
    
    def process_uvcells(self, real_fname, imag_fname):
        log.debug(f"Processing UV cells from {real_fname}")
        all_candidates = []
        vis_ts = self.loader(real_fname, imag_fname)
        vis_ts.get_sparse_unique_uv()
        
        nx = vis_ts.nu if vis_ts.nu is not None else vis_ts.header["NAXIS1"]
        ny = vis_ts.nv if vis_ts.nv is not None else vis_ts.header["NAXIS2"]
        skycoords = vis_ts.sky_coords if hasattr(vis_ts, "sky_coords") and vis_ts.sky_coords is not None else None

        for conf in self.range_confs:
            kw_search = dict(conf["ffa_search"])
            kw_search.update({"deredden": False, "already_normalised": True})

            # Step 1: run vis_ffa_search for each UV cell
            uv_ffa_list = []
            for uvcell in tqdm(vis_ts.unique_uv):
                ts = vis_ts.index(uvcell)
                ts, uv_ffa = vis_ffa_search(ts, uv_ind=uvcell, **kw_search)
                uv_ffa_list.append(uv_ffa)

            if not uv_ffa_list:
                continue
            base_periods = uv_ffa_list[0].base_periods
            tsamps = uv_ffa_list[0].tsamps
            num_trials = len(uv_ffa_list[0].ffa_list)

            # Step 3: for each trial period index, gather all uvcell blocks and run vis_ffa_image_candidates
            uv_indices = [uv_ffa.uv_ind for uv_ffa in uv_ffa_list]
            for trial_idx in tqdm(range(num_trials)):
                block_periods = uv_ffa_list[0].periods[trial_idx]
                base_period = base_periods[trial_idx]
                tsamp = tsamps[trial_idx]
                ffa_blocks = [uv_ffa.ffa_list[trial_idx] for uv_ffa in uv_ffa_list]

                trial_candidates = vis_ffa_image_candidates(
                    ffa_blocks,
                    uv_indices,
                    block_periods,
                    nx,
                    ny,
                    snr_thresh=8.0,
                    max_candidates_width=500,
                    max_candidates_all=10000,
                    ducy_max=0.95,
                    wtsp=1.5,
                    mask_radius=2
                )
                log.debug(f"Found {len(trial_candidates)} candidates in trial {trial_idx}")
                for cand in trial_candidates:
                    skycoord = skycoords[cand["y"], cand["x"]] if skycoords is not None else None
                    all_candidates.append({
                        "trial_idx": trial_idx,
                        "base_period": base_period,
                        "tsamp": tsamp,
                        "x_pix": cand["x"],
                        "y_pix": cand["y"],
                        "snr": cand["snr"],
                        "period": cand["period"],
                        "width_bins": cand["width"],
                        "ra_deg": skycoord.ra.deg if skycoord is not None else None,
                        "dec_deg": skycoord.dec.deg if skycoord is not None else None,
                        "skycoord": skycoord
                    })
        df_candidates = pd.DataFrame(all_candidates)
        self.candidates = df_candidates
        return df_candidates
    
    def process_uvcells_basep(self, real_fname, imag_fname):
        log.debug(f"Processing UV cells (basep) from {real_fname}")
        all_candidates = []
        
        vis_ts = self.loader(real_fname, imag_fname)
        log.debug("Loaded visibility data cube")

        #get list of unique u, v pixel cell coords (for all nonzero u,v pixels)
        vis_ts.get_sparse_unique_uv()
        
        nx = vis_ts.nu if vis_ts.nu is not None else vis_ts.header["NAXIS1"]
        ny = vis_ts.nv if vis_ts.nv is not None else vis_ts.header["NAXIS2"]

        #get and set other handy attributes
        nsamp = vis_ts.nsamp
        skycoords = vis_ts.sky_coords if hasattr(vis_ts, 'sky_coords') and vis_ts.sky_coords is not None else None
        tsamp = vis_ts.tsamp

        #perform one-off computation to densify the sparse cube
        #this deletes vis_ts.data
        vis_ts.make_dynamic_grid_array()
        log.debug(f"Constructed DGA (memory usage: {vis_ts.dga.nbytes} bytes)")
        dense_uv_ts = vis_ts.dga
        
        trial_idx_ctr = 0

        #range_confs is the period search parameter ranges defined in the config file
        for conf in self.range_confs:
            kw_search = dict(conf["ffa_search"])
            kw_search.update({"deredden": False, "already_normalised": True})

            period_min = kw_search["period_min"]
            period_max = kw_search["period_max"]
            bins_min = kw_search["bins_min"]
            bins_max = kw_search["bins_max"]  
            wtsp = kw_search["wtsp"]
            
            ffa_plans = plan_ffa(nsamp, tsamp, period_min, period_max, bins_min, bins_max)      
            
            for ffa_plan in ffa_plans:
                downsample_fac = ffa_plan["downsample_factor"]
                tau = ffa_plan["tau"]
                base_period = ffa_plan["base_period"]
                bins = ffa_plan["bins"]
                rows_eval = ffa_plan["rows_eval"]

                log.debug(f"Running FFA transform on {vis_ts.unique_uv.shape} cells with base period {bins} bins ({base_period} s)")
                uv_ffa = vis_ffa_search_basep(
                    dense_uv_ts,
                    bins,
                    vis_ts.unique_uv,
                    tau,
                    rows_eval=rows_eval,
                    downsample_factor=downsample_fac
                )

                # gather up results
                ffa_cube = uv_ffa.ffa_array
                block_periods = uv_ffa.periods
                base_period = uv_ffa.base_periods
                tsamp_eff = uv_ffa.tsamps

                # now image each period, phase trial in the output (u,v,period,phase) cube
                trial_candidates = vis_ffa_image_candidates(
                    ffa_cube,
                    vis_ts.unique_uv,
                    block_periods,
                    nx,
                    ny,
                    snr_thresh=8.0,
                    max_candidates_width=500,
                    max_candidates_all=10000,
                    ducy_max=0.5,
                    wtsp=wtsp,
                    mask_radius=2
                )
                log.debug(f"Found {len(trial_candidates)} candidates") 

                # gather candidates and add info to dataframe dict
                for cand in trial_candidates:
                    skycoord = skycoords[cand["y"], cand["x"]] if skycoords is not None else None
                    all_candidates.append({
                        "trial_idx": trial_idx_ctr,
                        "base_period": base_period,
                        "tsamp": tsamp_eff,
                        "x_pix": cand["x"],
                        "y_pix": cand["y"],
                        "snr": cand["snr"],
                        "period": cand["period"],
                        "width_bins": cand["width"],
                        "cutout": cand["cutout"],
                        "ra_deg": skycoord.ra.deg if skycoord is not None else None,
                        "dec_deg": skycoord.dec.deg if skycoord is not None else None,
                        "skycoord": skycoord
                    })
                    trial_idx_ctr += 1
        #construct dataframe
        df_candidates = pd.DataFrame(all_candidates)
        self.candidates = df_candidates
        return df_candidates



    def process_uvcells_basep_dask(self, real_fname, imag_fname,
                                u_chunks=4, v_chunks=4,
                                threshold=3e-3,
                                periods_per_block=256,
                                outdir="ffa_chunks"):
        """
        Dask workflow with Zarr storage:
        1) (u,v)-chunked load of time-cube -> per-chunk FFA (saved into zarr).
        2) Period-block imaging across ALL (u,v) by reading zarr datasets lazily.
        """

        _ensure_dir(outdir)

        # -------------------------------------------------------------
        # Global geometry + header info (no big data read)
        # -------------------------------------------------------------
        with fits.open(real_fname, memmap=True) as hdul:
            header = hdul[0].header
            nu = header["NAXIS1"]
            nv = header["NAXIS2"]
        nx, ny = nu, nv

        tmp_vis = VisTimeSeries.from_real_imag_cube_chunked(
            real_fname, imag_fname, slice(0, min(1, nv)), slice(0, min(1, nu)), threshold=threshold
        )
        tsamp = tmp_vis.tsamp
        nsamp = tmp_vis.nsamp

        # -------------------------------------------------------------
        # Build (u,v) chunk grid
        # -------------------------------------------------------------
        u_slices, v_slices = _uv_slices(nv, nu, u_chunks, v_chunks)

        # -------------------------------------------------------------
        # Stage 1: per-(u,v)-chunk delayed FFA computation to Zarr
        # Load the (u, v) chunk ONCE from FITS and reuse across all plans!
        # -------------------------------------------------------------
        plan_specs = []
        for conf in self.range_confs:
            kw_search = dict(conf["ffa_search"])
            plans = plan_ffa(nsamp, tsamp, kw_search["period_min"], kw_search["period_max"],
                             kw_search["bins_min"], kw_search["bins_max"])
            for p in plans:
                p_copy = dict(p)
                p_copy["wtsp"] = kw_search.get("wtsp", 1.5)
                p_copy["snr_min"] = self.config.get("candidate_filters", {}).get("snr_min", 8.0) if hasattr(self, "config") and isinstance(self.config, dict) else 8.0
                plan_specs.append(p_copy)

        @delayed
        def compute_ffa_for_uv_chunk(u_sl, v_sl, plan_specs_list):
            vis_ts = VisTimeSeries.from_real_imag_cube_chunked(
                real_fname, imag_fname, u_slice=u_sl, v_slice=v_sl, threshold=threshold
            )
            vis_ts.get_sparse_unique_uv()
            if vis_ts.unique_uv is None or len(vis_ts.unique_uv) == 0:
                return []

            vis_ts.make_dynamic_grid_array()
            dense_uv_ts = vis_ts.dga

            u0, v0 = u_sl.start, v_sl.start
            local_uv = vis_ts.unique_uv.astype(np.uint64, copy=False)
            uv_indices_global = local_uv.copy()
            uv_indices_global[:, 0] += np.uint64(u0)
            uv_indices_global[:, 1] += np.uint64(v0)

            root = zarr.open_group(outdir, mode="a")
            chunk_metas = []

            for plan_idx, p in enumerate(plan_specs_list):
                bins = p["bins"]
                tau = p["tau"]
                rows_eval = p["rows_eval"]
                downsample_fac = p["downsample_factor"]

                uv_ffa = vis_ffa_search_basep(
                    dense_uv_ts,
                    bins,
                    vis_ts.unique_uv,
                    tau,
                    rows_eval=rows_eval,
                    downsample_factor=downsample_fac
                )
                ffa_cube = uv_ffa.ffa_array

                chunk_key = f"plan_{plan_idx}_u{u_sl.start}_{u_sl.stop}_v{v_sl.start}_{v_sl.stop}"
                grp = root.create_group(chunk_key, overwrite=True)
                grp.create_dataset("ffa", data=ffa_cube, chunks=(ffa_cube.shape[0], min(periods_per_block, ffa_cube.shape[1]), ffa_cube.shape[2]))
                grp.create_dataset("uv", data=uv_indices_global, chunks=(uv_indices_global.shape[0], 2))

                chunk_metas.append({
                    "plan_idx": plan_idx,
                    "chunk_key": chunk_key,
                    "zarr_dir": outdir,
                    "nperiod": ffa_cube.shape[1],
                    "nphase": ffa_cube.shape[2],
                    "m": ffa_cube.shape[0],
                    "periods": uv_ffa.periods
                })

            return chunk_metas

        ffa_tasks = [
            compute_ffa_for_uv_chunk(u_sl, v_sl, plan_specs)
            for u_sl, v_sl in product(u_slices, v_slices)
        ]

        ffa_metas_nested = compute(*ffa_tasks)

        by_plan = {}
        for chunk_list in ffa_metas_nested:
            for m in chunk_list:
                if m["chunk_key"] is None:
                    continue
                by_plan.setdefault(m["plan_idx"], []).append(m)

        # -------------------------------------------------------------
        # Stage 2: per-period-block delayed imaging across all (u,v)
        # -------------------------------------------------------------
        @delayed
        def image_period_block(plan_idx, p_start, p_stop, plan_bins, plan_tau):
            metas = by_plan.get(plan_idx, [])
            if not metas:
                return []

            ffa_store = zarr.open_group(outdir, mode="r")
            ffa_slices = []
            uv_slices = []
            m_total = 0

            for m in metas:
                arr = ffa_store[f"{m['chunk_key']}/ffa"]
                sub = arr[:, p_start:p_stop, :]
                ffa_slices.append(np.asarray(sub))
                uv = ffa_store[f"{m['chunk_key']}/uv"][:]
                uv_slices.append(uv)
                m_total += uv.shape[0]

            if m_total == 0:
                return []

            uv_ffa_cube = np.concatenate(ffa_slices, axis=0)
            uv_indices = np.concatenate(uv_slices, axis=0)

            periods_full = metas[0]["periods"]
            periods_block = periods_full[p_start:p_stop]

            cands = vis_ffa_image_candidates(
                uv_ffa_cube, uv_indices.astype(np.uint64, copy=False),
                periods_block.astype(np.float32, copy=False),
                nx, ny,
                snr_thresh=8.0, max_candidates_width=500, max_candidates_all=10000,
                ducy_max=0.5, wtsp=1.5, mask_radius=2
            )
            for d in cands:
                d["plan_idx"] = plan_idx
                d["p_start"] = int(p_start)
                d["p_stop"] = int(p_stop)
            return cands

        imaging_tasks = []
        for plan_idx, plan in enumerate(plan_specs):
            metas = by_plan.get(plan_idx, [])
            if not metas:
                continue
            nperiod = max(m["nperiod"] for m in metas) if metas else 0
            if nperiod == 0:
                continue
            bins, tau = plan["bins"], plan["tau"]
            for p_start in range(0, nperiod, periods_per_block):
                p_stop = min(p_start + periods_per_block, nperiod)
                imaging_tasks.append(image_period_block(plan_idx, p_start, p_stop, bins, tau))

        block_results = compute(*imaging_tasks)

        # -------------------------------------------------------------
        # Assemble final dataframe and RA/Dec
        # -------------------------------------------------------------
        all_candidates = []
        for block_cands in block_results:
            if not block_cands:
                continue
            for cand in block_cands:
                x, y = int(cand["x"]), int(cand["y"])
                all_candidates.append({
                    "plan_idx": cand["plan_idx"],
                    "p_start": cand["p_start"],
                    "p_stop": cand["p_stop"],
                    "x_pix": x,
                    "y_pix": y,
                    "snr": cand["snr"],
                    "period": cand["period"],
                    "width_bins": cand["width"],
                    "cutout": cand["cutout"]
                })

        df = pd.DataFrame(all_candidates)
        self.candidates = df
        return df
        
    def process_uvcells_test(self, real_fname, imag_fname):
        log.debug(f"Processing UV cells (test) from {real_fname}")
        all_candidates = []
        vis_ts = self.loader(real_fname, imag_fname)
        vis_ts.get_sparse_unique_uv()
        
        nx = vis_ts.nu if vis_ts.nu is not None else vis_ts.header["NAXIS1"]
        ny = vis_ts.nv if vis_ts.nv is not None else vis_ts.header["NAXIS2"]
        nsamp = vis_ts.nsamp
        tsamp = vis_ts.tsamp
        vis_ts.make_dynamic_grid_array()
        dense_uv_ts = vis_ts.dga

        for conf in self.range_confs:
            kw_search = dict(conf["ffa_search"])
            kw_search.update({"deredden": False, "already_normalised": True})

            period_min = kw_search["period_min"]
            period_max = kw_search["period_max"]
            bins_min = kw_search["bins_min"]
            bins_max = kw_search["bins_max"]  
            
            ffa_plans = plan_ffa(nsamp, tsamp, period_min, period_max, bins_min, bins_max)      

            for ffa_plan in ffa_plans:
                downsample_fac = ffa_plan["downsample_factor"]
                tau = ffa_plan["tau"]
                base_period = ffa_plan["base_period"]
                bins = ffa_plan["bins"]
                rows_eval = ffa_plan["rows_eval"]

                log.debug(f"Running FFA transform on {vis_ts.unique_uv.shape} cells with base period {base_period}")
                uv_ffa = vis_ffa_search_basep(
                    dense_uv_ts,
                    bins,
                    vis_ts.unique_uv,
                    tau,
                    rows_eval=rows_eval,
                    downsample_factor=downsample_fac
                )

                ffa_cube = uv_ffa.ffa_array
                periods = uv_ffa.periods
                
                img_dict = test_vis_ffa_image_imgs(
                    ffa_cube,
                    vis_ts.unique_uv,
                    periods,
                    nx,
                    ny
                )

                all_candidates.append(img_dict)

        return all_candidates
