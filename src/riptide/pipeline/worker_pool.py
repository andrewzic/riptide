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
        print("here in process_uvcells", real_fname)
        all_candidates = []
        vis_ts = self.loader(real_fname, imag_fname) #this should be generalised
        #get list of unique u, v pixel cell coords (indices)
        vis_ts.get_sparse_unique_uv()
        
        #this will set self.ra_deg, self.dec_deg, and self.psf
        self.load_psf_img(psf_img_file=self.psf_file)
        #now set the phase centre attribute of the visibility timeseries
        vis_ts.set_phase_centre(self.ra_deg, self.dec_deg)
        #now set the vis_ts.sky_coords grid by calling get_skycoords_from_psf_header
        vis_ts.get_skycoords_from_psf_header(vis_ts.header)
        
        for conf in self.range_confs:
            kw_search = dict(conf["ffa_search"])
            kw_search.update({"deredden": False, "already_normalised": True})

            # Step 1: run vis_ffa_search for each UV cell
            uv_ffa_list = []
            for uvcell in tqdm(vis_ts.unique_uv):
                ts = vis_ts.index(uvcell)
                ts, uv_ffa = vis_ffa_search(ts, uv_ind=uvcell, **kw_search)
                uv_ffa_list.append(uv_ffa)

            # periods = uv_ffa_list[0].periods #sum(nrow) for nrow in ffa_blocks) elements
            # foldbins = uv_ffa_list[0].foldbins #sum(nrow) for nrow in ffa_blocks) elements
            base_periods = uv_ffa_list[0].base_periods  #len(ffa_blocks) elements
            tsamps = uv_ffa_list[0].tsamps #len(ffa_blocks) elements
            nx = self.nx
            # Step 2: figure out how many trial periods we have
            if not uv_ffa_list:
                continue
            num_trials = len(uv_ffa_list[0].ffa_list)

            # Step 3: for each trial period index, gather all uvcell blocks and run vis_ffa_image_candidates
            uv_indices = [uv_ffa.uv_ind for uv_ffa in uv_ffa_list]
            psf_img = self.psf
            nx, ny = (self.nx, self.ny)
            for trial_idx in tqdm(range(num_trials)):
                block_periods = uv_ffa_list[0].periods[trial_idx]
                block_foldbins = uv_ffa_list[0].foldbins[trial_idx]

                base_period = base_periods[trial_idx]
                tsamp = tsamps[trial_idx]
                ffa_blocks = [uv_ffa.ffa_list[trial_idx] for uv_ffa in uv_ffa_list]

                trial_candidates = vis_ffa_image_candidates(
                    ffa_blocks,
                    uv_indices,
                    psf_img,
                    block_periods,
                    nx,
                    ny,
                    snr_thresh=8.0,
                    max_candidates_width=500,
                    max_candidates_all=10000,
                    ducy_max=0.95,
                    wtsp=1.5, #width spacing
                    mask_radius=2
                )
                print(f"found {len(trial_candidates)} cands") 
                #conf["candidate_filters"]["snr_min"],                
                #conf["candidate_filters"]["max_candidates"],
                #conf["candidate_filters"]["max_candidates_all"],
                for cand in trial_candidates:
                    # cand.x, cand.y are image pixel coords
                    skycoord = sky_coords[cand.y, cand.x]  # lookup
                    all_candidates.append({
                        "trial_idx": trial_idx,
                        "base_period": base_period,
                        "tsamp": tsamp,
                        "x_pix": cand["x"],
                        "y_pix": cand["y"],
                        "snr": cand["snr"],
                        "period": cand["period"],
                        "width_bins": cand["width"],
                        "ra_deg": skycoord.ra.deg,
                        "dec_deg": skycoord.dec.deg,
                        "skycoord": skycoord
                    })
        df_candidates = pd.DataFrame(all_candidates)
        self.candidates = df_candidates
        return df_candidates
    
    def process_uvcells_basep(self, real_fname, imag_fname):
        print("here in process_uvcells", real_fname)
        all_candidates = []
        
        #load in the data all in one blob
        # this is horrible on i/o and RAM
        vis_ts = self.loader(real_fname, imag_fname) #this should be generalised
        #vis_ts should be a sparse.COO cube
        print("loaded data cube")

        #get list of unique u, v pixel cell coords (for all nonzero u,v pixels)
        vis_ts.get_sparse_unique_uv()
        
        #now set self.ra_deg, self.dec_deg, and self.psf
        self.load_psf_img(psf_img_file=self.psf_file)
        print("loaded psf image")

        #now set the phase centre attribute of the visibility timeseries
        vis_ts.set_phase_centre(self.ra_deg, self.dec_deg)

        #now set the vis_ts.sky_coords grid by calling get_skycoords_from_psf_header
        vis_ts.get_skycoords_from_psf_header(vis_ts.header)

        #get and set other handy attributes
        nsamp = vis_ts.nsamp
        skycoords = vis_ts.sky_coords
        tsamp = vis_ts.tsamp

        #perform one-off computation to densify the sparse cube
        #this deletes vis_ts.data
        vis_ts.make_dynamic_grid_array()
        # the result is a 2D numpy array for every nonzero (u,v) pixel in the 3D cube
        # each row of the 2D array has a filled time series of size nsamp
        print("made DGA")
        dense_uv_ts = vis_ts.dga
        
        print(f"DGA memory usage: {vis_ts.dga.nbytes}")
        
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
            
            #plan out the FFA - base periods, etc given the input period min, max, bins min, max, tsamp, etc
            ffa_plans = plan_ffa(nsamp, tsamp, period_min, period_max, bins_min, bins_max)      
            # ffa_plan : list of dict
            # Each dict contains:
            #     - 'downsample_factor'
            #     - 'tau' (effective sample time)
            #     - 'bins'
            #     - 'base_period' (tau*bins)
            #     - 'rows_eval' (rows used in FFA transform)    
            
            for ffa_plan in ffa_plans:
                downsample_fac = ffa_plan["downsample_factor"]
                tau = ffa_plan["tau"]
                base_period = ffa_plan["base_period"] #need to make sure this is int bins, not real base period [s]
                print(base_period)
                bins = ffa_plan["bins"] #this is what we should pass downstream for base folding period... I think
                rows_eval = ffa_plan["rows_eval"]

                # MAJOR Step 1: run vis_ffa_search for each UV cell
                print(f"doing FFA transform on {vis_ts.unique_uv.shape} cells with base period {bins}, corresponding to {base_period*tau} s. ")
                uv_ffa = vis_ffa_search_basep(dense_uv_ts, bins, vis_ts.unique_uv, tau)
                # uv_ffa_list.append(uv_ffa)

                # gather up results
                ffa_cube = uv_ffa.ffa_array
                psf_img = self.psf
                nx, ny = (self.nx, self.ny)
                block_periods = uv_ffa.periods
                block_foldbins = uv_ffa.foldbins

                base_period = uv_ffa.base_period
                tsamp = uv_ffa.tsamp

                # now image each period, phase trial in the output (u,v,period,phase) cube
                trial_candidates = vis_ffa_image_candidates(
                    ffa_cube,
                    vis_ts.unique_uv,
                    psf_img,
                    block_periods,
                    nx,
                    ny,
                    snr_thresh=8.0,
                    max_candidates_width=500,
                    max_candidates_all=10000,
                    ducy_max=0.5,
                    wtsp=1.5, #width spacing
                    mask_radius=2
                )
                print(f"found {len(trial_candidates)} cands") 

                # gather candidates and add info to dataframe dict
                for cand in trial_candidates:
                    # cand.x, cand.y are image pixel coords
                    skycoord = skycoords[cand["y"], cand["x"]]  # lookup coord
                    all_candidates.append({
                        "trial_idx": trial_idx_ctr,
                        "base_period": base_period,
                        "tsamp": tsamp,
                        "x_pix": cand["x"],
                        "y_pix": cand["y"],
                        "snr": cand["snr"],
                        "period": cand["period"],
                        "width_bins": cand["width"],
                        "cutout": cand["cutout"],
                        "ra_deg": skycoord.ra.deg,
                        "dec_deg": skycoord.dec.deg,
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
        self.load_psf_img(psf_img_file=self.psf_file)
        psf_img = self.psf
        nx, ny = self.nx, self.ny

        tmp_vis = VisTimeSeries.from_real_imag_cube_chunked(
            real_fname, imag_fname, slice(0, min(1, nu)), slice(0, min(1, nv)), threshold=threshold
        )
        tsamp = tmp_vis.tsamp
        nsamp = tmp_vis.nsamp
        tmp_vis.set_phase_centre(self.ra_deg, self.dec_deg)
        skycoords = tmp_vis.get_skycoords_from_psf_header(tmp_vis.header)

        # -------------------------------------------------------------
        # Build (u,v) chunk grid
        # -------------------------------------------------------------
        u_slices, v_slices = _uv_slices(nu, nv, u_chunks, v_chunks)

        # -------------------------------------------------------------
        # Stage 1: per-(u,v)-chunk delayed FFA computation to Zarr
        # -------------------------------------------------------------
        @delayed
        def compute_ffa_for_chunk(plan_idx, u_sl, v_sl, bins, tau):
            vis_ts = VisTimeSeries.from_real_imag_cube_chunked(
                real_fname, imag_fname, u_slice=u_sl, v_slice=v_sl, threshold=threshold
            )
            vis_ts.get_sparse_unique_uv()
            if vis_ts.unique_uv is None or len(vis_ts.unique_uv) == 0:
                return {"plan_idx": plan_idx, "chunk_key": None, "zarr_dir": None,
                        "nperiod": 0, "nphase": 0, "m": 0}

            vis_ts.make_dynamic_grid_array()
            dense_uv_ts = vis_ts.dga

            uv_ffa = vis_ffa_search_basep(dense_uv_ts, bins, vis_ts.unique_uv, tau)
            ffa_cube = uv_ffa.ffa_array

            u0, v0 = u_sl.start, v_sl.start
            local_uv = vis_ts.unique_uv.astype(np.uint64, copy=False)
            uv_indices_global = local_uv.copy()
            uv_indices_global[:, 0] += np.uint64(u0)
            uv_indices_global[:, 1] += np.uint64(v0)

            # save to zarr
            zarr_dir = os.path.join(outdir, f"plan{plan_idx}")
            ffa_store = zarr.open_group(zarr_dir, mode="a")
            chunk_key = f"u{u_sl.start}_{u_sl.stop}_v{v_sl.start}_{v_sl.stop}"

            ffa_dset = ffa_store.require_dataset(
                name=f"{chunk_key}/ffa",
                shape=ffa_cube.shape,
                chunks=(ffa_cube.shape[0], min(256, ffa_cube.shape[1]), ffa_cube.shape[2]),
                dtype="complex64",
                overwrite=True
            )
            ffa_dset[:] = ffa_cube.astype(np.complex64, copy=False)

            uv_dset = ffa_store.require_dataset(
                name=f"{chunk_key}/uv",
                shape=uv_indices_global.shape,
                chunks=uv_indices_global.shape,
                dtype="uint64",
                overwrite=True
            )
            uv_dset[:] = uv_indices_global

            return {"plan_idx": plan_idx, "chunk_key": chunk_key, "zarr_dir": zarr_dir,
                    "nperiod": ffa_cube.shape[1], "nphase": ffa_cube.shape[2], "m": ffa_cube.shape[0]}

        # -------------------------------------------------------------
        # Build plans from config
        # -------------------------------------------------------------
        plan_specs = []
        for conf in self.range_confs:
            kw = dict(conf["ffa_search"])
            kw.update({"deredden": False, "already_normalised": True})
            ffa_plans = plan_ffa(nsamp, tsamp,
                                kw["period_min"], kw["period_max"],
                                kw["bins_min"], kw["bins_max"])
            plan_specs.extend(ffa_plans)

        all_plan_chunk_tasks = []
        for plan_idx, plan in enumerate(plan_specs):
            bins, tau = plan["bins"], plan["tau"]
            for u_sl, v_sl in product(u_slices, v_slices):
                all_plan_chunk_tasks.append(compute_ffa_for_chunk(plan_idx, u_sl, v_sl, bins, tau))

        chunk_meta_list = list(compute(*all_plan_chunk_tasks))

        # -------------------------------------------------------------
        # Stage 2: period-block imaging across ALL (u,v)
        # -------------------------------------------------------------
        by_plan = {}
        for meta in chunk_meta_list:
            by_plan.setdefault(meta["plan_idx"], []).append(meta)

        @delayed
        def image_period_block(plan_idx, p_start, p_stop, plan_bins, plan_tau):
            metas = by_plan[plan_idx]
            ffa_slices, uv_slices, m_total = [], [], 0

            for m in metas:
                if m["nperiod"] == 0 or m["m"] == 0:
                    continue
                ffa_store = zarr.open_group(m["zarr_dir"], mode="r")
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

            periods_full = libffa.ffaprd(N=nsamp, p=int(plan_bins), dt=plan_tau)
            periods_block = periods_full[p_start:p_stop]

            cands = vis_ffa_image_candidates(
                uv_ffa_cube, uv_indices.astype(np.uint64, copy=False),
                psf_img.astype(np.float32, copy=False), periods_block.astype(np.float32, copy=False),
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
                sc = skycoords[y, x]
                all_candidates.append({
                    "plan_idx": cand["plan_idx"],
                    "p_start": cand["p_start"],
                    "p_stop": cand["p_stop"],
                    "x_pix": x,
                    "y_pix": y,
                    "snr": float(cand["snr"]),
                    "period": float(cand["period"]),
                    "width_bins": int(cand["width"]),
                    "cutout": cand["cutout"],
                    "ra_deg": sc.ra.deg,
                    "dec_deg": sc.dec.deg,
                    "skycoord": sc
                })

        df = pd.DataFrame(all_candidates)
        self.candidates = df
        return df
        
    def process_uvcells_test(self, real_fname, imag_fname):
        print("here in process_uvcells", real_fname)
        all_candidates = []
        vis_ts = self.loader(real_fname, imag_fname) #this should be generalised
        #get list of unique u, v pixel cell coords (indices)
        vis_ts.get_sparse_unique_uv()
        

        #this will set self.ra_deg, self.dec_deg, and self.psf
        self.load_psf_img(psf_img_file=self.psf_file)
        #now set the phase centre attribute of the visibility timeseries
        vis_ts.set_phase_centre(self.ra_deg, self.dec_deg)
        #now set the vis_ts.sky_coords grid by calling get_skycoords_from_psf_header
        vis_ts.get_skycoords_from_psf_header(vis_ts.header)
        nsamp = vis_ts.nsamp
        skycoords = vis_ts.sky_coords
        tsamp = vis_ts.tsamp
        print(vis_ts.data.shape, "vis ts shape")
        #one-off computation to densify the sparse cube
        vis_ts.make_dynamic_grid_array()
        # the result is a 2D numpy array for every nonzero (u,v) pixel in the 3D cube
        # each row of the 2D array has a filled time series of size nsamp
        print("made DGA")
        dense_uv_ts = vis_ts.dga
        # uv_mask = vis_ts.data.max(axis=0) != 0  # shape (U, V)
        # print("UV mask shape:", uv_mask.shape)
        # u_coords, v_coords = uv_mask.coords  # 1D arrays of active indices
        # dense_uv_ts = vis_ts.data[:, u_coords, v_coords].todense()
        # dense_uv_ts = np.ascontiguousarray(np.transpose(dense_uv_ts)) #cast to uv, time
        print("dga shape:", dense_uv_ts.shape)
        # print("len uvcoords:", len(u_coords), uv_mask.shape)
        # grid_ = np.zeros((self.ny, self.nx))
        # for u, v in zip(u_coords, v_coords):
        #     #print(u,v)
        #     grid_[v, u] = 1.0
        # import matplotlib.pyplot as plt
        # plt.imshow(grid_)
        # plt.show()

        for conf in self.range_confs:
            kw_search = dict(conf["ffa_search"])
            kw_search.update({"deredden": False, "already_normalised": True})

            period_min = kw_search["period_min"]
            period_max = kw_search["period_max"]
            bins_min = kw_search["bins_min"]
            bins_max = kw_search["bins_max"]  
            
            ffa_plans = plan_ffa(nsamp, tsamp, period_min, period_max, bins_min, bins_max)      
            # ffa_plan : list of dict
            # Each dict contains:
            #     - 'downsample_factor'
            #     - 'tau' (effective sample time)
            #     - 'bins'
            #     - 'base_period' (tau*bins)
            #     - 'rows_eval' (rows used in FFA transform)    


            for ffa_plan in ffa_plans:
                downsample_fac = ffa_plan["downsample_factor"]
                tau = ffa_plan["tau"]
                base_period = ffa_plan["base_period"]
                print(base_period)
                bins = ffa_plan["bins"]
                rows_eval = ffa_plan["rows_eval"]


                # Step 1: run vis_ffa_search for each UV cell

                print(f"doing FFA transform on {vis_ts.unique_uv.shape} cells with base period {base_period}")
                #print(f"vis_ts data shape is {vis_ts.data.shape}")
                uv_ffa = vis_ffa_search_basep(dense_uv_ts, bins, vis_ts.unique_uv, tau)
                # uv_ffa_list.append(uv_ffa)

                ffa_cube = uv_ffa.ffa_array
                psf_img = self.psf
                #ffa_cube = (1 + 1j*0)*np.ones((vis_ts.unique_uv.shape[0], 10, 10))

                nx, ny = (self.nx, self.ny)

                print(vis_ts.unique_uv.shape)
                print(ffa_cube.shape)
                grid = np.zeros((ny, nx))
                print(vis_ts.unique_uv.shape)
                for uv, cube in zip(vis_ts.unique_uv, ffa_cube):
                    #print(uv)
                    #print(cube.shape)
                    grid[*uv] = cube[0, 0]
                # import matplotlib.pyplot as plt
                # plt.imshow(np.abs(grid), vmin=-3*np.std(grid), vmax=3*np.std(grid))
                # plt.show()
                # fft = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(grid))).real
                # plt.imshow(fft, vmin=-3*np.std(fft), vmax=3*np.std(fft))
                # plt.show()
                
                img_dict = test_vis_ffa_image_imgs(
                    ffa_cube,
                    vis_ts.unique_uv,
                    psf_img,
                    nx,
                    ny
                )

                all_candidates.append(img_dict)

        return all_candidates
