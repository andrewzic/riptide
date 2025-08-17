import logging
import multiprocessing

from riptide import TimeSeries, ffa_search, find_peaks
from riptide import VisTimeSeries, vis_ffa_search

from astropy.wcs import WCS


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
        header = fits.getheader(self.psf.file)
        ny, nx = psf.shape
        self.nx = nx
        self.ny = ny
        wcs = WCS(header)
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
    
    def process_uvcells(self, fname):

        all_candidates = []
        vis_ts = self.loader(fname)
        #get list of unique u, v pixel cell coords (indices)
        uvcells = vis_ts.get_sparse_unique_xy()

        #this will set self.ra_deg, self.dec_deg, and self.psf
        self.load_psf_img(psf_img_file=self.psf_file)
        #now set the phase centre attribute of the visibility timeseries
        vis_ts.set_phase_centre(self.ra_deg, self.dec_deg)
        #now set the vis_ts.sky_coords grid by calling get_skycoords_from_psf_header
        vis_ts.get_skycoords_from_psf_header(self.psf_header)
        
        for conf in self.range_confs:
            kw_search = dict(conf["ffa_search"])
            kw_search.update({"deredden": False, "already_normalised": True})

            # Step 1: run vis_ffa_search for each UV cell
            uv_ffa_list = []
            for uvcell in uvcells:
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
            psf_img = vis_ts.psf
            nx, ny = (self.nx, self.ny)
            for trial_idx in range(num_trials):
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
                    snr_thresh=conf["candidate_search"]["snr_thresh"],
                    max_candidates_width=conf["candidate_search"]["max_candidates"],
                    max_candidates_all=conf["candidate_search"]["max_candidates_all"]
                )
                    # py::dict d;
                    # d["x"] = c.x;
                    # d["y"] = c.y;
                    # d["snr"] = c.snr;
                    # d["width"] = c.width_bins;  // width in bins
                # vis_ffa_image_candidates(
                #                         ffa_blocks,
                #                         uv_indices,
                #                         psf_image,
                #                         nx,
                #                         ny,
                #                         snr_thresh=8.0,
                #                         ducy_max=0.5,
                #                         wtsp=1.5,
                #                         max_candidates_width=100,
                #                         max_candidates_all=1000,
                #                         mask_radius=2
                #                     )
                for cand in trial_candidates:
                    # cand.x, cand.y are image pixel coords
                    skycoord = vis_ts.sky_coords[cand.y, cand.x]  # lookup
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