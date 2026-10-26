import copy
import warnings

##### Non-standard imports #####
import numpy as np
try:
    import sparse
except ImportError:
    sparse = None
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u

##### Local imports #####
from .running_medians import fast_running_median
from .libffa import complex_downsample, downsample, generate_signal
from .reading import PrestoInf, SigprocHeader
from .metadata import Metadata
from .folding import fold
from .timing import timing


class TimeSeries(object):
    """
    Container for time series data to be searched with the FFA.
    **Use classmethods to create a new TimeSeries object.**

    Parameters
    ----------
    data : array_like
        Time series data to search.
    tsamp : float
        Sampling time of data in seconds.
    metadata : Metadata or dict, optional
        Optional Metadata object / dict describing the observation from which
        the data originate
    copy : bool, optional
        If set to True, the resulting time series will hold a new copy of data,
        otherwise it only holds a reference to it

    See Also
    --------
    TimeSeries.from_presto_inf : Load dedispersed data produced by PRESTO
    TimeSeries.from_sigproc : Load dedispersed data produced by SIGPROC
    TimeSeries.from_visibilities : Load gridded visibility timeseries
    TimeSeries.generate : Generate a noisy time series containing a fake pulsar signal
    """

    def __init__(self, data, tsamp, metadata=None, copy=False, dtype=None):
        if dtype is None:
            dtype = np.complex64 if np.iscomplexobj(data) else np.float32

        if copy:
            self._data = np.asarray(data, dtype=dtype).copy()
        else:
            self._data = np.asarray(data, dtype=dtype)
        self._tsamp = float(tsamp)
        self.metadata = Metadata(metadata) if metadata is not None else Metadata({})

        # Carrying a tobs attribute is quite practical in later stages of
        # the pipeline (peak detection in periodograms)
        self.metadata["tobs"] = self.length
        self.dtype = self._data.dtype

    @property
    def data(self):
        """numpy array holding the time series data, in float32 format."""
        return self._data

    @property
    def tsamp(self):
        """Sampling time in seconds."""
        return self._tsamp

    def copy(self):
        """Returns a new copy of the TimeSeries"""
        return copy.deepcopy(self)

    def normalise(self, inplace=False):
        """Normalise to zero mean and unit variance. if 'inplace' is False,
        a new TimeSeries object with the normalized data is returned.

        Parameters
        ----------
        inplace : bool, optional
            If set to True, perform the operation in-place, otherwise, a new
            TimeSeries object is returned

        Returns
        -------
        out : TimeSeries or None
            The normalised TimeSeries, if 'inplace' was set to False
        """
        if np.iscomplexobj(self.data):
            norm = self.data.var(dtype=np.complex128, axis=0)**0.5
            norm_data = self.data / norm
        else:
            m = self.data.mean(dtype=np.float64, axis=0)
            norm = self.data.std(dtype=np.float64, axis=0)
            norm_data = (self.data - m) / norm

        if inplace:
            self._data = norm_data
        else:
            return TimeSeries(
                norm_data, self.tsamp, metadata=self.metadata, dtype=self.dtype
            )

    @timing
    def deredden(self, width, minpts=101, inplace=False):
        """Subtract from the data an aproximate running median. To save time,
        this running median is computed on a downsampled copy of the data, then
        upsampled back to the original resolution and finally subtracted from
        the original data.

        Parameters
        ----------
        width : float
            Width of the running median window in seconds.
        minpts : int, optional
            Downsample the data so that the width of the running median window
            is equal to 'minpoints' samples. The running median will be computed
            on that downsampled copy of the data, and then upsampled
        inplace : bool, optional
            If set to True, perform the operation in-place, otherwise, a new
            TimeSeries object is returned

        Returns
        -------
        out : TimeSeries or None
            The de-reddened TimeSeries, if 'inplace' was set to False
        """
        width_samples = int(round(width / self.tsamp))
        rmed = fast_running_median(self.data, width_samples, minpts)
        if inplace:
            self._data -= rmed
        else:
            return TimeSeries(self.data - rmed, self.tsamp, metadata=self.metadata, dtype=self.dtype)

    def downsample(self, factor, inplace=False):
        """Downsample data by a real-valued factor, by grouping and adding
        together consecutive samples (or fractions of samples).

        Parameters
        ----------
        factor : float
            Downsampling factor
        inplace : bool, optional
            If set to True, perform the operation in-place, otherwise, a new
            TimeSeries object is returned

        Returns
        -------
        out : TimeSeries or None
            The downsampled TimeSeries, if 'inplace' was set to False
        """
        if np.issubdtype(self.dtype, np.complexfloating):
            downsample_func = complex_downsample
        else:
            downsample_func = downsample
        if inplace:
            self._data = downsample_func(self.data, factor)
            self._tsamp *= factor
        else:
            return TimeSeries(
                downsample_func(self.data, factor),
                factor * self.tsamp,
                metadata=self.metadata,
                dtype=self.dtype
            )

    def fold(self, period, bins, subints=None):
        """
        Fold TimeSeries at given period.

        Parameters
        ----------
        period : float
            Period in seconds
        bins : int
            Number of phase bins
        subints: int or None, optional
            Number of desired sub-integrations. If None, the number of
            sub-integrations will be the number of full periods that fit in
            the data

        Returns
        -------
        folded : ndarray
            The folded data as a numpy array. If subints > 1, it has a shape
            (subints, bins). Otherwise it is a 1D array with 'bins' elements.
        """
        return fold(self, period, bins, subints=subints)

    @classmethod
    def generate(
        cls, length, tsamp, period, phi0=0.5, ducy=0.02, amplitude=10.0, stdnoise=1.0
    ):
        """
        Generate a time series containing a periodic signal with a von Mises
        pulse profile, and some background white noise (optional).

        Parameters
        ----------
        length : float
            Length of the data in seconds.
        tsamp : float
            Sampling time in seconds.
        period : float
            Signal period in seconds
        phi0 : float, optional
            Initial pulse phase in number of periods
        ducy : float, optional
            Duty cycle of the pulse, i.e. the ratio FWHM / Period
        amplitude : float, optional
            True amplitude of the signal as defined in the reference paper.
            The *expectation* of the S/N of the generated signal is
                S/N_true = amplitude / stdnoise,
            assuming that a matched filter with the exact shape of the pulse is
            employed to measure S/N (here: von Mises with given duty cycle).
            riptide employs boxcar filters in the search, which results in a slight
            S/N loss. See the reference paper for details.
            A further degradation will be observed on bright signals, because
            they bias the estimation of the mean and standard deviation of the
            noise in a blind search.
        stdnoise : float, optional
            Standard deviation of the background noise (default: 1.0).
            If set to 0, a noiseless signal is generated.

        Returns
        -------
        tseries : ndarray (1D, float)
            Output time series.
        """
        nsamp = int(round(length / tsamp))
        period_samples = period / tsamp
        data = generate_signal(
            nsamp,
            period_samples,
            phi0=phi0,
            ducy=ducy,
            amplitude=amplitude,
            stdnoise=stdnoise,
        )
        metadata = Metadata(
            {
                "source_name": "fake",
                "signal_shape": "Von Mises",
                "signal_period": period,
                "signal_initial_phase": phi0,
                "signal_duty_cycle": ducy,
            }
        )
        return cls(data, tsamp, copy=False, metadata=metadata)

    @classmethod
    def from_numpy_array(cls, array, tsamp, copy=False):
        """Create a new TimeSeries from a numpy array (or array-like).

        Parameters
        ----------
        array : array-like
            The time series data.
        tsamp : float
            Sampling time of the data in seconds.
        copy : bool, optional
            If set to True, the resulting time series will hold a new copy of
            'array', otherwise it only holds a reference to it

        Returns
        -------
        out: TimeSeries
            TimeSeries object.
        """
        return cls(array, tsamp, copy=copy)

    @classmethod
    def from_binary(cls, fname, tsamp, dtype=np.float32):
        """Create a new TimeSeries from a raw binary file, containing the
        time series data without any header or footer. This will work as long
        as the data can be loaded with numpy.fromfile().

        Parameters
        ----------
        fname : str
            File name to load.
        tsamp : float
            Sampling time of the data in seconds.
        dtype : numpy data type, optional
            Data type of the file

        Returns
        -------
        out: TimeSeries
            TimeSeries object.
        """
        data = np.fromfile(fname, dtype=dtype)
        return cls(data, tsamp, copy=False, dtype=dtype)

    @classmethod
    def from_npy_file(cls, fname, tsamp):
        """Create a new TimeSeries from a .npy file, written with numpy.save().

        Parameters
        ----------
        fname : str
            File name to load.
        tsamp : float
            Sampling time of the data in seconds.

        Returns
        -------
        out : TimeSeries
            TimeSeries object.
        """
        data = np.load(fname)
        return cls(data, tsamp, copy=False)

    @classmethod
    @timing
    def from_presto_inf(cls, fname):
        """Create a new TimeSeries from a .inf file written by PRESTO. The
        associated .dat file must be in the same directory.

        Parameters
        ----------
        fname : str
            File name to load.

        Returns
        -------
        out: TimeSeries
            TimeSeries object.
        """
        inf = PrestoInf(fname)
        metadata = Metadata.from_presto_inf(inf)
        # TODO: check that the number of samples read from the .inf file
        # matches what is actually in the .dat file, although the possibility of
        # 'data breaks' could make this difficult
        ts = cls(inf.load_data(), tsamp=inf["tsamp"], metadata=metadata)

        em_band = ts.metadata["em_band"]
        if em_band in ("X-ray", "Gamma"):
            msg = (
                f" You have loaded file {fname!r}, which contains data observed at"
                f" a high-energy band {em_band!r}."
                " riptide is NOT designed to process low photon count time series,"
                " i.e. where the background noise statistics are non-Gaussian."
                " Be VERY careful when interpreting any search outputs."
            )
            warnings.warn(msg, category=UserWarning)
        return ts

    @classmethod
    @timing
    def from_sigproc(cls, fname, extra_keys={}):
        """Create a new TimeSeries from a file written by SIGPROC's dedisperse
        routine.

        Parameters
        ----------
        fname : str
            File name to load.
        extra_keys : dict, optional
            Optional {key: type} dictionary. Use it to specify how to parse any
            non-standard keys that could be found in the header, or even to
            override the data type of standard keys.

            Example:
                {
                'telescope_diameter' : float,
                'num_trusses': int,
                'planet_name': str
                }

        Returns
        -------
        out : TimeSeries
            TimeSeries object.
        """
        sig = SigprocHeader(fname, extra_keys=extra_keys)

        # This call checks if the file contains a dedispersed time series
        # in either 8-bit or 32-bit format
        # For 8-bit data, the signedness is specified via the 'signed' boolean key
        metadata = Metadata.from_sigproc(sig, extra_keys=extra_keys)

        # Load time series data
        with open(fname, "rb") as fobj:
            fobj.seek(sig.bytesize)
            if metadata["nbits"] == 8:
                dtype = np.int8 if metadata["signed"] else np.uint8
                # Don't forget to cast to float32 after reading !
                data = np.fromfile(fobj, dtype=dtype).astype(np.float32)
            else:  # assume float32
                data = np.fromfile(fobj, dtype=np.float32)

        return cls(data, tsamp=sig["tsamp"], metadata=metadata)

    @property
    def nsamp(self):
        """Number of samples in the data."""
        return self.data.size

    @property
    def length(self):
        """Length of the data in seconds"""
        return self.nsamp * self.tsamp

    @property
    def tobs(self):
        """Length of the data in seconds. Alias of property 'length'."""
        return self.length

    def __str__(self):
        name = type(self).__name__
        out = "{name} {{nsamp = {x.nsamp:d}, tsamp = {x.tsamp:.4e}, tobs = {x.length:.3f}}}".format(
            name=name, x=self
        )
        return out

    def __repr__(self):
        return str(self)

    @classmethod
    def from_dict(cls, items, dtype=np.float32):
        return cls(
            items["data"], items["tsamp"], metadata=items["metadata"], copy=False, dtype=dtype
        )

    def to_dict(self):
        return {"data": self.data, "tsamp": self.tsamp, "metadata": self.metadata}



class VisTimeSeries(object):
    """
    Container for time series data to be searched with the FFA.
    **Use classmethods to create a new TimeSeries object.**

    Parameters
    ----------
    data : array_like (multi-dimensional)
        Time series data to search.
    tsamp : float
        Sampling time of data in seconds.
    metadata : Metadata or dict, optional
        Optional Metadata object / dict describing the observation from which
        the data originate
    copy : bool, optional
        If set to True, the resulting time series will hold a new copy of data,
        otherwise it only holds a reference to it

    See Also
    --------
    TimeSeries.from_presto_inf : Load dedispersed data produced by PRESTO
    TimeSeries.from_sigproc : Load dedispersed data produced by SIGPROC
    TimeSeries.from_visibilities : Load gridded visibility timeseries
    TimeSeries.generate : Generate a noisy time series containing a fake pulsar signal
    """

    def __init__(self, data, tsamp, metadata=None, copy=False, nu=None, nv=None, du=None, dv=None, phase_centre=None, dtype=None, header=None, dga=None, unique_uv=None):
        if data is None:
            self._data = None
        elif copy:
            self._data = np.ascontiguousarray(data, dtype=np.complex64).copy()
        elif sparse is not None and isinstance(data, sparse.COO):
            self._data = data
        else:
            self._data = np.ascontiguousarray(data, dtype=np.complex64)
        self.dga = dga
        self.unique_uv = unique_uv
        self._tsamp = float(tsamp)
        self.metadata = Metadata(metadata) if metadata is not None else Metadata({})

        # Carrying a tobs attribute is quite practical in later stages of
        # the pipeline (peak detection in periodograms)
        self.metadata["tobs"] = self.length
        self.dtype = np.complex64 if dtype is None else dtype
        self.nu = nu
        self.nv = nv
        self.du = du
        self.dv = dv
        if phase_centre is None:
            self.phase_centre = None
        else:
            self.phase_centre = SkyCoord(phase_centre[0]*u.deg, phase_centre[1]*u.deg, frame="icrs") if isinstance(phase_centre, tuple) else phase_centre
        self.header = header

    @property
    def data(self):
        """numpy array holding the time series data, in complex64 format."""
        return self._data

    @data.setter
    def data(self, new_value):
        self._data = new_value

    @property
    def tsamp(self):
        """Sampling time in seconds."""
        return self._tsamp

    def copy(self):
        """Returns a new copy of the TimeSeries"""
        return copy.deepcopy(self)
    
    def get_sparse_unique_uv(self):
        # Take the max along the time axis (axis=2) to find any uv with nonzero timeseries
        # assumes self.data shape is (..., u, v)
        axes_to_reduce = tuple(range(self.data.ndim - 2))
        uv_mask = (self.data != 0).any(axis=axes_to_reduce)  # shape (u, v)

        if hasattr(uv_mask, "coords"):  # sparse array
            u_coords, v_coords = uv_mask.coords
        else:  # dense numpy array
            u_coords, v_coords = np.nonzero(uv_mask)

        self.unique_uv = np.column_stack([u_coords, v_coords])

    def make_dynamic_grid_array(self):
        if self.unique_uv is None:
            self.get_sparse_unique_uv()
        u_coords = self.unique_uv[:, 0]
        v_coords = self.unique_uv[:, 1]
        sub = self.data[:, u_coords, v_coords]
        if hasattr(sub, "todense"):
            sub = sub.todense()
        dynamic_grid_array = np.ascontiguousarray(np.transpose(sub))
        self.data = None
        self.dga = dynamic_grid_array
        
        
    def set_grid_params(self, nu, nv, du, dv):
        """
        set grid parameters nu, nv, du, dv i.e. size of grid on u and v axes,
        and the pixel size du, dv
        """
        self.nu = nu
        self.nv = nv
        self.du = du
        self.dv = dv

    def set_phase_centre(self, ra_deg, dec_deg):
        self.phase_centre = SkyCoord(ra_deg*u.deg, dec_deg*u.deg, frame="icrs")

    def set_header(self, header):
        self.header = header

    def get_skycoords_from_psf_header(self, header=None):
        """
        Convert (u, v) gridded visibilities into absolute sky coordinates.

        Parameters:
        -----------
        header : astropy.io.fits.Header
            FITS header containing gridded uv data (with NAXIS1/2 and CDELT1/2).

        Returns:
        --------
        sky_coords : astropy.coordinates.SkyCoord
            2D array of absolute sky coordinates (RA/Dec) with shape (ny, nx).
        """
        if self.phase_centre is None:
            raise ValueError("phase_centre is not set. Please set with set_phase_centre")

        if header is None:
            header = self.header
        
        # Image dimensions
        nx = header.get('NAXIS1')
        ny = header.get('NAXIS2')

        # uv pixel size (in wavelengths^-1)
        du = header.get('CDELT1', 1.0)  # Δu
        dv = header.get('CDELT2', 1.0)  # Δv

        #set the grid parameter attributes
        self.set_grid_params(nx, ny, du, dv)

        # Angular pixel size in radians (Fourier dual of baseline spacing)
        delta_l = 1.0 / (nx * abs(du))  # radians
        delta_m = 1.0 / (ny * abs(dv))  # radians

        # Offset grids (direction cosines l, m), centered
        l = (np.arange(nx) - nx // 2) * delta_l
        m = (np.arange(ny) - ny // 2) * delta_m
        l_grid, m_grid = np.meshgrid(l, m)  # shape (ny, nx)
        
        # Convert to astropy Quantity (radians)
        l_offsets = l_grid * u.rad
        m_offsets = m_grid * u.rad

        # Define offset frame centered on phase center
        offset_frame = SkyOffsetFrame(origin=self.phase_centre)

        # Create offset coordinates and transform to absolute frame
        offset_coords = SkyCoord(l_offsets, m_offsets, frame=offset_frame)
        sky_coords = offset_coords.transform_to(self.phase_centre.frame)
        self.sky_coords = sky_coords
        #a grid of SkyCoords of the same shape as the input image that can be indexed with pixel coords
        return sky_coords

    def normalise(self, inplace=False):
        """Normalise to zero mean and unit variance. if 'inplace' is False,
        a new TimeSeries object with the normalized data is returned.

        Parameters
        ----------
        inplace : bool, optional
            If set to True, perform the operation in-place, otherwise, a new
            TimeSeries object is returned

        Returns
        -------
        out : TimeSeries or None
            The normalised TimeSeries, if 'inplace' was set to False
        """
        # NOTE: use float64 accumulator to avoid saturation issues when the
        # data have large values
        m = np.mean(np.abs(self.data))*np.exp(1j*np.mean(np.angle(self.data)))
        v = np.var(self.data)
        norm = v**0.5

        if inplace:
            self._data = (self.data - m) / norm
        else:
            return VisTimeSeries(
                (self.data - m) / norm, self.tsamp, metadata=self.metadata
            )

    @timing
    def deredden(self, width, minpts=101, inplace=False):
        """Subtract from the data an aproximate running median. To save time,
        this running median is computed on a downsampled copy of the data, then
        upsampled back to the original resolution and finally subtracted from
        the original data.

        Parameters
        ----------
        width : float
            Width of the running median window in seconds.
        minpts : int, optional
            Downsample the data so that the width of the running median window
            is equal to 'minpoints' samples. The running median will be computed
            on that downsampled copy of the data, and then upsampled
        inplace : bool, optional
            If set to True, perform the operation in-place, otherwise, a new
            TimeSeries object is returned

        Returns
        -------
        out : TimeSeries or None
            The de-reddened TimeSeries, if 'inplace' was set to False
        """
        width_samples = int(round(width / self.tsamp))
        rmed = fast_running_median(self.data, width_samples, minpts)
        if inplace:
            self._data -= rmed
        else:
            return VisTimeSeries(self.data - rmed, self.tsamp, metadata=self.metadata)

    def downsample(self, factor, inplace=False):
        """Downsample data by a real-valued factor, by grouping and adding
        together consecutive samples (or fractions of samples).

        Parameters
        ----------
        factor : float
            Downsampling factor
        inplace : bool, optional
            If set to True, perform the operation in-place, otherwise, a new
            TimeSeries object is returned

        Returns
        -------
        out : TimeSeries or None
            The downsampled TimeSeries, if 'inplace' was set to False
        """
        if inplace:
            self._data = complex_downsample(self.data, factor)
            self._tsamp *= factor
        else:
            return VisTimeSeries(
                complex_downsample(self.data, factor),
                factor * self.tsamp,
                metadata=self.metadata,
            )

    def fold(self, period, bins, subints=None):
        """
        Fold TimeSeries at given period.

        Parameters
        ----------
        period : float
            Period in seconds
        bins : int
            Number of phase bins
        subints: int or None, optional
            Number of desired sub-integrations. If None, the number of
            sub-integrations will be the number of full periods that fit in
            the data

        Returns
        -------
        folded : ndarray
            The folded data as a numpy array. If subints > 1, it has a shape
            (subints, bins). Otherwise it is a 1D array with 'bins' elements.
        """
        return fold(self, period, bins, subints=subints)

    @classmethod
    def generate(
        cls, length, tsamp, period, phi0=0.5, ducy=0.02, amplitude=10.0, stdnoise=1.0
    ):
        """
        Generate a time series containing a periodic signal with a von Mises
        pulse profile, and some background white noise (optional).

        Parameters
        ----------
        length : float
            Length of the data in seconds.
        tsamp : float
            Sampling time in seconds.
        period : float
            Signal period in seconds
        phi0 : float, optional
            Initial pulse phase in number of periods
        ducy : float, optional
            Duty cycle of the pulse, i.e. the ratio FWHM / Period
        amplitude : float, optional
            True amplitude of the signal as defined in the reference paper.
            The *expectation* of the S/N of the generated signal is
                S/N_true = amplitude / stdnoise,
            assuming that a matched filter with the exact shape of the pulse is
            employed to measure S/N (here: von Mises with given duty cycle).
            riptide employs boxcar filters in the search, which results in a slight
            S/N loss. See the reference paper for details.
            A further degradation will be observed on bright signals, because
            they bias the estimation of the mean and standard deviation of the
            noise in a blind search.
        stdnoise : float, optional
            Standard deviation of the background noise (default: 1.0).
            If set to 0, a noiseless signal is generated.

        Returns
        -------
        tseries : ndarray (1D, float)
            Output time series.
        """
        nsamp = int(round(length / tsamp))
        period_samples = period / tsamp
        data = generate_signal(
            nsamp,
            period_samples,
            phi0=phi0,
            ducy=ducy,
            amplitude=amplitude,
            stdnoise=stdnoise,
        )
        metadata = Metadata(
            {
                "source_name": "fake",
                "signal_shape": "Von Mises",
                "signal_period": period,
                "signal_initial_phase": phi0,
                "signal_duty_cycle": ducy,
            }
        )
        return cls(data, tsamp, copy=False, metadata=metadata)

    @classmethod
    def from_real_imag_cube(cls, real_cube_file, imag_cube_file, threshold=3e-3, tsamp=None):

        with fits.open(real_cube_file, memmap=True) as real_hdul, fits.open(imag_cube_file, memmap=True) as imag_hdul:

            header = real_hdul[0].header
            wcs = WCS(header)
            nu = header["NAXIS1"]
            nv = header["NAXIS2"]
            du = header["CDELT1"]
            dv = header["CDELT2"]
            if tsamp is None:
                try:
                    fits_idx = wcs.axis_type_names.index("TIME") + 1
                    tsamp = float(header[f"CDELT{fits_idx}"])
                except (ValueError, KeyError):
                    tsamp = 0.110
            real_data = np.nan_to_num(real_hdul[0].data.squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
            imag_data = np.nan_to_num(imag_hdul[0].data.squeeze(), nan=0.0, posinf=0.0, neginf=0.0)

            data = real_data + 1j * imag_data

            # Apply threshold
            if sparse is not None:
                grid = sparse.COO(np.where(np.abs(data) > threshold, data, 0))
            else:
                grid = np.where(np.abs(data) > threshold, data, 0).astype(np.complex64)

        return cls(grid, tsamp, dtype=grid.dtype, nu=nu, nv=nv, du=du, dv=dv, header=header)

    @classmethod
    def from_real_imag_cube_chunked(cls, real_cube_file, imag_cube_file, u_slice=None, v_slice=None, threshold=3e-3, tsamp=None):
        """
        Load only a chunk of the cube along (u, v) dimensions.
        If u_slice or v_slice are None, take the full range in that dimension.
        """
        with fits.open(real_cube_file, memmap=True) as real_hdul, fits.open(imag_cube_file, memmap=True) as imag_hdul:

            header = real_hdul[0].header
            wcs = WCS(header)
            nu = header["NAXIS1"]
            nv = header["NAXIS2"]
            du = header["CDELT1"]
            dv = header["CDELT2"]
            if tsamp is None:
                try:
                    fits_idx = wcs.axis_type_names.index("TIME") + 1
                    tsamp = float(header[f"CDELT{fits_idx}"])
                except (ValueError, KeyError):
                    tsamp = 0.110

            # default full slices (axis -2 is V of size nv, axis -1 is U of size nu)
            u_slice = slice(0, nv) if u_slice is None else u_slice
            v_slice = slice(0, nu) if v_slice is None else v_slice

            # load just the requested chunk and sanitize NaNs/Infs
            real_data = np.nan_to_num(real_hdul[0].data[..., u_slice, v_slice], nan=0.0, posinf=0.0, neginf=0.0)
            imag_data = np.nan_to_num(imag_hdul[0].data[..., u_slice, v_slice], nan=0.0, posinf=0.0, neginf=0.0)

            raw = real_data + 1j * imag_data
            # Squeeze only leading singleton axes (e.g. STOKES, FREQ), preserving (Nt, chunk_v, chunk_u)
            if raw.ndim > 3:
                data = raw.reshape(-1, raw.shape[-2], raw.shape[-1])
            else:
                data = raw

            # Apply threshold
            if sparse is not None:
                grid = sparse.COO(np.where(np.abs(data) > threshold, data, 0))
            else:
                grid = np.where(np.abs(data) > threshold, data, 0).astype(np.complex64)

        return cls(grid, tsamp, dtype=grid.dtype,
                   nu=v_slice.stop - v_slice.start,
                   nv=u_slice.stop - u_slice.start,
                   du=du, dv=dv, header=header)

    @classmethod
    def from_visibility_stream(
        cls,
        msname_or_data,
        npix_x=384,
        npix_y=384,
        pixsize_x=22.0 / 206265.0,
        pixsize_y=22.0 / 206265.0,
        epsilon=1e-5,
        threshold=1e-4,
        nthreads=1,
        sigma_min=1.1,
        sigma_max=2.6,
        phase_centre=None,
        start_time_idx=None,
        end_time_idx=None,
        direct_dga=True
    ):
        """
        Directly stream visibility timestamps into a gridded VisTimeSeries container
        using ducc0.wgridder.vis2grid, eliminating intermediate 3D FITS cubes.

        Parameters
        ----------
        msname_or_data : str, tuple, or list
            Path to Measurement Set or tuple of (vis_3d, wgt_3d, uvw_3d, chan_freq, dt).
            - vis_3d: (nbl, nchan, nt) or (nbl, nt)
            - wgt_3d: (nbl, nchan, nt) or None
            - uvw_3d: (nbl, 3, nt) or (nbl, 3) in meters
            - chan_freq: 1D array of channel frequencies in Hz
            - dt: float sampling time in seconds
        npix_x, npix_y : int
            Target dirty image dimensions in pixels.
        pixsize_x, pixsize_y : float
            Projected pixel angular resolution in radians.
        epsilon : float
            DUCC gridding precision tolerance (e.g. 1e-5).
        threshold : float
            Cutoff amplitude for sparse UV cell retention.
        nthreads : int
            Number of CPU threads for parallel DUCC C++ gridding.
        direct_dga : bool
            If True, construct the 2D Dynamic Grid Array directly in memory.
        """
        import ducc0

        # 1. If MS path is provided, load into 3D arrays in a single pass
        if isinstance(msname_or_data, str):
            try:
                from fastducc.imaging import load_chunk_data
                _, chan_freq, dt, vis_3d, wgt_3d, uvw_3d = load_chunk_data(
                    msname_or_data,
                    start_time_idx=start_time_idx,
                    end_time_idx=end_time_idx,
                    corr_mode="average",
                    average_correlations=True,
                )
            except ImportError:
                try:
                    from casacore.tables import table
                except ImportError:
                    raise ImportError("casacore is required to stream from a Measurement Set.")
                with table(msname_or_data, readonly=True, ack=False) as t_main:
                    with table(f"{msname_or_data}/SPECTRAL_WINDOW", readonly=True, ack=False) as t_spw:
                        chan_freq = np.asarray(t_spw.getcol("CHAN_FREQ")[0], dtype=np.float64)
                    colnames = set(t_main.colnames())
                    time_col = "TIME_CENTROID" if "TIME_CENTROID" in colnames else "TIME"
                    all_times = t_main.getcol(time_col)
                    unique_times = np.unique(all_times)
                    if start_time_idx is not None or end_time_idx is not None:
                        s_idx = 0 if start_time_idx is None else int(start_time_idx)
                        e_idx = len(unique_times) if end_time_idx is None else int(end_time_idx) + 1
                        unique_times = unique_times[s_idx:e_idx]
                        t_sub = t_main.query(f"{time_col} >= {unique_times[0]:.17g} AND {time_col} <= {unique_times[-1]:.17g}")
                    else:
                        t_sub = t_main
                    try:
                        time_vals = t_sub.getcol(time_col)
                        a1 = t_sub.getcol("ANTENNA1")
                        a2 = t_sub.getcol("ANTENNA2")
                        uvw_all = np.asarray(t_sub.getcol("UVW"), dtype=np.float64)
                        data_all = np.asarray(t_sub.getcol("DATA"), dtype=np.complex64)
                        flags_all = np.asarray(t_sub.getcol("FLAG"), dtype=bool)
                    finally:
                        if t_sub is not t_main:
                            t_sub.close()

                good = ~flags_all
                if data_all.ndim == 3 and data_all.shape[-1] > 1:
                    # Average unflagged parallel/available correlations
                    wgt_all = good.astype(np.float32)
                    wsum = wgt_all.sum(axis=-1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        vis_flat = (data_all * wgt_all).sum(axis=-1) / np.where(wsum > 0, wsum, np.nan)
                    vis_flat = np.nan_to_num(vis_flat, nan=0.0).astype(np.complex64)
                    wgt_flat = wsum
                else:
                    vis_flat = data_all.squeeze(-1) if data_all.ndim == 3 else data_all
                    wgt_flat = good.squeeze(-1).astype(np.float32) if good.ndim == 3 else good.astype(np.float32)

                bl_keys = (a1.astype(np.int64) << 32) | a2.astype(np.int64)
                u_bls, bl_inv = np.unique(bl_keys, return_inverse=True)
                u_times, time_inv = np.unique(time_vals, return_inverse=True)
                nbl = len(u_bls)
                nt = len(u_times)
                nchan = len(chan_freq)
                dt = float(np.median(np.diff(u_times))) if nt > 1 else 0.110

                vis_3d = np.zeros((nbl, nchan, nt), dtype=np.complex64)
                wgt_3d = np.zeros((nbl, nchan, nt), dtype=np.float32)
                uvw_3d = np.zeros((nbl, 3, nt), dtype=np.float64)
                vis_3d[bl_inv, :, time_inv] = vis_flat
                wgt_3d[bl_inv, :, time_inv] = wgt_flat
                uvw_3d[bl_inv, :, time_inv] = uvw_all
        elif isinstance(msname_or_data, (tuple, list)):
            if len(msname_or_data) == 5:
                vis_3d, wgt_3d, uvw_3d, chan_freq, dt = msname_or_data
            elif len(msname_or_data) == 4:
                vis_3d, uvw_3d, chan_freq, dt = msname_or_data
                wgt_3d = None
            else:
                raise ValueError("Tuple input must be (vis, uvw, chan_freq, dt) or (vis, wgt, uvw, chan_freq, dt)")
        else:
            raise TypeError("Unsupported msname_or_data type: must be MS path or tuple.")

        if vis_3d.ndim == 2:
            vis_3d = vis_3d[:, np.newaxis, :]
        if wgt_3d is not None and wgt_3d.ndim == 2:
            wgt_3d = wgt_3d[:, np.newaxis, :]
        nbl, nchan, nt = vis_3d.shape
        chan_freq = np.asarray(chan_freq, dtype=np.float64)

        uvw_0 = uvw_3d[..., 0] if uvw_3d.ndim == 3 else uvw_3d
        vis_0 = vis_3d[..., 0]
        wgt_0 = wgt_3d[..., 0] if wgt_3d is not None else None

        grid_0, cfu, cfv = ducc0.wgridder.vis2grid(
            uvw=uvw_0, freq=chan_freq, vis=vis_0, wgt=wgt_0,
            npix_x=npix_x, npix_y=npix_y,
            pixsize_x=pixsize_x, pixsize_y=pixsize_y,
            epsilon=epsilon, nthreads=nthreads,
            sigma_min=sigma_min, sigma_max=sigma_max,
            return_corfunc=True
        )
        nu, nv = grid_0.shape

        uv_cube = np.zeros((nt, nu, nv), dtype=np.complex64)
        uv_cube[0] = grid_0
        grid_buf = np.zeros((nu, nv), dtype=np.complex64)

        for it in range(1, nt):
            uvw_t = uvw_3d[..., it] if uvw_3d.ndim == 3 else uvw_3d
            vis_t = vis_3d[..., it]
            wgt_t = wgt_3d[..., it] if wgt_3d is not None else None
            if wgt_t is not None and np.count_nonzero(wgt_t) == 0:
                uv_cube[it] = 0.0
                continue
            ducc0.wgridder.vis2grid(
                uvw=uvw_t, freq=chan_freq, vis=vis_t, wgt=wgt_t,
                npix_x=npix_x, npix_y=npix_y,
                pixsize_x=pixsize_x, pixsize_y=pixsize_y,
                epsilon=epsilon, nthreads=nthreads,
                sigma_min=sigma_min, sigma_max=sigma_max,
                grid=grid_buf
            )
            uv_cube[it] = grid_buf

        du = 1.0 / (nu * pixsize_x)
        dv = 1.0 / (nv * pixsize_y)

        if direct_dga:
            # Extract active UV cell time series directly without constructing a 3D sparse.COO object
            if threshold > 0:
                uv_mask = (np.abs(uv_cube) > threshold).any(axis=0)
            else:
                uv_mask = (uv_cube != 0).any(axis=0)
            u_coords, v_coords = np.nonzero(uv_mask)
            unique_uv = np.column_stack([u_coords, v_coords])
            sub = uv_cube[:, u_coords, v_coords]
            if threshold > 0:
                sub = np.where(np.abs(sub) > threshold, sub, 0)
            dga = np.ascontiguousarray(sub.T, dtype=np.complex64)
            del uv_cube, sub

            obj = cls(
                None, tsamp=dt, dtype=np.complex64,
                nu=nu, nv=nv, du=du, dv=dv, phase_centre=phase_centre,
                dga=dga, unique_uv=unique_uv
            )
        else:
            if threshold > 0 and sparse is not None:
                grid_data = sparse.COO(np.where(np.abs(uv_cube) > threshold, uv_cube, 0))
            elif threshold > 0:
                grid_data = np.where(np.abs(uv_cube) > threshold, uv_cube, 0).astype(np.complex64)
            else:
                grid_data = uv_cube

            obj = cls(
                grid_data, tsamp=dt, dtype=np.complex64,
                nu=nu, nv=nv, du=du, dv=dv, phase_centre=phase_centre
            )

        obj.cfu = cfu
        obj.cfv = cfv
        obj.npix_x = npix_x
        obj.npix_y = npix_y
        obj.pixsize_x = pixsize_x
        obj.pixsize_y = pixsize_y

        return obj

    @classmethod
    def from_img_cube(cls, cube_file):
        with fits.open(cube_file, memmap=True) as hdul:
            header = hdul[0].header
            wcs = WCS(header)
            nu = header["NAXIS1"]
            nv = header["NAXIS2"]
            du = header["CDELT1"]
            dv = header["CDELT2"]
            ctype = "TIME"
            fits_idx = wcs.axis_type_names.index(ctype) + 1
            tsamp = float(header[f"CDELT{fits_idx}"])
            cubedata = np.nan_to_num(hdul[0].data.squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
        
        return cls(cubedata, tsamp, dtype=cubedata.dtype, nu=nu, nv=nv, du=du, dv=dv)

    @classmethod
    def from_sparse_npz(cls, cube_file):
        raise NotImplementedError("from_sparse_npz is not implemented yet....")

    @classmethod
    def from_numpy_array(cls, array, tsamp, copy=False):
        """Create a new TimeSeries from a numpy array (or array-like).

        Parameters
        ----------
        array : array-like
            The time series data.
        tsamp : float
            Sampling time of the data in seconds.
        copy : bool, optional
            If set to True, the resulting time series will hold a new copy of
            'array', otherwise it only holds a reference to it

        Returns
        -------
        out: TimeSeries
            TimeSeries object.
        """
        return cls(array, tsamp, copy=copy)

    @classmethod
    def from_binary(cls, fname, tsamp, dtype=np.complex64):
        """Create a new TimeSeries from a raw binary file, containing the
        time series data without any header or footer. This will work as long
        as the data can be loaded with numpy.fromfile().

        Parameters
        ----------
        fname : str
            File name to load.
        tsamp : float
            Sampling time of the data in seconds.
        dtype : numpy data type, optional
            Data type of the file

        Returns
        -------
        out: TimeSeries
            TimeSeries object.
        """
        data = np.fromfile(fname, dtype=dtype)
        return cls(data, tsamp, copy=False)

    @classmethod
    def from_npy_file(cls, fname, tsamp):
        """Create a new TimeSeries from a .npy file, written with numpy.save().

        Parameters
        ----------
        fname : str
            File name to load.
        tsamp : float
            Sampling time of the data in seconds.

        Returns
        -------
        out : TimeSeries
            TimeSeries object.
        """
        data = np.load(fname)
        return cls(data, tsamp, copy=False)

    @property
    def nsamp(self):
        """Number of samples in the data."""
        #assuming time is along first axis
        if self.data is not None:
            return self.data.shape[0]
        elif hasattr(self, 'dga') and self.dga is not None:
            return self.dga.shape[1]
        return 0

    @property
    def img_size(self):
        return(self.data.shape[1:])

    @property
    def length(self):
        """Length of the data in seconds"""
        return self.nsamp * self.tsamp

    @property
    def tobs(self):
        """Length of the data in seconds. Alias of property 'length'."""
        return self.length

    def __str__(self):
        name = type(self).__name__
        out = "{name} {{nsamp = {x.nsamp:d}, tsamp = {x.tsamp:.4e}, tobs = {x.length:.3f}}}".format(
            name=name, x=self
        )
        return out

    def __repr__(self):
        return str(self)

    @classmethod
    def from_dict(cls, items):
        return cls(
            items["data"], items["tsamp"], metadata=items["metadata"], copy=False
        )

    def to_dict(self):
        return {"data": self.data, "tsamp": self.tsamp, "metadata": self.metadata}
    
    def index(self, ind):
        """grabs one index of multi dimensional array and returns TimeSeries object for 1D ts data"""
        return TimeSeries.from_dict({
            "data": self.data[(slice(None),) + tuple(ind)].todense(),
            "tsamp": self.tsamp,
            "metadata": self.metadata
            }, 
            dtype=np.complex64
        )
    
    def index_np(self, ind):
        """grabs one index of multi dimensional array and returns TimeSeries object for 1D ts data"""
        return self.data[(slice(None),) + tuple(ind)].todense()
