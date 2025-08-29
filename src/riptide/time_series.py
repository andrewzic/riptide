import copy
import warnings

##### Non-standard imports #####
import numpy as np
import sparse
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
        if copy:
            self._data = np.asarray(data, dtype=dtype).copy()
        else:
            self._data = np.asarray(data, dtype=dtype)
        self._tsamp = float(tsamp)
        self.metadata = Metadata(metadata) if metadata is not None else Metadata({})

        # Carrying a tobs attribute is quite practical in later stages of
        # the pipeline (peak detection in periodograms)
        self.metadata["tobs"] = self.length
        if dtype is None:
            self.dtype = np.float32
        else:
            self.dtype = dtype #probably want to check this is somethign sensible here

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
        # NOTE: use float64 accumulator to avoid saturation issues when the
        # data have large values
        m = self.data.mean(dtype=np.complex128, axis=0) #complex
        v = self.data.var(dtype=np.complex128, axis=0) #real and positive
        norm = v**0.5

        if inplace:
            # i think we don't want to mean subtract as that will mess complex phases
            self._data = (self.data ) / norm 
        else:
            return TimeSeries(
                (self.data ) / norm, self.tsamp, metadata=self.metadata, dtype=self.dtype
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
        return cls(data, tsamp, copy=False, dtype=type(data))

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

    def __init__(self, data, tsamp, metadata=None, copy=False, nu=None, nv=None, du=None, dv=None, phase_centre=None, dtype=None, header=None):
        if copy:
            self._data = np.ascontiguousarray(data, dtype=np.complex64).copy()
        elif isinstance(data, sparse.COO):
            self._data = data
        else:
            self._data = np.ascontiguousarray(data, dtype=np.complex64)
        self._tsamp = float(tsamp)
        self.metadata = Metadata(metadata) if metadata is not None else Metadata({})

        # Carrying a tobs attribute is quite practical in later stages of
        # the pipeline (peak detection in periodograms)
        self.metadata["tobs"] = self.length
        if dtype is None:
            self.dtype = np.complex64
        self.nu = nu
        self.nv = nv
        self.du = du
        self.dv = dv
        if phase_centre is None:
            self.phase_centre = None #
        else:
            self.phase_centre = SkyCoord(phase_centre[0]*u.deg, phase_centre[1]*u.deg, frame="icrs") if isinstance(phase_centre, tuple) else phase_centre
        self.unique_uv = None
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
        dynamic_grid_array = np.ascontiguousarray(np.transpose(self.data[:, u_coords, v_coords].todense()))
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

        #phase_centre: tuple with ra_deg, dec_deg
        print(ra_deg, dec_deg)
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
        print(l_grid)
        
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
            return TimeSeries(
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
            return TimeSeries(self.data - rmed, self.tsamp, metadata=self.metadata)

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
    def from_real_imag_cube(cls, real_cube_file, imag_cube_file, threshold=3e-3):

        with fits.open(real_cube_file, memmap=True) as real_hdul, fits.open(imag_cube_file, memmap=True) as imag_hdul:

            header = real_hdul[0].header
            wcs = WCS(header)
            nu = header["NAXIS1"]
            nv = header["NAXIS2"]
            du = header["CDELT1"]
            dv = header["CDELT2"]
            ctype = "TIME" #hard coded
            fits_idx = wcs.axis_type_names.index(ctype) + 1
            tsamp = float(header[f"CDELT{fits_idx}"])
            print(real_hdul[0])
            print(real_hdul[0].data.shape)
            real_data = np.nan_to_num(real_hdul[0].data.squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
            imag_data = np.nan_to_num(imag_hdul[0].data.squeeze(), nan=0.0, posinf=0.0, neginf=0.0)

            data = real_data + 1j * imag_data

            # Apply threshold
            sparse_grid = sparse.COO(np.where(np.abs(data) > threshold, data, 0))

        return cls(sparse_grid, tsamp, dtype=sparse_grid.dtype, nu=nu, nv=nv, du=du, dv=dv, header=header)

    @classmethod
    def from_real_imag_cube_chunked(cls, real_cube_file, imag_cube_file, u_slice=None, v_slice=None, threshold=3e-3):
        """
        Load only a chunk of the cube along (u, v) dimensions.
        If u_slice or v_slice are None, take the full range in that dimension.
        """
        print(f"in here with {u_slice.start}, {u_slice.stop}; {v_slice.start}, {v_slice.stop}")
        with fits.open(real_cube_file, memmap=True) as real_hdul, fits.open(imag_cube_file, memmap=True) as imag_hdul:

            header = real_hdul[0].header
            wcs = WCS(header)
            nu = header["NAXIS1"]
            nv = header["NAXIS2"]
            du = header["CDELT1"]
            dv = header["CDELT2"]
            ctype = "TIME"
            fits_idx = wcs.axis_type_names.index(ctype) + 1
            tsamp = float(header[f"CDELT{fits_idx}"])

            # default full slices
            u_slice = slice(0, nu) if u_slice is None else u_slice
            v_slice = slice(0, nv) if v_slice is None else v_slice

            # load just the requested chunk
            real_data = real_hdul[0].data[..., u_slice, v_slice] #, nan=0.0, posinf=0.0, neginf=0.0)
            imag_data = imag_hdul[0].data[..., u_slice, v_slice] #, nan=0.0, posinf=0.0, neginf=0.0)

            data = real_data + 1j * imag_data
            print(f"size of data is {data.nbytes}")
            # real = da.from_array(real_hdul[0].data, chunks=(time_chunk, u_chunk, v_chunk))
            # imag = da.from_array(imag_hdul[0].data, chunks=(time_chunk, u_chunk, v_chunk))
            # data = da.map_blocks(lambda r, i: r + 1j*i, real, imag)
            # data = da.where(da.abs(data) > threshold, data, 0)


            # Apply threshold
            sparse_grid = sparse.COO(np.where(np.abs(data) > threshold, data, 0))

        return cls(sparse_grid, tsamp, dtype=sparse_grid.dtype,
                   nu=u_slice.stop - u_slice.start,
                   nv=v_slice.stop - v_slice.start,
                   du=du, dv=dv, header=header)
    
    # @classmethod
    # def from_real_imag_cube_chunked(cls, real_cube_file, imag_cube_file,
    #                                 threshold=3e-3, chunk_size=64):
    #     """
    #     Chunked loader for large visibility cubes.
    #     - Leaves time dimension intact
    #     - Chunks along last 2 axes (u,v)
    #     - Builds a sparse cube lazily with dask

    #     Parameters
    #     ----------
    #     real_cube_file : str
    #         Path to FITS file containing real part of visibilities
    #     imag_cube_file : str
    #         Path to FITS file containing imaginary part of visibilities
    #     threshold : float
    #         Absolute value threshold below which values are set to zero
    #     chunk_size : int
    #         Spatial chunk size along u,v dimensions

    #     Returns
    #     -------
    #     VisTimeSeries
    #         Instance with dask-backed sparse data cube
    #     """
    #     # Open headers
    #     with fits.open(real_cube_file, memmap=True) as real_hdul:
    #         header = real_hdul[0].header
    #         wcs = WCS(header)

    #     nu = header["NAXIS1"]
    #     nv = header["NAXIS2"]
    #     fits_idx = wcs.axis_type_names.index("TIME") + 1
    #     tsamp = float(header[f"CDELT{fits_idx}"])
    #     du = header["CDELT1"]
    #     dv = header["CDELT2"]

    #     # Open lazily with dask
    #     real_darr = da.from_array(
    #         fits.getdata(real_cube_file, memmap=True),
    #         chunks=(-1, chunk_size, chunk_size)  # keep time axis whole
    #     )
    #     imag_darr = da.from_array(
    #         fits.getdata(imag_cube_file, memmap=True),
    #         chunks=(-1, chunk_size, chunk_size)
    #     )

    #     darr = real_darr + 1j * imag_darr

    #     # Threshold lazily
    #     darr = da.where(da.abs(darr) > threshold, darr, 0.0)

    #     # At this stage darr is (time, v, u) with dask backing
    #     # Convert to sparse for efficient indexing
    #     # NOTE: keep as dask.array until densification step
    #     sparse_grid = sparse.from_numpy(darr.compute())  # or wrap lazy later

    #     return cls(
    #         sparse_grid,
    #         tsamp,
    #         dtype=sparse_grid.dtype,
    #         nu=nu,
    #         nv=nv,
    #         du=du,
    #         dv=dv,
    #         header=header
    #     )

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
        return self.data.shape[0]

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
            "data": self.data[:, *ind].todense(),
            "tsamp": self.tsamp,
            "metadata": self.metadata
            }, 
            dtype=np.complex64
        )
    
    def index_np(self, ind):
        """grabs one index of multi dimensional array and returns TimeSeries object for 1D ts data"""
        return self.data[:, *ind].todense()
