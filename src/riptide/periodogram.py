##### Non-standard imports #####
import numpy as np
import matplotlib.pyplot as plt

##### Local imports #####
from .metadata import Metadata


class Periodogram(object):
    """
    Stores the raw output of the FFA search of a time series.

    Attributes
    ----------
    widths : ndarray
        Sequence of pulse width trials, in number of phase bins

    periods : ndarray
        Sequence of trial periods in seconds

    foldbins : ndarray
        Sequence with the same length as `periods`, containing the exact number of phase bins with
        which the data were folded for each particular trial period.

    snrs : ndarray
        Two dimensional array with shape (num_periods, num_widths) containing the S/N as a function
        of trial pulse width and period.
    """

    def __init__(self, widths, periods, foldbins, snrs, metadata=None):
        self.widths = widths
        self.periods = periods
        self.foldbins = foldbins
        self.snrs = snrs
        self.metadata = metadata if metadata is not None else Metadata({})

    @property
    def freqs(self):
        """Sequence of trial frequencies in Hz, in **decreasing** order"""
        return 1.0 / self.periods

    @property
    def tobs(self):
        """Length in seconds of the TimeSeries that was searched"""
        return self.metadata["tobs"]

    def to_dict(self):
        return {
            "widths": self.widths,
            "periods": self.periods,
            "foldbins": self.foldbins,
            "snrs": self.snrs,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, items):
        return cls(
            items["widths"],
            items["periods"],
            items["foldbins"],
            items["snrs"],
            metadata=items["metadata"],
        )

    def plot(self, iwidth=None):
        """
        Make a S/N versus trial period plot in the current matplotlib figure.

        Parameters
        ----------
        iwidth : int or None, optional
            Display only the data for this specific pulse width trial index.
            If None, for each trial period, plot the highest S/N across all trial pulse widths.
        """
        if iwidth is None:
            snr = self.snrs.max(axis=1)
        else:
            snr = self.snrs[:, iwidth]

        plt.plot(self.periods, snr, marker="o", markersize=2, alpha=0.5)
        plt.xlim(self.periods.min(), self.periods.max())
        plt.xlabel("Trial Period (s)", fontsize=16)
        plt.ylabel("S/N", fontsize=16)

        if iwidth is None:
            plt.title("Best S/N at any trial width", fontsize=18)
        else:
            width_bins = self.widths[iwidth]
            plt.title("S/N at trial width = %d" % width_bins, fontsize=18)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(linestyle=":")
        plt.tight_layout()

    def display(self, iwidth=None, figsize=(20, 5), dpi=100):
        """
        Display a plot S/N versus trial period. Creates a matplotlib figure, calls `plot()` and
        `pyplot.show()`.
        """
        plt.figure(figsize=figsize, dpi=dpi)
        self.plot(iwidth=iwidth)
        plt.show()


class UV_FFA(object):
    """
    Stores the raw output of the FFA search of a time series.

    Attributes
    ----------
    uv_ind : tuple (or similar)
            Simple tuple to indicate the relevant grid cell (indexes, not sky-bsed)

    periods : ndarray
        Sequence of trial periods in seconds

    foldbins : ndarray
        Sequence with the same length as `periods`, containing the exact number of phase bins with
        which the data were folded for each particular trial period.

    snrs : ndarray
        Two dimensional array with shape (num_periods, num_widths) containing the S/N as a function
        of trial pulse width and period.
    """

    def __init__(self, uv_ind, periods, foldbins, base_periods, tsamps, ffa_list, metadata=None):
        self.uv_ind = uv_ind
        self.periods = periods
        self.foldbins = foldbins
        self.base_periods = base_periods
        self.tsamps = tsamps
        self.ffa_list = ffa_list
        self.metadata = metadata if metadata is not None else Metadata({})

    @property
    def freqs(self):
        """Sequence of trial frequencies in Hz, in **decreasing** order"""
        return 1.0 / self.periods
    
    @property
    def u(self):
        return uv_inds[0]
    
    @property
    def v(self):
        return uv_inds[1]

    @property
    def tobs(self):
        """Length in seconds of the TimeSeries that was searched"""
        return self.metadata["tobs"]

    def to_dict(self):
        return {
            "uv_ind": uv_ind,
            "periods": self.periods,
            "foldbins": self.foldbins,
            "base_periods": self.base_periods,
            "tsamps": self.tsamps,
            "ffas": self.ffa_list,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, items):
        return cls(
            items["uv_ind"],
            items["periods"],
            items["foldbins"],
            items["base_periods"],
            items["tsamps"],
            items["ffas"],
            metadata=items["metadata"],
        )