# NOTE: best to place this at the top in case we want to import
# it in other files
from ._version import version as __version__

### Major classes
from .time_series import TimeSeries, VisTimeSeries
from .periodogram import Periodogram, UV_FFA
from .metadata import Metadata
from .candidate import Candidate

### Major functions
from .search import ffa_search
from .vis_search import vis_ffa_search, vis_ffa_image_candidates
from .running_medians import running_median, fast_running_median

from .libffa import ffa1, ffa2, ffafreq, ffaprd, generate_signal, downsample, complex_downsample, boxcar_snr

from .peak_detection import find_peaks

### Serialization
from .serialization import save_json, load_json

__all__ = [
    "TimeSeries",
    "VisTimeSeries",
    "Periodogram",
    "UV_FFA",
    "Metadata",
    "Candidate",
    "ffa_search",
    "vis_ffa_search",
    "vis_ffa_image_candidates",
    "ffa1",
    "ffa2",
    "ffafreq",
    "ffaprd",
    "generate_signal",
    "downsample",
    "boxcar_snr",
    "find_peaks",
    "save_json",
    "load_json",
]
