

import riptide
import os
import glob
import tempfile
from copy import deepcopy

import yaml
import numpy as np
from pytest import raises
from riptide import load_json
from riptide import TimeSeries
from riptide.pipeline.pipeline import get_parser, run_program
from riptide.pipeline.config_validation import InvalidPipelineConfig, InvalidSearchRange
from riptide.pipeline.worker_pool import VisWorkerPool
from riptide.pipeline import VisPipeline

from riptide import VisTimeSeries, vis_ffa_search

import pandas as pd
import  matplotlib.pyplot as plt

config = "example_vis_J1832.yaml"
pipe = VisPipeline.from_yaml_config(config)

#big ones
# real_files = glob.glob("data/J0901-4046_small-t2000*real.cube.fits")
# imag_files = glob.glob("data/J0901-4046_small-t2000*imag.cube.fits")

real_files = glob.glob("data/J1832_real_cube.fits")
imag_files = glob.glob("data/J1832_imag_cube.fits")


pipe.prepare(real_files, imag_files)
all_results = pipe.vis_test_img(real_files, imag_files)
print(all_results)
d = all_results[0]
print(d)

for img, first_img, first_grid, period in zip(d[0]["images"], d[0]["first_images"], d[0]["first_grids"], d[0]["periods"]):
    fig, axs = plt.subplots(1,3, figsize=(10,4))
    grid_ = np.fft.fftshift(np.abs(first_grid))
    grid_[grid_<1e-3] = np.nan
    axs[0].imshow(grid_, origin='lower', interpolation='none', aspect='auto')
    axs[1].imshow(first_img, origin='lower', interpolation='none', aspect='auto')
    axs[2].imshow(img, origin='lower', interpolation='none', aspect='auto')
    plt.title(f"P={period:.3f}")
    plt.show()

# for res in all_results:
#     plt.imshow()
#     print(res)
#     print(len(res))
#     for r in res:
#         fig, axs = plt.subplots(1,2)
#         axs[0].imshow(r["image"])
#         axs[1].imshow(np.abs(r["uvgrid"]))
#         plt.show()
