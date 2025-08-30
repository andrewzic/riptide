

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

config = "example_vis_70.yaml"
pipe = VisPipeline.from_yaml_config(config)

#big ones
# real_files = glob.glob("data/J0901-4046_small-t2000*real.cube.fits")
# imag_files = glob.glob("data/J0901-4046_small-t2000*imag.cube.fits")

real_files = glob.glob("data/J0901-4046_short_small-t0004-uv*real.cube.fits")
imag_files = glob.glob("data/J0901-4046_short_small-t0004-uv*imag.cube.fits")


pipe.prepare(real_files, imag_files)
all_results = pipe.vis_test_img(real_files, imag_files)
d = all_results[0]
for img, period in zip(d["images"], d["periods"]):
    plt.imshow(img)
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
