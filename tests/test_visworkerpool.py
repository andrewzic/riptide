

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

config = "example_vis.yaml"
pipe = VisPipeline.from_yaml_config(config)

real_files = glob.glob("data/*real*.fits")
imag_files = glob.glob("data/*imag*.fits")
pipe.prepare(real_files, imag_files)
all_cands = pipe.vis_search(real_files, imag_files)
print(all_cands[0])
for i, df in enumerate(all_cands):
    df.to_pickle(f"cands_{real_files[i]}.pkl")
