import glob
import matplotlib.pyplot as plt
import pandas as pd

from riptide.pipeline import VisPipeline

# Load pipeline config
config = "example_vis.yaml"
pipe = VisPipeline.from_yaml_config(config)

# Input files
real_files = glob.glob("data/J0901-4046_small-t2000*real.cube.fits")
imag_files = glob.glob("data/J0901-4046_small-t2000*imag.cube.fits")

# Optional: shorter test set
# real_files = glob.glob("data/J0901-4046_short_small-t0004-uv*real.cube.fits")
# imag_files = glob.glob("data/J0901-4046_short_small-t0004-uv*imag.cube.fits")

# Prepare pipeline
pipe.prepare(real_files, imag_files)

# Run with local Dask cluster: 32 workers, 8GB each
all_cands = pipe.vis_search_basep_dask(
    real_files,
    imag_files,
    use_slurm=False,
    n_workers=32,
    memory_per_worker="8GB"
)

# Inspect results
print(all_cands[0])

# Save and visualise cutouts
for i, df in enumerate(all_cands):
    df.to_pickle(f"cands_{real_files[i]}.pkl")
    for idx, row in df.iterrows():
        plt.imshow(row["cutout"], origin="lower", interpolation="none")
        plt.title(f"Candidate SNR={row['snr']:.2f}, P={row['period']:.2f}")
        plt.show()