import glob
import matplotlib.pyplot as plt
import pandas as pd
from riptide.pipeline import VisPipeline


def main():
    config = "example_vis.yaml"
    pipe = VisPipeline.from_yaml_config(config)

    # Input data
    real_files = glob.glob("data/J0901-4046_small-t2000*real.cube.fits")
    imag_files = glob.glob("data/J0901-4046_small-t2000*imag.cube.fits")

    pipe.prepare(real_files, imag_files)

    # Run search with local cluster (32 workers, 8GB each)
    all_cands = pipe.vis_search_basep_dask(
        real_files,
        imag_files,
        use_slurm=False,
        n_workers=24,
        memory_per_worker="5GB",
        periods_per_block=16,
        u_chunks=8, v_chunks=8,
        dashboard_address=":8790",  # avoids port 8787 conflict
    )

    # Save and visualise results
    print(all_cands[0])
    for i, df in enumerate(all_cands):
        df.to_pickle(f"cands_{real_files[i]}.pkl")
        for _, row in df.iterrows():
            plt.imshow(row["cutout"], origin="lower", interpolation="none")
            plt.show()


if __name__ == "__main__":
    main()
