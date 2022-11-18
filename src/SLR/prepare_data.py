import os
import sys
from box import Box
import argparse
import warnings

warnings.filterwarnings("ignore")
sys.path.append("./")

from src.SLR.data import preprocess_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess specified dataset.")
    parser.add_argument(
        "--config",
        default="configs/SLR/phoenix2014T-res18.yaml",
        help="path to the configuration file",
    )
    args = parser.parse_args()
    # * Load config and dataset info
    cfg = Box.from_yaml(open(args.config, "r").read())

    dataset_cfg = cfg.dataset_args
    preprocess_dataset(
        dataset_name=dataset_cfg.dataset_name,
        raw_info_dir=dataset_cfg.raw_info_dir,
        raw_img_dir=dataset_cfg.raw_img_dir,
        processed_info_dir=dataset_cfg.processed_info_dir,
        processed_img_dir=dataset_cfg.processed_img_dir,
        target_size=dataset_cfg.img_size,
        n_jobs=dataset_cfg.num_worker,
        resize_img=True,
    )
