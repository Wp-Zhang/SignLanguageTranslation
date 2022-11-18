import os
import sys
import argparse
import pickle
from box import Box
from pytorch_lightning import Trainer

sys.path.append("./")

from src.SLR.data import VideoDataModule
from src.SLR.models import SLR_Lightning

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on specified dataset.")
    parser.add_argument(
        "--config",
        default="configs/SLR/phoenix2014T-res18.yaml",
        help="path to the configuration file",
    )
    parser.add_argument(
        "--weights",
        help="path to the checkpoint file",
    )
    args = parser.parse_args()

    # * Load config and dataset info
    cfg = Box.from_yaml(open(args.config, "r").read())
    dataset_cfg = cfg.dataset_args
    model_cfg = cfg.model_args
    optimizer_cfg = cfg.optimizer_args
    trainer_cfg = cfg.trainer_args
    eval_cfg = cfg.evaluate_args

    gloss_dict = pickle.load(
        open(os.path.join(dataset_cfg.processed_info_dir, "gloss_dict.pkl"), "rb")
    )

    # * Define model and data module
    slr_model = SLR_Lightning(
        gloss_dict=gloss_dict,
        eval_label_dir=dataset_cfg.processed_info_dir,
        **model_cfg,
        **optimizer_cfg,
        **eval_cfg
    )

    dm = VideoDataModule(
        info_dir=dataset_cfg.processed_info_dir,
        img_dir=dataset_cfg.processed_img_dir,
        gloss_dict=gloss_dict,
        batch_size=dataset_cfg.batch_size,
        num_worker=dataset_cfg.num_worker,
    )

    # * Define trainer
    trainer = Trainer(
        accelerator=trainer_cfg.accelerator,
        devices=1,
        max_epochs=trainer_cfg.max_epochs,
    )

    trainer.validate(model=slr_model, datamodule=dm, ckpt_path=args.weights)
    trainer.test(model=slr_model, datamodule=dm, ckpt_path=args.weights)
