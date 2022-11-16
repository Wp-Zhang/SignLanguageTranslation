import os
import sys

sys.path.append("./")

from box import Box
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.SLR.data import VideoDataModule
from src.SLR.models import SLR_Lightning

if __name__ == "__main__":
    # * Load config and dataset info
    cfg = Box.from_yaml(open("configs/SLR/cfg-T.yaml", "r").read())
    dataset_cfg = cfg.dataset_args
    model_cfg = cfg.model_args
    optimizer_cfg = cfg.optimizer_args
    trainer_cfg = cfg.trainer_args
    eval_cfg = cfg.evaluate_args

    gloss_dict = np.load(
        os.path.join(dataset_cfg.info_dir, "gloss_dict.npy"), allow_pickle=True
    ).item()

    # * Define model and data module
    slr_model = SLR_Lightning(
        gloss_dict=gloss_dict, **model_cfg, **optimizer_cfg, **eval_cfg
    )
    dataset_cfg.batch_size *= 2
    dm = VideoDataModule(gloss_dict=gloss_dict, **dataset_cfg)

    # * Define trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=trainer_cfg.ckpt_dir,
        filename="Phoenix2014T-SLR-{epoch:02d}-{val_loss:.2f}",
    )
    trainer = Trainer(
        accelerator=trainer_cfg.accelerator,
        devices=1,  # trainer_cfg.devices
        max_epochs=trainer_cfg.max_epochs,
        callbacks=[checkpoint_callback],
    )

    trainer.validate(model=slr_model, datamodule=dm)
    trainer.test(model=slr_model, datamodule=dm)
