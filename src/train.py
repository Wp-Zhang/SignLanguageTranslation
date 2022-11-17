import os
import sys

sys.path.append("./")

from box import Box
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.SLR.data import VideoDataModule
from src.SLR.models import SLR_Lightning

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
dm = VideoDataModule(gloss_dict=gloss_dict, **dataset_cfg)

# * Define trainer
wandb_logger = WandbLogger(
    project="Sign Language Translation", entity="neu-ds5500-team13"
)

checkpoint_callback = ModelCheckpoint(
    dirpath=trainer_cfg.ckpt_dir, filename="Phoenix2014T-SLR-{epoch:02d}-{val_loss:.2f}"
)
trainer = Trainer(
    accelerator=trainer_cfg.accelerator,
    devices=trainer_cfg.devices,
    max_epochs=trainer_cfg.max_epochs,
    sync_batchnorm=True,
    strategy="ddp_find_unused_parameters_false",
    num_nodes=1,
    callbacks=[checkpoint_callback],
    logger=wandb_logger,
)

trainer.fit(slr_model, dm)
