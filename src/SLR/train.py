import os
import sys
import warnings

warnings.filterwarnings("ignore")
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
    os.path.join(dataset_cfg.processed_info_dir, "gloss_dict.npy"), allow_pickle=True
).item()

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
wandb_logger = WandbLogger(
    project="Sign Language Translation", entity="neu-ds5500-team13"
)
try:
    wandb_logger.experiment.config.update(cfg.to_dict())
except:
    pass

checkpoint_callback = ModelCheckpoint(
    dirpath=trainer_cfg.ckpt_dir,
    filename="Phoenix2014T-SLR-{epoch:02d}-{dev_loss:.2f}",
    monitor="dev_loss",
    mode="min",
    save_last=True,
)
trainer = Trainer(
    accelerator=trainer_cfg.accelerator,
    devices=trainer_cfg.devices,
    max_epochs=trainer_cfg.max_epochs,
    sync_batchnorm=True,
    strategy="ddp_find_unused_parameters_false",
    num_nodes=1,
    precision=trainer_cfg.precision,
    callbacks=[checkpoint_callback],
    logger=wandb_logger,
)

# * Train model
trainer.fit(slr_model, dm)

# * Evaluate model
dataset_cfg.batch_size = 2
eval_dm = VideoDataModule(gloss_dict=gloss_dict, **dataset_cfg)

val_trainer = Trainer(
    accelerator=trainer_cfg.accelerator,
    devices=1,
    max_epochs=trainer_cfg.max_epochs,
)

val_trainer.validate(model=slr_model, datamodule=eval_dm)
test_res = val_trainer.test(model=slr_model, datamodule=eval_dm)
for k in test_res[0]:
    wandb_logger.experiment.summary[k] = test_res[0][k]
