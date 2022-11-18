import os
import sys
import argparse
import warnings
import pickle
from box import Box
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

sys.path.append("./")

from src.SLR.data import VideoDataModule
from src.SLR.models import SLR_Lightning

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # * Load config and dataset info
    parser = argparse.ArgumentParser(
        description="Train SLR model on specified dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/SLR/phoenix2014T-res18.yaml",
        help="path to the configuration file",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="W&B runner name",
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
    wandb_logger = WandbLogger(
        project="Sign Language Translation", entity="neu-ds5500-team13", name=args.name
    )
    try:
        wandb_logger.experiment.config.update(cfg.to_dict())
    except:
        pass

    checkpoint_callback = ModelCheckpoint(
        dirpath=trainer_cfg.ckpt_dir,
        filename=dataset_cfg.dataset_name + "-SLR-{epoch:02d}-{dev_loss:.2f}",
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
    val_trainer = Trainer(
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.devices,
        max_epochs=trainer_cfg.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    dev_res = val_trainer.validate(model=slr_model, datamodule=dm, ckpt_path="best")
    test_res = val_trainer.test(model=slr_model, datamodule=dm, ckpt_path="best")
    for k in test_res[0]:
        wandb_logger.experiment.summary[k] = test_res[0][k]
