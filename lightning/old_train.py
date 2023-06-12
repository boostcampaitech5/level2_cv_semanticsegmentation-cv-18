import os

import albumentations as A
import hydra
import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from models.base_module import Module
from models.components.fcn import FCN
from omegaconf import DictConfig

import wandb
from data.data_module import DataModule


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    L.seed_everything(cfg["seed"])
    os.makedirs(f"/opt/ml/directory/baseline/checkpoints/{cfg['exp_name']}", exist_ok=True)

    transforms = A.Compose([instantiate(aug) for _, aug in cfg["augmentation"].items()])
    datamodule = DataModule(num_workers=4, transforms=transforms)

    model = instantiate(cfg["model"])
    criterion = nn.BCEWithLogitsLoss()
    module = Module(model, criterion, cfg)

    logger = [WandbLogger(project="semantic-segmentation", name=str(cfg["exp_name"]), entity="cv-18", config=cfg)]
    callbacks = [
        RichProgressBar(),
        ModelCheckpoint(
            f"/opt/ml/directory/baseline/checkpoints/{cfg['exp_name']}",
            "best",
            monitor="Valid Dice",
            mode="max",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="Valid Dice", patience=5, mode="max"),
    ]

    trainer = Trainer(max_epochs=cfg["epoch"], logger=logger, callbacks=callbacks, precision="16-mixed")
    trainer.fit(module, datamodule=datamodule)

    wandb.finish()


if __name__ == "__main__":
    main()
