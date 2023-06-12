import os

import albumentations as A
import hydra
import lightning as L
import torch.nn as nn
import wandb
from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from models.base_module import Module
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold

from data.new_data_module import DataModule, NewXRayDataset, preprocessing


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    L.seed_everything(cfg["seed"])
    os.makedirs(f"/opt/ml/directory/baseline/checkpoints/{cfg['exp_name']}", exist_ok=True)

    pngs, pkls = preprocessing(make=cfg["make_pickle"])

    groups = [os.path.dirname(fname) for fname in pngs]
    y = [0] * len(pngs)
    group_kfold = GroupKFold(n_splits=cfg["fold"])

    for fold_idx, (train_idx, valid_idx) in enumerate(group_kfold.split(pngs, y, groups)):
        train_data = (pngs[train_idx], pkls[train_idx])
        valid_data = (pngs[valid_idx], pkls[valid_idx])

        transforms = A.Compose([instantiate(aug) for _, aug in cfg["augmentation"].items()])

        train_dataset = NewXRayDataset(train_data, train=True, transforms=transforms)
        valid_dataset = NewXRayDataset(valid_data, train=True, transforms=transforms)

        datamodule = DataModule(train_dataset, valid_dataset, cfg)

        model = instantiate(cfg["model"]["model"])
        criterion = nn.BCEWithLogitsLoss()
        module = Module(model, criterion, cfg)
        exp_name = cfg["exp_name"] if fold_idx == 0 else f"{cfg['exp_name']}-{fold_idx}"
        logger = [WandbLogger(project="semantic-segmentation", name=f"{exp_name}", entity="cv-18", config=cfg)]
        callbacks = [
            RichProgressBar(),
            ModelCheckpoint(
                f"/opt/ml/directory/baseline/checkpoints/{cfg['exp_name']}",
                "best",
                monitor="Valid Dice",
                mode="max",
                save_last=True,
                save_weights_only=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="Valid Dice", patience=10, mode="max"),
        ]

        trainer = Trainer(max_epochs=cfg["epoch"], logger=logger, callbacks=callbacks, precision="16-mixed")
        trainer.fit(module, datamodule=datamodule)

        if not cfg["k-fold"]:
            break

        wandb.finish()


if __name__ == "__main__":
    main()
