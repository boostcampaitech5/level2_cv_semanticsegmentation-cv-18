import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from lightning import LightningModule

import wandb

CLASSES = [
    "finger-1",
    "finger-2",
    "finger-3",
    "finger-4",
    "finger-5",
    "finger-6",
    "finger-7",
    "finger-8",
    "finger-9",
    "finger-10",
    "finger-11",
    "finger-12",
    "finger-13",
    "finger-14",
    "finger-15",
    "finger-16",
    "finger-17",
    "finger-18",
    "finger-19",
    "Trapezium",
    "Trapezoid",
    "Capitate",
    "Hamate",
    "Scaphoid",
    "Lunate",
    "Triquetrum",
    "Pisiform",
    "Radius",
    "Ulna",
]


def label2rgb(label):
    PALETTE = [
        (220, 20, 60),
        (119, 11, 32),
        (0, 0, 142),
        (0, 0, 230),
        (106, 0, 228),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 70),
        (0, 0, 192),
        (250, 170, 30),
        (100, 170, 30),
        (220, 220, 0),
        (175, 116, 175),
        (250, 0, 30),
        (165, 42, 42),
        (255, 77, 255),
        (0, 226, 252),
        (182, 182, 255),
        (0, 82, 0),
        (120, 166, 157),
        (110, 76, 0),
        (174, 57, 255),
        (199, 100, 0),
        (72, 0, 118),
        (255, 179, 240),
        (0, 125, 92),
        (209, 0, 151),
        (188, 208, 182),
        (0, 220, 176),
    ]

    image_size = label.shape[1:] + (3,)
    image = np.zeros(image_size, dtype=np.uint8)

    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]

    return image


class Module(LightningModule):
    def __init__(self, model: nn.Module, criterion: nn.Module, cfg: dict, fold_idx: int) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.cfg = cfg
        self.train_dice = []
        self.valid_dice = []
        self.val_batch = None
        self.fold_idx = fold_idx
        self.ex_val_dice = 0

    def dice_coef(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_true_f = y_true.flatten(2).cuda()
        y_pred_f = y_pred.flatten(2).cuda()
        intersection = torch.sum(y_true_f * y_pred_f, -1)

        eps = 0.0001
        return (2.0 * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg["optimizer"], params=self.parameters())
        if self.cfg["scheduler"] is not None:
            scheduler = instantiate(self.cfg["scheduler"], optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "name": "lr", "monitor": "Valid Dice"},
            }
        return optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        image, mask = batch
        image, mask = image.float(), mask.float()

        output = self.model(image)
        loss = self.criterion(output, mask)

        output = (torch.sigmoid(output) > self.cfg["threshold"]).detach().cpu()
        dice = self.dice_coef(mask, output)
        self.train_dice.append(dice)

        self.log("Train Loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        image, mask = batch
        image, mask = image.float(), mask.float()

        output = self.model(image)
        output_h, output_w = output.size(-2), output.size(-1)
        mask_h, mask_w = mask.size(-2), mask.size(-1)
        if output_h != mask_h or output_w != mask_w:
            output = F.interpolate(output, size=(mask_h, mask_w), mode="bilinear")
        loss = self.criterion(output, mask)

        output = (torch.sigmoid(output) > self.cfg["threshold"]).detach().cpu()
        dice = self.dice_coef(mask, output)
        self.valid_dice.append(dice)

        self.log("Valid Loss", loss)
        self.val_batch = (image[:2], mask[:2], output[:2])

    def on_validation_epoch_end(self):
        def print_dice(dice, step="train"):
            print(f"{step.capitalize()} step mean Dice Coefficient: {torch.mean(dice).item():.3f}")
            for index, (key, value) in enumerate(zip(CLASSES, dice), 1):
                print(f"{key:10}: {value.item():.3f}", end=" | ")
                if index % 4 == 0 or index == 29:
                    print()

        if self.train_dice and self.valid_dice:
            print(f"Epoch: {self.current_epoch}")
            train_dice = torch.mean(torch.cat(self.train_dice, 0), 0)
            valid_dice = torch.mean(torch.cat(self.valid_dice, 0), 0)

            print_dice(train_dice, step="train")
            print_dice(valid_dice, step="valid")

            self.log("Train Dice", torch.mean(train_dice).item())
            self.log("Valid Dice", torch.mean(valid_dice).item())
            print()

            images = []
            for image, mask, output in zip(*self.val_batch):
                _, ax = plt.subplots(1, 3, figsize=(18, 6))
                ax[0].imshow(image.detach().cpu().permute(1, 2, 0))
                ax[1].imshow(label2rgb(output.detach().cpu()))
                ax[2].imshow(label2rgb(mask.detach().cpu()))
                plt.tight_layout()
                images.append(wandb.Image(plt))
            wandb.log({"Prediction": images})
            plt.close()
            images = []
            self.val_batch = None

            if torch.mean(valid_dice).item() > self.ex_val_dice:
                print(
                    f"Best Dice Coefficient Renewal! ({self.ex_val_dice:.3f} -> {torch.mean(valid_dice).item():.3f})\n"
                )
                torch.save(
                    self.model,
                    f"./checkpoints/{self.cfg['exp_name']}/best-{self.fold_idx}fold.pt",
                )
                self.ex_val_dice = torch.mean(valid_dice).item()

        self.train_dice = []
        self.valid_dice = []
