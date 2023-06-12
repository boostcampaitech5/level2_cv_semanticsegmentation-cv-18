import json
import os
import pickle
import time

import cv2
import lightning as L
import numpy as np
import ray
import torch
from torch.utils.data import DataLoader, Dataset

IMAGE_ROOT = "/opt/ml/data/train/DCM"
LABEL_ROOT = "/opt/ml/data/train/outputs_json"
MASK_ROOT = "/opt/ml/data/train/mask"

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

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}


@ray.remote
def save_mask(jsons):
    LABEL_SHAPE = (2048, 2048, 29)

    label = np.zeros(LABEL_SHAPE, dtype=np.uint8)

    for label_name in jsons:
        label_path = os.path.join(LABEL_ROOT, label_name)
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # polygon to mask
            class_label = np.zeros(LABEL_SHAPE[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        mask_path, mask_name = label_name.split("/")
        os.makedirs(os.path.join(MASK_ROOT, mask_path), exist_ok=True)
        with open(os.path.join(MASK_ROOT, f"{mask_path}/{mask_name.split('.')[0]}.pkl"), mode="wb") as file:
            pickle.dump(np.packbits(label), file)


class NewXRayDataset(Dataset):
    def __init__(self, data, train=True, transforms=None) -> None:
        super().__init__()
        self.images, self.label = data
        self.train = train
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(IMAGE_ROOT, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0

        label_name = self.label[index]
        with open(os.path.join(MASK_ROOT, label_name), mode="rb") as file:
            label = np.unpackbits(pickle.load(file)).reshape((2048, 2048, 29))

        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.train else {"image": image}
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"] if self.train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  # make channel first
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label


class DataModule(L.LightningDataModule):
    def __init__(self, train_dataset, valid_dataset, cfg) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.cfg = cfg

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=self.cfg["num_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )


def preprocessing(make=True):
    os.makedirs(MASK_ROOT, exist_ok=True)
    pngs = {
        os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
        for root, _, files in os.walk(IMAGE_ROOT)
        for fname in files
        if fname.endswith(".png")
    }
    jsons = {
        os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
        for root, _, files in os.walk(LABEL_ROOT)
        for fname in files
        if fname.endswith(".json")
    }
    pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}
    jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}

    assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
    assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

    pngs = np.array(sorted(pngs))
    jsons = np.array(sorted(jsons))

    if make:
        ray.init()

        result = [save_mask.remote(json_list) for json_list in np.split(jsons, 8)]
        ray.wait(result)

        time.sleep(5)

    pkls = {
        os.path.relpath(os.path.join(root, fname), start=MASK_ROOT)
        for root, _, files in os.walk(MASK_ROOT)
        for fname in files
        if fname.endswith(".pkl")
    }
    pkls_fn_prefix = {os.path.splitext(fname)[0] for fname in pkls}

    assert len(pkls_fn_prefix - pngs_fn_prefix) == 0

    pkls = np.array(sorted(pkls))

    return pngs, pkls
