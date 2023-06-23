# python native
import os
import json
import random
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models


IMAGE_ROOT = "../data/train/DCM"
LABEL_ROOT = "../data/train/outputs_json"

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v: i for i, v in enumerate(CLASSES)}

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}

pngs = sorted(pngs)
jsons = sorted(jsons)


def init_transform(name):
    transform_dict = {
        "base" : A.Resize(512, 512),
        "base2" : A.Resize(1024, 1024),
        "clahe1" : A.Compose([
            A.Resize(1024, 1024),
            A.CLAHE(clip_limit=4, p=1.0),  #1.0
        ]),
        "clahe2" : A.Compose([
            A.Resize(512, 512),
            A.CLAHE(clip_limit=4, p=0.5), 
        ]),
        "clahe3" : A.Compose([
            A.Resize(1024, 1024),
            A.CLAHE(clip_limit=4, p=0.5), 
        ]),
        "clahe4" : A.Compose([
            A.Resize(512, 512),
            A.CLAHE(clip_limit=8, p=1.0), 
        ]),
        "bright" : A.Compose([
            A.Resize(1024, 1024),
            A.RandomBrightnessContrast(brightness_limit=0.2),
        ]),
        "bright2" : A.Compose([
            A.Resize(512, 512),
            A.RandomBrightnessContrast(brightness_limit=0.2)
        ]),
        "rotate" : A.Compose([
            A.Resize(512, 512),
            A.RandomRotate90()
        ]),
        "vertical" : A.Compose([
            A.Resize(512, 512),
            A.VerticalFlip()
        ]),
        'brightclahe': A.Compose([
            A.Resize(512, 512), 
            A.RandomBrightnessContrast(brightness_limit=0.2),
            A.CLAHE(clip_limit=4),
        ]),
        'brightclaherotate': A.Compose([
            A.Resize(512, 512), 
            A.RandomBrightnessContrast(brightness_limit=0.2),
            A.CLAHE(clip_limit=4),
            A.RandomRotate90()

        ]),
        "gridmask" : A.Compose([
            A.Resize(512, 512),
            A.GridDropout(ratio=0.2, random_offset=True, holes_number_x=4, holes_number_y=4, p=1.0),
        ]),
        "cropnonempty" : A.Compose([
            A.Resize(512, 512),
            A.augmentations.crops.transforms.CropNonEmptyMaskIfExists(height=256, width=256),
        ]),
        "cropnonempty" : A.Compose([
            A.Resize(512, 512),
            A.augmentations.crops.transforms.CropNonEmptyMaskIfExists(height=256, width=256),
        ]),
        "cropnonempty2" : A.Compose([
            A.Resize(512, 512),
            A.augmentations.crops.transforms.CropNonEmptyMaskIfExists(height=464, width=464),
        ]),
        "grid_empty" : A.Compose([
            A.Resize(512, 512),
            A.GridDropout(ratio=0.2, random_offset=True, holes_number_x=4, holes_number_y=4, p=1.0),
            A.augmentations.crops.transforms.CropNonEmptyMaskIfExists(height=256, width=256),
        ]),
        "centercrop" : A.Compose([
            A.Resize(512, 512),
            A.CenterCrop(352, 352)
        ]),
        "centercrop2" : A.Compose([
            A.Resize(512, 512),
            A.CenterCrop(464, 464)
        ]),
        "center_grid" : A.Compose([
            A.Resize(512, 512),
            A.CenterCrop(464,464),
            A.GridDropout(ratio=0.2, random_offset=True, holes_number_x=4, holes_number_y=4, p=1.0),
        ]),
        "center_empty" : A.Compose([
            A.Resize(512, 512),
            A.CenterCrop(352, 352),
            A.augmentations.crops.transforms.CropNonEmptyMaskIfExists(height=256, width=256),
        ]),
        
    }

    return transform_dict[name]


# For Copy&Paste
from PIL import Image, ImageDraw
import random

class XRayTrainDataset(Dataset):
    def __init__(self, is_train=True, transforms=None, copypaste=True, k=3):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        self.copypaste = copypaste
        self.k = k
          
        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)
    
    def get_coord(self, polygon):
        polygon = polygon
        for i in range(len(polygon)):
            polygon[i] = tuple(polygon[i])
        polygon_np = np.array(polygon)
        max = np.max(polygon_np, axis=0)
        min = np.min(polygon_np, axis=0)
        return max, min
        
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        # clahe, randombrightnesscontrast 사용시
        image = np.asarray(image, dtype=np.uint8)
        # image = image / 255.        
        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])

            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        if self.copypaste:
            randoms = random.choices([i for i in range(640)], k=self.k)
            for i in randoms:
                # clahe, randombrightcontrast 사용시 
                target_image = cv2.imread(os.path.join(IMAGE_ROOT, self.filenames[i])) #/ 255.
                target_image = np.asarray(target_image, dtype=np.uint8)
                target_label_path = os.path.join(LABEL_ROOT, self.labelnames[i])

                with open(target_label_path, "r") as f:
                    target_annotations = json.load(f)
                target_annotations = target_annotations["annotations"]

                for ann in target_annotations:
                    target_c = ann["label"]
                    target_c = CLASS2IND[target_c]
                    if target_c == 26 or target_c == 19 or target_c == 20 or target_c == 25:
                        points = np.array(ann["points"])
                        max, min = self.get_coord(ann['points'])
                        x = random.randint(100,1800)
                        y = random.randint(100,1800)
                        alpha = random.randint(25,50)

                        # 0. check whether generated (x,y) coordinate is in the background 
                        bone_area_x = [i for i in range(400,1600)]
                        while x in bone_area_x:
                            x = random.randint(100,1800)
                        x -= alpha
                        
                        # 1. create mask for new image
                        img = Image.new('L', target_image.shape[:2], 0)
                        ImageDraw.Draw(img).polygon(ann['points'], outline=0, fill=1)
                        mask = np.array(img)
                        
                        # 2. paste maskout poly to source image
                        new_image = cv2.bitwise_or(target_image, target_image, mask=mask)
                        image[y:y+max[1]-min[1], x:x+max[0]-min[0], ...] = new_image[min[1]:max[1], min[0]:max[0], ...]

                        # 3. update label
                        ori_label = label[..., target_c]
                        ori_label[y:y+max[1]-min[1], x:x+max[0]-min[0]] = mask[min[1]:max[1], min[0]:max[0]]
                        label[..., target_c] = ori_label
        

        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label
        
        image = image / 255. # 전처리

        image = image.transpose(2, 0, 1)    
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label



class XRayDataset(Dataset):
    def __init__(self, is_train=True, transforms=None):
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]
        
        # dummy label
        ys = [0 for fname in _filenames]
        
        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)
        
        filenames = []
        labelnames = []
        for i, (x, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                # 0번을 validation dataset으로 사용합니다.
                if i == 0:
                    continue
                    
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])
            
            else:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                
                # skip i > 0
                break
        
        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)
        
        image = cv2.imread(image_path)
        image = np.asarray(image, dtype=np.uint8)

        label_name = self.labelnames[item]
        label_path = os.path.join(LABEL_ROOT, label_name)
        
        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(CLASSES), )
        label = np.zeros(label_shape, dtype=np.uint8)
        
        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]
        
        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = CLASS2IND[c]
            points = np.array(ann["points"])
            
            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1) # polygon 내부 채워주는 함수
            label[..., class_ind] = class_label
        
        if self.transforms is not None:
            inputs = {"image": image, "mask": label} if self.is_train else {"image": image}
            result = self.transforms(**inputs)
            
            image = result["image"]
            label = result["mask"] if self.is_train else label

        image = image / 255. # 전처리

        # to tenser will be done later
        image = image.transpose(2, 0, 1)    # make channel first
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
            
        return image, label