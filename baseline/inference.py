import os
import cv2
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A

import numpy as np
import pandas as pd
import ttach as tta

# visualization
import matplotlib.pyplot as plt

from dataset import init_transform


CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

CLASS2IND = {v:i for i, v in enumerate(CLASSES)}
IND2CLASS = {v:k for k, v in CLASS2IND.items()}

IMAGE_ROOT = "../data/test/DCM/"

pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == '.png'
}


def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)


class XrayInferenceDataset(Dataset):
    def __init__(self, transforms=None):
        _filenames = pngs
        _filenames = np.array(sorted(_filenames))

        self.filenames = _filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(IMAGE_ROOT, image_name)

        image = cv2.imread(image_path)
        
        #####
        # image = np.asarray(image, dtype=np.uint8)
        image = image/255.

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]
        
        ####
        # image = image/255.

        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()

        return image, image_name


def test(model, data_loader, thr=0.5):
        model = model.cuda()
        model.eval()

        rles = []
        filename_and_class = []
        with torch.no_grad():
            n_class = 29

            for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
                images = images.cuda()    
                outputs = model(images)
                
                # restore original size
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > thr).detach().cpu().numpy()
                
                for output, image_name in zip(outputs, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                        
        return rles, filename_and_class


def inference(saved_dir, exp_name):
    
    model = torch.load(os.path.join(saved_dir, f"{exp_name}_best_model.pt"))

    tf = init_transform('base2') #A.Resize(512, 512)
    # tf = A.Compose([
    #     A.Resize(1024, 1024),
    #     A.CLAHE(clip_limit=4, p=0.5),
    #     A.RandomBrightnessContrast(),
    # ]) 


    ############## TTA #############
    # transform = tta.Compose([
    #     # tta.HorizontalFlip(),
    #     tta.Rotate90(angles=[0,90]),
    #     # tta.Scale()
    #     # tta.Multiply(factors=[0.9, 1, 1.1])
    # ])

    # tta_model = tta.SegmentationTTAWrapper(model, transform)
    ########################################

    test_dataset = XrayInferenceDataset(transforms=tf)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False, num_workers=2, drop_last=False)

    
    rles, filename_and_class = test(model, test_loader)

    preds = []
    for rle in rles[:len(CLASSES)]:
        pred = decode_rle_to_mask(rle, height=2048, width=2048)
        preds.append(pred)

    preds = np.stack(preds, 0)

    classes, filename = zip(*[x.split("_") for x in filename_and_class])

    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name" : image_name,
        "class" : classes,
        "rle" : rles
    })

    output_path = '../inference/'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    df.to_csv(f"../inference/{exp_name}_output.csv", index=False) #TTA


# if __name__=='__main__':
#     exp_name = '344_unet2plus_r152_Adam_dicefocal_bright_1e-3_CosineAnnealingLR_resized1024'
#     saved_dir = f'../checkpoints/result_{exp_name}'
    
#     inference(saved_dir, exp_name)

