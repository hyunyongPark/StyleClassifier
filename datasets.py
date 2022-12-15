import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset,DataLoader
import cv2

from config import CFG

#########################################

# albumentations for augs
import albumentations
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(CFG.img_size,CFG.img_size,always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
            albumentations.Normalize(mean = CFG.MEAN, std = CFG.STD),
            ToTensorV2(p=1.0)
        ]
    )

def get_valid_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(CFG.img_size,CFG.img_size, always_apply=True),
            albumentations.Normalize(mean = CFG.MEAN, std = CFG.STD),
            ToTensorV2(p=1.0)
        ]
    )

#########################################

class Style_dataset(Dataset):
    def __init__(self, img_path=CFG.img_path, csv_path=CFG.csv_path, transform=None, mode='train'):
        """
        Args:
            csv_path : csv 파일 경로
            img_path : image 디렉토리
            transform : optional transform
            mode : train | val
        """
        # train or val mode
        self.mode = mode

        # Load csv data
        if self.mode == "train":
            self.df = pd.read_csv(f'{csv_path}/multi_style_tr_sample.csv', index_col=0)
            self.image_arr = np.asarray(self.df.iloc[:, 0])
            self.label_arr = np.asarray(self.df.iloc[:, 1:])
        elif self.mode == "val":
            self.df = pd.read_csv(f'{csv_path}/multi_style_val.csv', index_col=0)
            self.image_arr = np.asarray(self.df.iloc[:, 0])
            self.label_arr = np.asarray(self.df.iloc[:, 1:])
        elif self.mode == "test":
            self.df = pd.read_csv(f'{csv_path}/avantgarde_style_test.csv', index_col=0)
            self.image_arr = np.asarray(self.df.iloc[:, 0])
            self.label_arr = np.asarray(self.df.iloc[:, 1:])
            
        # define images path
        self.img_path = img_path

        # transform 여부
        self.transform = transform

    def __len__(self):
        self.data_len = len(self.label_arr)
        return self.data_len

    def __getitem__(self, index):
        row = self.image_arr[index]  # csv이미지파일명 index선언
        label_row = self.label_arr[index]  # label값에 대해 index선언
        if self.mode == "train":
            image = cv2.imread(f'{CFG.img_path}/train/{row}')  # 해당 index에 대한 이미지 로드
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color차원 부여 -> tensor자료형
        elif self.mode == "val":
            image = cv2.imread(f'{CFG.img_path}/val/{row}')  # 해당 index에 대한 이미지 로드
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color차원 부여 -> tensor자료형
        elif self.mode == "test":
            image = cv2.imread(f'{CFG.img_path}/test/{row}')  # 해당 index에 대한 이미지 로드
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color차원 부여 -> tensor자료형
        
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label_row)

class Style_dataset_sp(Dataset):
    def __init__(self, img_path=CFG.img_path, csv_path=CFG.csv_path, transform=None, mode='train', sty="sexy"):
        """
        Args:
            csv_path : csv 파일 경로
            img_path : image 디렉토리
            transform : optional transform
            mode : train | val
        """
        # train or val mode
        self.mode = mode
        self.sty = sty
        # Load csv data
        if self.mode == "train":
            self.df = pd.read_csv(f'{csv_path}/multi_style_tr_sample.csv', index_col=0)
            self.image_arr = np.asarray(self.df.iloc[:, 0])
            self.label_arr = np.asarray(self.df.iloc[:, 1:])
        elif self.mode == "val":
            self.df = pd.read_csv(f'{csv_path}/multi_style_val.csv', index_col=0)
            self.image_arr = np.asarray(self.df.iloc[:, 0])
            self.label_arr = np.asarray(self.df.iloc[:, 1:])
        elif self.mode == "test":
            self.df = pd.read_csv(f'{csv_path}/{self.sty}', index_col=0)
            print(self.df.shape)
            self.image_arr = np.asarray(self.df.iloc[:, 0])
            self.label_arr = np.asarray(self.df.iloc[:, 1:])
            
        # define images path
        self.img_path = img_path

        # transform 여부
        self.transform = transform

    def __len__(self):
        self.data_len = len(self.label_arr)
        return self.data_len

    def __getitem__(self, index):
        row = self.image_arr[index]  # csv이미지파일명 index선언
        label_row = self.label_arr[index]  # label값에 대해 index선언
        if self.mode == "train":
            image = cv2.imread(f'{CFG.img_path}/train/{row}')  # 해당 index에 대한 이미지 로드
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color차원 부여 -> tensor자료형
        elif self.mode == "val":
            image = cv2.imread(f'{CFG.img_path}/val/{row}')  # 해당 index에 대한 이미지 로드
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color차원 부여 -> tensor자료형
        elif self.mode == "test":
            image = cv2.imread(f'{CFG.img_path}/{row}')  # 해당 index에 대한 이미지 로드
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # color차원 부여 -> tensor자료형
        
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label_row)