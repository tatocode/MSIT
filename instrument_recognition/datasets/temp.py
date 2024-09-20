import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import albumentations as A
import cv2

class InstrumentDataset(Dataset):
    def __init__(self, data_root: str, dataset_type: str, is_test=False, transform=None):
        if is_test:
            file_path = os.path.join(data_root, 'test.txt')
        else:
            file_path = os.path.join(data_root, f'train.txt' if dataset_type=='train' else f'val.txt')
        with open(file_path, 'r') as f:
            context = f.readlines()
        self.data = context
        self.transform = transform
        self.data_root = data_root
        self.std = [0.229, 0.224, 0.225]
        self.mean = [0.485, 0.456, 0.406]
        self.T_image = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=self.mean, std=self.std)
                ])

    def __getitem__(self, item):
        info = self.data[item].strip()
        # print(info)

        path = ''
        for i in info.split(' ')[:-1]:
            path += i+' '
        path = path.strip()

        label = info.split(' ')[-1]

        path = path.replace('/mnt/c/Users/Tatocode/Desktop/classification/new/', r'C:\Users\Tatocode\Documents\desk\dataset\tool_recognition\\')

        image = cv2.imread(os.path.join(self.data_root, path), 1)
        return self.T_image(self.transform(image=image)['image']), int(label)

    def __len__(self):
        return len(self.data)