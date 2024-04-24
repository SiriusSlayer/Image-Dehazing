import os
import torch
import numpy as np
from PIL import Image as Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
from torchvision.transforms import Resize,ToTensor
ImageFile.LOAD_TRUNCATED_IMAGES = True

def collate_fn(batch):
    # new_size = (500, 500)
    # transform = Resize(new_size)

    # inputs = []
    # labels = []

    # for item in batch:
    #     img, label, _ = item
    #     # img, label = item
    #     img = ToTensor()(img)  # Convert PIL image to tensor
    #     img = transform(img)  # Resize the tensor
    #     inputs.append(img)
    #     labels.append(label)

    # # Stack all inputs and labels along a new dimension to create a batch
    # inputs = torch.stack(inputs, dim=0)
    # labels = torch.stack(labels, dim=0)
    new_size = (500, 500)
    transform = Resize(new_size)

    inputs = []
    labels = []
    names=[]

    for item in batch:
        img, label,_ = item
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Expected img to be a PyTorch tensor, but got {type(img)}")
        img = transform(img)  # Resize the tensor
        label = transform(label)  # Resize the tensor
        inputs.append(img)
        labels.append(label)
        names.append(_)

    # Stack all inputs and labels along a new dimension to create a batch
    inputs = torch.stack(inputs, dim=0)
    labels = torch.stack(labels, dim=0)
    # names =  torch.stack(names, dim=0)
    # print(type(inputs[0]),labels[0])

    return inputs, labels,names
def train_dataloader(path, batch_size=64, num_workers=0):
    image_dir = os.path.join(path, 'train')

    dataloader = DataLoader(
        DeblurDataset(image_dir, ps=256),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Add this line
        )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
        
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test'), is_valid=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader

import random
class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, is_valid=False, ps=None):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.is_valid = is_valid
        self.ps = ps
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx])).convert('RGB')
        if self.is_valid or self.is_test:      
            label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'_outdoor_GT.jpg')).convert('RGB')
            # label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'.png')).convert('RGB')
        else:
            label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'_outdoor_GT.jpg')).convert('RGB')
            # label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'.jpg')).convert('RGB')
        
        ps = self.ps

        if self.ps is not None:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

            hh, ww = label.shape[1], label.shape[2]

            rr = random.randint(0, hh-ps)
            cc = random.randint(0, ww-ps)
            
            image = image[:, rr:rr+ps, cc:cc+ps]
            label = label[:, rr:rr+ps, cc:cc+ps]

            if random.random() < 0.5:
                image = image.flip(2)
                label = label.flip(2)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label



    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
