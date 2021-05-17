import os
from multiprocessing.dummy import freeze_support

from PIL import Image
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import numpy as np
import torchvision.utils as vutils
from torch.utils.data import Dataset
from natsort import natsorted


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


def LR_Source_Dataloader(image_size, device, batch):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # transforms.Normalize((0, 0, 0), (1, 1, 1))
    ])

    path_to_source_data = 'D:\DataSets\DPEDiphone-tr-x'
    source_dataset = CustomDataSet(path_to_source_data, transform=transform)

    print(len(source_dataset))
    # batch size cnt pics
    source_dataloader = torch.utils.data.DataLoader(source_dataset, batch_size=batch, shuffle=True, num_workers=1)
    print(len(source_dataloader))
    real_batch = next(iter(source_dataloader))
    print(real_batch[::].shape)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[::].to(device), padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
    return real_batch, source_dataloader


def HR_Source_Dataloader(image_size, device, batch):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # transforms.Normalize((0, 0, 0), (1, 1, 1))
    ])

    path_to_source_data = 'D:\DataSets\DPEDiphone-tr-y'
    source_dataset = CustomDataSet(path_to_source_data, transform=transform)

    print(len(source_dataset))
    # batch size cnt pics
    source_dataloader = torch.utils.data.DataLoader(source_dataset, batch_size=batch, shuffle=True, num_workers=2)
    print(len(source_dataloader))
    real_batch = next(iter(source_dataloader))
    print(real_batch[::].shape)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[::].to(device), padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
    return real_batch, source_dataloader


def Downscaled_DataLoader(image_size, device, batch):
    # transform = transforms.ToTensor()
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # transforms.Normalize((0, 0, 0), (1, 1, 1))
    ])

    path_to_source_data = 'D:\DataSets\imagetrain'
    source_dataset = CustomDataSet(path_to_source_data, transform=transform)
    source_dataloader = torch.utils.data.DataLoader(source_dataset, batch_size=batch, shuffle=True, num_workers=1)
    print(len(source_dataloader))
    real_batch = next(iter(source_dataloader))
    print(real_batch[::].shape)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[::].to(device), padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
    return real_batch, source_dataloader
