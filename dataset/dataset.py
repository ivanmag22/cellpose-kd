import torch
from torch.utils.data import Subset, DataLoader, Dataset
from PIL import Image
import torchvision
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tifffile as tiff
import skimage.io
import numpy as np

import random

import os
from pathlib import Path

# data augmentation transformations: consult TorchIO
# data augmentation
import torchvision.transforms as T

hflip_t = T.RandomHorizontalFlip(p=1)
vflip_t = T.RandomVerticalFlip(p=1)
hvflip_t = T.Compose([hflip_t, vflip_t])
rotate_t = T.RandomRotation(degrees=(0, 180))
# gauss_t = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))

augmentation_transforms = [hflip_t, vflip_t, hvflip_t, rotate_t]

classes = ["Chouxfleurs", "Compact", "Cystiques"]


class OrgDataset(Dataset):
    def __init__(
        self,
        base_root,
        mode="train",
        augmentation=False,
        num_samples=10,
        samples_per_class=False,
        train_test_ratio=2 / 3,
    ):
        """
        Args:
            base_root: where the samples are located
            mode: 'train' (training mode) or 'val' (validation mode)
            augmentation: True if we want to increase the number of samples by applying different trasformations, False if we want to deal with samples that we find in the folder
            train_test_ratio: how to split the samples, how many for training and how many for validation
        """
        super(OrgDataset, self).__init__()

        self.image_paths = []  # images
        self.augmentation = augmentation
        self.mode = mode

        assert os.path.isdir(base_root)  # it checks if it is a right folder or not

        self.image_paths = []
        if samples_per_class:
            # take the same number of samples per each class
            d, length = imgs_in_subfolders_per_class(base_root, max_images=num_samples)
            num = int(min(length, num_samples))
            n_items = num * train_test_ratio
            last_item = 0
            if not n_items.is_integer():
                last_item = 1
            for i, key in enumerate(d.keys()):
                t_n, v_n = 0, 0
                if i < len(d.keys()) - 1:
                    t_n, v_n = int(n_items), int(n_items) - num
                else:
                    t_n, v_n = (
                        int(n_items) + last_item,
                        (int(n_items) + last_item) - num,
                    )
                if mode == "train":
                    self.image_paths.extend(d.get(key)[:t_n])
                else:
                    self.image_paths.extend(d.get(key)[v_n:])
        else:
            # take a number of samples independently by the class
            samples = imgs_in_subfolders(base_root, max_images=num_samples)
            n_items = int(num_samples * train_test_ratio)
            if mode == "train":
                self.image_paths.extend(samples[:n_items])
            else:
                self.image_paths.extend(samples[n_items:])

        assert len(self.image_paths) != 0

    def __getitem__(self, index):

        image = tiff.imread(self.image_paths[index])  # path -> array
        if image.ndim == 4 and image.shape[-1] == 3:
            # if 3D mask and RGB image
            tp = image.dtype
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(
                tp
            )  # it casts a RGB image into a grayscale one
        image = image.squeeze()

        return image

    def __len__(self):
        return len(self.image_paths)


"""
def files_in_subfolders(root_dir, format=".tif", max_images=10):  # tif
    d = {}
    unique_d = {}
    value = 999999

    # Iterate through subfolders in the root directory
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)

        if not any(ele in subfolder_path for ele in classes):
            continue

        # Check if it is a directory
        if os.path.isdir(subfolder_path):
            unique_images = set()  # Track unique .tif image names
            files = os.listdir(subfolder_path)
            samples = []

            for file in files:
                if file.split(".")[-1] in format:
                    # Extract the .tif base name
                    tif_base_name = file
                    if "_patch_" in file:
                        tif_base_name = file.split("_patch_")[0]
                    elif "_sliced" in file:
                        tif_base_name = file.split("_sliced")[0]

                    image_path = os.path.abspath(os.path.join(subfolder_path, file))

                    # Check if we already collected enough unique .tif images
                    if tif_base_name not in unique_images:
                        if len(unique_images) == max_images:
                            break
                        unique_images.add(tif_base_name)
                        samples.append(image_path)
                    elif tif_base_name in unique_images:
                        # Add patches only if the base image is already in unique_images
                        samples.append(image_path)

            if (
                samples
            ):  # not necessary for labels, for example training samples don't have labels
                d.update({subfolder: samples})
                unique_d.update({subfolder: len(unique_images)})
                value = min(value, len(files))

    # it checks if all the classes contain the same number of images (not patches)
    first_value = list(unique_d.values())[0]
    for key in unique_d.keys():
        assert unique_d.get(key) == first_value

    return d, value
"""


def list_files_recursive(path="."):
    files = []

    path = os.path.abspath(path)
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            files.extend(list_files_recursive(full_path))
        else:
            files.append(os.path.abspath(full_path))

    return files


def imgs_in_subfolders(root_dir, max_images=10):
    samples = []

    files = list_files_recursive(root_dir)
    indeces = np.random.permutation(len(files))[:max_images]
    samples = [files[i] for i in indeces][:max_images]

    return samples


def imgs_in_subfolders_per_class(root_dir, max_images=10):
    d = {}
    unique_d = {}

    # Iterate through subfolders in the root directory
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)

        if not any(ele in subfolder_path for ele in classes):
            continue

        # Check if it is a directory
        if os.path.isdir(subfolder_path):
            files = os.listdir(subfolder_path)
            samples = []

            for file in files:
                if len(samples) == max_images:
                    break

                image_path = os.path.abspath(os.path.join(subfolder_path, file))
                samples.append(image_path)

            d.update({subfolder: samples})
            unique_d.update({subfolder: len(samples)})
    # it checks if all the classes contain the same number of images
    first_value = list(unique_d.values())[0]
    for key in unique_d.keys():
        assert unique_d.get(key) == first_value

    return d, first_value


class OrgDataset_patches(Dataset):
    def __init__(
        self,
        base_root,
        mode="train",
        augmentation=False,
        num_samples=300,
        samples_per_class=False,
        train_test_ratio=2 / 3,
    ):
        """
        Args:
            base_root: where the samples are located
            mode: 'train' (training mode) or 'val' (validation mode)
            augmentation: True if we want to increase the number of samples by applying different trasformations, False if we want to deal with samples that we find in the folder
            train_test_ratio: how to split the samples, how many for training and how many for validation
        """
        super(OrgDataset_patches, self).__init__()

        self.mode = mode
        self.image_paths = []  # images
        self.augmentation = augmentation
        self.image_t = T.ToTensor()

        assert os.path.isdir(base_root)  # it checks if it is a right folder or not
        assert mode == "train" or mode == "val"

        self.image_paths = []
        self.label_paths = []

        if samples_per_class:
            # take the same number of samples per each class
            d, length = imgs_in_subfolders(base_root, max_images=num_samples)

            num = int(min(length, num_samples))
            n_items = num * train_test_ratio
            last_item = 0
            if not n_items.is_integer():
                last_item = 1
            for i, key in enumerate(d.keys()):
                t_n, v_n = 0, 0
                if i < len(d.keys()) - 1:
                    t_n, v_n = int(n_items), int(n_items) - num
                else:
                    t_n, v_n = (
                        int(n_items) + last_item,
                        (int(n_items) + last_item) - num,
                    )
                if mode == "train":
                    self.image_paths.extend(d.get(key)[:t_n])
                else:
                    self.image_paths.extend(d.get(key)[v_n:])
        else:
            # take a number of samples independently by the class
            samples = imgs_in_subfolders(base_root, max_images=num_samples)
            n_items = int(num_samples * train_test_ratio)
            if mode == "train":
                self.image_paths.extend(samples[:n_items])
            else:
                self.image_paths.extend(samples[n_items:])

        assert len(self.image_paths) != 0

    def __getitem__(self, index):
        image = skimage.io.imread(self.image_paths[index], as_gray=True)
        img = Image.fromarray(image)
        image = self.image_t(img).squeeze()

        if self.mode == "train" and self.augmentation and random.choice([True, False]):
            idx = random.randint(0, len(augmentation_transforms) - 1)

            img = augmentation_transforms[idx](img)

        return image

    def __len__(self):
        return len(self.image_paths)


class OrgDataset_slices(Dataset):
    def __init__(
        self,
        base_root,
        mode="train",
        augmentation=False,
        num_samples=3,
        samples_per_class=False,
        size=1024,
        train_test_ratio=2 / 3,
    ):
        """
        Args:
            base_root: where the samples are located
            mode: 'train' (training mode) or 'val' (validation mode)
            augmentation: True if we want to increase the number of samples by applying different trasformations, False if we want to deal with samples that we find in the folder
            train_test_ratio: how to split the samples, how many for training and how many for validation
        """
        super(OrgDataset_slices, self).__init__()

        self.mode = mode
        self.image_paths = []  # images
        self.augmentation = augmentation
        self.image_t = T.Compose(
            [
                T.Resize((size, size), T.InterpolationMode.BILINEAR),
                T.ToTensor(),
            ]
        )

        assert os.path.isdir(base_root)  # it checks if it is a right folder or not
        assert mode == "train" or mode == "val"

        self.image_paths = []

        if samples_per_class:
            # take the same number of samples per each class
            d, length = imgs_in_subfolders(base_root, max_images=num_samples)

            num = int(min(length, num_samples))
            n_items = num * train_test_ratio
            last_item = 0
            if not n_items.is_integer():
                last_item = 1
            for i, key in enumerate(d.keys()):
                t_n, v_n = 0, 0
                if i < len(d.keys()) - 1:
                    t_n, v_n = int(n_items), int(n_items) - num
                else:
                    t_n, v_n = (
                        int(n_items) + last_item,
                        (int(n_items) + last_item) - num,
                    )
                if mode == "train":
                    self.image_paths.extend(d.get(key)[:t_n])
                else:
                    self.image_paths.extend(d.get(key)[v_n:])
        else:
            # take a number of samples independently by the class
            samples = imgs_in_subfolders(base_root, max_images=num_samples)
            n_items = int(num_samples * train_test_ratio)
            if mode == "train":
                self.image_paths.extend(samples[:n_items])
            else:
                self.image_paths.extend(samples[n_items:])

        assert len(self.image_paths) != 0

    def __getitem__(self, index):
        image = skimage.io.imread(self.image_paths[index], as_gray=True)
        img = Image.fromarray(image)
        image = self.image_t(img).squeeze()

        if self.mode == "train" and self.augmentation and random.choice([True, False]):
            idx = random.randint(0, len(augmentation_transforms) - 1)

            img = augmentation_transforms[idx](img)

        return image

    def __len__(self):
        return len(self.image_paths)
