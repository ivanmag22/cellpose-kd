import sys
import os

# Add the parent Implementation directory to sys.path
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(1, os.getcwd())
# print(sys.path)

import argparse
import logging
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import subprocess

import skimage
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2

from torch.utils.data import Subset, DataLoader, Dataset
from sklearn.model_selection import KFold

from cellpose.models import Cellpose

from dataset.dataset import OrgDataset, OrgDataset_patches, OrgDataset_slices

from train_kd_batch import train_kd, val_iou


def main(args):
    (
        device,
        bs,
        n_epochs,
        epoch_start_i,
        checkp_step,
        val_step,
        lr,
        optim,
        saved_model_dir,
        samples_root,
        diam,
        s_path,
        t_path,
        s_dilation,
        s_upsample_compr,
        data_aug,
        loss_func,
        display,
    ) = (
        torch.device(args.device),
        args.batch_size,
        args.num_epochs,
        args.epoch_start_i,
        args.checkpoint_step,
        args.validation_step,
        args.lr,
        args.optimizer,
        args.saved_model_dir,
        args.samples_root,
        args.diameter,
        args.s_path,
        args.t_path,
        args.s_dilation,
        args.s_upsample_compr,
        args.data_aug,
        args.loss_func,
        args.display,
    )

    # Set the random seed for reproducible experiments
    random.seed(230)
    torch.manual_seed(230)
    if device.type == "cuda":
        torch.cuda.manual_seed(230)

    ## dataset and dataloader
    if args.patches:
        train_dataset = OrgDataset_patches(
            samples_root,
            augmentation=data_aug,
            num_samples=args.num_samples,
            samples_per_class=args.samples_per_class,
            mode="train",
        )
        test_dataset = OrgDataset_patches(
            samples_root,
            augmentation=data_aug,
            num_samples=args.num_samples,
            samples_per_class=args.samples_per_class,
            mode="val",
        )
    elif args.slices:
        train_dataset = OrgDataset_slices(
            samples_root,
            augmentation=data_aug,
            num_samples=args.num_samples,
            samples_per_class=args.samples_per_class,
            mode="train",
        )
        test_dataset = OrgDataset_slices(
            samples_root,
            augmentation=data_aug,
            num_samples=args.num_samples,
            samples_per_class=args.samples_per_class,
            mode="val",
        )
    else:
        train_dataset = OrgDataset(
            samples_root,
            augmentation=data_aug,
            num_samples=args.num_samples,
            samples_per_class=args.samples_per_class,
            size=args.image_size,
            mode="train",
        )
        test_dataset = OrgDataset(
            samples_root,
            augmentation=data_aug,
            num_samples=args.num_samples,
            samples_per_class=args.samples_per_class,
            mode="val",
        )

    kf = KFold(n_splits=args.folds, shuffle=True)

    # acc_dict = {}

    for lr in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        lr_acc_list = []
        print(f"Learning rate {lr:.2e}\n---------")
        for bs in [8]:  # [8, 16, 24]:
            bs_acc_list = []
            print(f"Batch Size {bs}\n---------")
            for d in [17, 25, 40, 60]:
                diam_acc_list = []
                # acc_list = []
                print(f"Diameter {d}\n---------")
                for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
                    print(f"Fold {fold + 1}\n-------------")

                    dataloader_train = DataLoader(
                        train_dataset,
                        batch_size=bs,
                        # shuffle=True,
                        drop_last=True,
                        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
                        # num_workers=4,
                        # pin_memory=True,
                    )
                    dataloader_val = DataLoader(
                        train_dataset,
                        batch_size=bs,
                        # shuffle=True,
                        drop_last=True,
                        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
                        # num_workers=4,
                        # pin_memory=True,
                    )

                    ## model
                    # teacher
                    t_model = Cellpose(
                        model_type="nuclei",
                        gpu=True if device.type == "cuda" else False,
                        device=device,
                    )
                    if t_path:
                        t_model.cp.net.load_state_dict(
                            torch.load(t_path, map_location=device)
                        )
                    t_model.cp.net.eval()

                    # student
                    s_model = Cellpose(
                        model_type="nuclei",
                        dilation=s_dilation,
                        upsample_compr=s_upsample_compr,
                        gpu=True if device.type == "cuda" else False,
                        device=device,
                    )
                    if s_path:
                        s_model.cp.net.load_state_dict(
                            torch.load(s_path, map_location=device)
                        )

                    ## optimizer
                    # build optimizer
                    if optim == "rmsprop":
                        optimizer = torch.optim.RMSprop(s_model.cp.net.parameters(), lr)
                    elif optim == "sgd":
                        optimizer = torch.optim.SGD(
                            s_model.cp.net.parameters(),
                            lr,
                            momentum=0.9,
                            weight_decay=1e-4,
                        )
                    elif optim == "adam":
                        optimizer = torch.optim.Adam(s_model.cp.net.parameters(), lr)
                    else:
                        print("not supported optimizer \n")
                        return None

                    ## load a saved model from start epoch
                    assert os.path.isdir(os.path.abspath(saved_model_dir))

                    if not os.path.exists(
                        os.path.abspath(
                            os.path.join(saved_model_dir, f"d{d}", f"f{fold+1}")
                        )
                    ):
                        if not os.path.exists(
                            os.path.abspath(os.path.join(saved_model_dir, f"d{d}"))
                        ):
                            os.makedirs(
                                os.path.abspath(os.path.join(saved_model_dir, f"d{d}"))
                            )
                        os.makedirs(
                            os.path.abspath(
                                os.path.join(saved_model_dir, f"d{d}", f"f{fold+1}")
                            )
                        )

                    if torch.cuda.is_available() and device.type == "cuda":
                        s_model.cp.net = torch.nn.DataParallel(s_model.cp.net).cuda()
                        t_model.cp.net = torch.nn.DataParallel(t_model.cp.net).cuda()
                    elif device.type == "cpu":
                        s_model.cp.net = torch.nn.DataParallel(s_model.cp.net)
                        t_model.cp.net = torch.nn.DataParallel(t_model.cp.net)

                    acc = train_kd(
                        s_model,
                        t_model,
                        lr,
                        optimizer,
                        loss_func,
                        dataloader_train,
                        dataloader_val,
                        n_epochs,
                        bs,
                        epoch_start_i,
                        device,
                        checkp_step,
                        val_step,
                        os.path.join(saved_model_dir, f"d{d}", f"f{fold+1}"),
                        d,
                        display,
                    )
                    val_acc, _, _ = val_iou(
                        s_model,
                        t_model,
                        dataloader=dataloader_val,
                        loss_func=loss_func,
                        device=device,
                        bs=bs,
                        diam=d,
                        display=display,
                    )
                    print(f"Diameter={d}, fold={fold+1}, IoU={val_acc}")
                    diam_acc_list.append(val_acc)
                print(
                    f"Diameter={d}, Learning Rate={lr:.1e}, Batch Size={bs}, IoU={np.mean(diam_acc_list):.2f}"
                )
                bs_acc_list.append(np.mean(diam_acc_list))
            lr_acc_list.append(np.mean(bs_acc_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device where to run the code ('cpu'/'cuda')",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of epochs of training"
    )
    parser.add_argument(
        "--epoch_start_i",
        type=int,
        default=0,
        help="Resume the training from this epoch and loading the last saved model",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=1,
        help="It is useful to save the model weights at the end of n training epochs, in order to not lose everything if we lose the session during the training",
    )
    parser.add_argument(
        "--validation_step",
        type=int,
        default=None,
        help="It tells the frequency to do validation after some training epochs, like validation step after n training epochs; useful to record the best model found till that moment",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="STUDENT: Learning Rate")
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="STUDENT: Optimizer"
    )
    parser.add_argument(
        "--loss_func",
        type=str,
        default="new",
        help="Loss function: 'default' from train.py (lambda=5), 'new' for CP loss function with lambda=1, 'bce' for binary cross entropy loss. Default: 'new'",
    )
    parser.add_argument(
        "--saved_model_dir",
        type=str,
        default=None,
        help="STUDENT: Path for saved model directory, it contains saved models from each checkpoint",
    )
    parser.add_argument(
        "--samples_root", type=str, default=None, help="Path for the samples folder"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=300,
        help="Number of images to load. Default: 300",
    )
    parser.add_argument(
        "--samples_per_class",
        action="store_true",
        help="Pick num_samples per each class",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1024,
        help="Image size (only a side) (useful to load images with different size through torchvision.transforms.Resize())",
    )
    parser.add_argument(
        "--diameter",
        default=17.0,
        type=float,
        help="STUDENT and TEACHER: Value for mean diameter for object (pixel)",
    )
    parser.add_argument(
        "--s_path",
        type=str,
        default=None,
        help="STUDENT: model to load (pretrained or pruned)",
    )
    parser.add_argument(
        "--t_path", type=str, default=None, help="TEACHER: model to load (pretrained)"
    )
    parser.add_argument(
        "--s_dilation",
        action="store_true",
        help="STUDENT: model with dilated convolutional layers",
    )
    parser.add_argument(
        "--s_upsample_compr",
        action="store_true",
        help="STUDENT: compression on upsample part",
    )
    parser.add_argument(
        "--data_aug",
        action="store_true",
        help="STUDENT: data augmentation during training",
    )
    parser.add_argument(
        "--patches",
        action="store_true",
        help="Training with patches from 3D images (jpg files)",
    )
    parser.add_argument(
        "--slices",
        action="store_true",
        help="Training with slices from 3D images (tif file)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display images and masks",
    )
    parser.add_argument(
        "--tool_path",
        action="store_true",
        help="Use DAccuracy by calling tool using the path",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Do only evaluation and not training",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=4,
        help="Number of folds for cross-validation",
    )

    args = parser.parse_args()

    main(args)
