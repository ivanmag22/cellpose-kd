# https://github.com/haitongli/knowledge-distillation-pytorch/blob/master/train.py
# https://github.com/ivanmag22/Semantic_Segmentation_project/blob/main/notebook_files/run_stdc_bisenetv1.ipynb

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
import torchvision.transforms as T

from cellpose.models import Cellpose
from cellpose.models_fastcp import Cellpose as FastCellpose

# from FastCellpose.cellpose.models import Cellpose as FastCellpose
from cellpose.transforms import (
    convert_image,
    resize_image,
    get_pad_yx,
    make_tiles,
    make_tiles_tensor,
    average_tiles,
    average_tiles_tensor,
)

from cellpose.dynamics import compute_masks_torch, compute_masks

from cellpose.train import (
    _loss_fn_seg,
    _get_batch,
    _reshape_norm,
    _loss_fn_seg1,
)

from cellpose.utils import poly_lr_scheduler

from dataset.dataset import OrgDataset, OrgDataset_patches, OrgDataset_slices

# from utils.image_utils import resize_3D


def preprocess_and_infer_tensor(
    img,
    cp,
    diameter=17,
    bsize=224,
    nclasses=3,
    tile_overlap=0.1,
    batch_size=8,
    device=torch.device("cpu"),
):
    """
    Simplified function for resizing, padding, and inference on an image.

    Args:
        img (ndarray or torch.Tensor): Input image (1x1024x1024x1 expected).
        cp (CellposeModel): Model for inference.
        diameter (float): Desired diameter for resizing.
        bsize (int): Size of tiles for model processing.
        nclasses (int): Number of output channels.
        tile_overlap (float): Overlap fraction for tiles.
        batch_size (int): Batch size for model inference.

    Returns:
        ndarray: Processed output.
    """

    img, _ = _get_batch(
        list(range(batch_size)), data=img
    )  # list of bs elements whose shape is (256, 256)

    img = _reshape_norm(
        img, channels=[0, 0]
    )  # list of bs elements whose shape is (2, 256, 256)
    img = np.array(img)  # .transpose(0, 2, 3, 1)

    # Compute rescale factor based on the given diameter
    model_diam = cp.net.module.diam_mean.item()  # 17
    rescale = (
        model_diam / diameter
    )  # rescale = diameter / model_diam (read 02/04 notes)
    rsz = [rescale, rescale]

    # Resize image
    # Lz, Ly0, Lx0, nchan = img.shape
    Lz, nchan, Ly0, Lx0 = img.shape
    Lyr, Lxr = int(Ly0 * rsz[0]), int(Lx0 * rsz[1])

    # Compute padding
    ypad1, ypad2, xpad1, xpad2 = get_pad_yx(Lyr, Lxr)
    pads = np.array([[0, 0], [ypad1, ypad2], [xpad1, xpad2]])

    # Determine tiling dimensions
    Ly, Lx = Lyr + ypad1 + ypad2, Lxr + xpad1 + xpad2
    ny = 1 if Ly <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Ly / bsize))
    nx = 1 if Lx <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Lx / bsize))
    ly, lx = min(bsize, Ly), min(bsize, Lx)

    # Initialize outputs for predictions (yf) and tiles (IMGa)
    yf = torch.zeros(Lz, nclasses, Ly, Lx, dtype=torch.float32).to(device)
    ntiles = ny * nx  # Number of tiles per image
    nimgs = max(1, batch_size // ntiles)  # Number of images per batch
    niter = int(np.ceil(Lz / nimgs))
    for k in range(niter):
        inds = np.arange(
            k * nimgs, min(Lz, (k + 1) * nimgs)
        )  # Indices for batch processing
        IMGa = torch.zeros(ntiles * len(inds), nchan, ly, lx, dtype=torch.float32).to(
            device
        )
        for i, b in enumerate(inds):
            # # Resize and pad the image, so Ly and Lx are divisible by 4
            resize_t = T.Resize(
                (int(img[b].shape[-2] * rsz[0]), int(img[b].shape[-1] * rsz[1])),
                interpolation=T.InterpolationMode.BILINEAR,
            )
            imgb = resize_t(
                torch.from_numpy(img[b])
            )  # resize_image(img[b], rsz=rsz) if rsz is not None else img[b].copy()
            pads = (xpad1, xpad2, ypad1, ypad2, 0, 0)
            imgb = F.pad(
                imgb, pads, mode="constant"
            )  # imgb = np.pad(imgb.transpose(2, 0, 1), pads, mode="constant")

            # Create tiles for large images
            IMG, ysub, xsub, Ly, Lx = make_tiles_tensor(
                imgb, bsize=bsize, augment=False, tile_overlap=tile_overlap
            )
            IMGa[i * ntiles : (i + 1) * ntiles] = torch.reshape(
                IMG, (ny * nx, nchan, ly, lx)
            ).to(device)
        # Run the network on the tiles and combine results
        ya = torch.zeros(IMGa.shape[0], nclasses, ly, lx, dtype=torch.float32).to(
            device
        )

        for j in range(0, IMGa.shape[0], batch_size):
            bslc = slice(j, min(j + batch_size, IMGa.shape[0]))
            ya[bslc] = cp.net(IMGa[bslc])[0]
        for i, b in enumerate(inds):
            y = ya[i * ntiles : (i + 1) * ntiles]
            yfi = average_tiles_tensor(y.cpu(), ysub, xsub, Ly, Lx).to(device)
            yf[b] = yfi[:, : imgb.shape[-2], : imgb.shape[-1]]

    # Remove padding from predictions and resize back to original dimensions
    yf = yf[:, :, ypad1 : Ly - ypad2, xpad1 : Lx - xpad2]
    if rescale != 1.0:
        # yf = resize_image(yf, Ly0, Lx0)  # Resize back to original size if needed
        resize_t = T.Resize((Ly0, Lx0), interpolation=T.InterpolationMode.BILINEAR)
        yf = resize_t(yf)
    # Extract cell probabilities and compute masks
    cellprob = yf[:, 2, :, :]
    dP = yf[:, :2, :, :]

    return cellprob, dP, yf


def preprocess_and_infer(
    img,
    cp,
    diameter=17,
    bsize=224,
    nclasses=3,
    tile_overlap=0.1,
    batch_size=8,
    device=torch.device("cpu"),
):
    """
    Simplified function for resizing, padding, and inference on an image.

    Args:
        img (ndarray or torch.Tensor): Input image (1x1024x1024x1 expected).
        cp (CellposeModel): Model for inference.
        diameter (float): Desired diameter for resizing.
        bsize (int): Size of tiles for model processing.
        nclasses (int): Number of output channels.
        tile_overlap (float): Overlap fraction for tiles.
        batch_size (int): Batch size for model inference.

    Returns:
        ndarray: Processed output.
    """
    if isinstance(img, torch.Tensor):
        # Convert to NumPy array
        img = img.detach().cpu().numpy()

    img, _ = _get_batch(
        list(range(batch_size)), data=img
    )  # list of bs elements whose shape is (256, 256)

    img = _reshape_norm(
        img, channels=[0, 0]
    )  # list of bs elements whose shape is (2, 256, 256)
    img = np.array(img).transpose(0, 2, 3, 1)

    # Compute rescale factor based on the given diameter
    model_diam = 17
    rescale = (
        model_diam / diameter
    )  # rescale = diameter / model_diam (read 02/04 notes)
    rsz = [rescale, rescale]

    # Resize image
    Lz, Ly0, Lx0, nchan = img.shape
    Lyr, Lxr = int(Ly0 * rsz[0]), int(Lx0 * rsz[1])

    # Compute padding
    ypad1, ypad2, xpad1, xpad2 = get_pad_yx(Lyr, Lxr)
    pads = np.array([[0, 0], [ypad1, ypad2], [xpad1, xpad2]])

    # Determine tiling dimensions
    Ly, Lx = Lyr + ypad1 + ypad2, Lxr + xpad1 + xpad2
    ny = 1 if Ly <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Ly / bsize))
    nx = 1 if Lx <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Lx / bsize))
    ly, lx = min(bsize, Ly), min(bsize, Lx)

    # Initialize outputs for predictions (yf) and tiles (IMGa)
    yf = np.zeros((Lz, nclasses, Ly, Lx), "float32")
    ntiles = ny * nx  # Number of tiles per image
    nimgs = max(1, batch_size // ntiles)  # Number of images per batch
    niter = int(np.ceil(Lz / nimgs))
    for k in range(niter):
        inds = np.arange(
            k * nimgs, min(Lz, (k + 1) * nimgs)
        )  # Indices for batch processing
        IMGa = np.zeros((ntiles * len(inds), nchan, ly, lx), "float32")
        for i, b in enumerate(inds):
            # # Resize and pad the image, so Ly and Lx are divisible by 4
            imgb = resize_image(img[b], rsz=rsz) if rsz is not None else img[b].copy()
            imgb = np.pad(imgb.transpose(2, 0, 1), pads, mode="constant")

            # Create tiles for large images
            IMG, ysub, xsub, Ly, Lx = make_tiles(
                imgb, bsize=bsize, augment=False, tile_overlap=tile_overlap
            )
            IMGa[i * ntiles : (i + 1) * ntiles] = np.reshape(
                IMG, (ny * nx, nchan, ly, lx)
            )
        # Run the network on the tiles and combine results
        ya = np.zeros((IMGa.shape[0], nclasses, ly, lx), "float32")
        for j in range(0, IMGa.shape[0], batch_size):
            bslc = slice(j, min(j + batch_size, IMGa.shape[0]))
            ya[bslc] = cp.net(torch.from_numpy(IMGa[bslc]))[0]
        for i, b in enumerate(inds):
            y = ya[i * ntiles : (i + 1) * ntiles]
            yfi = average_tiles(y, ysub, xsub, Ly, Lx)
            yf[b] = yfi[:, : imgb.shape[-2], : imgb.shape[-1]]

    # Remove padding from predictions and resize back to original dimensions
    yf = yf[:, :, ypad1 : Ly - ypad2, xpad1 : Lx - xpad2]
    yf = yf.transpose(0, 2, 3, 1)
    if rescale != 1.0:
        yf = resize_image(yf, Ly0, Lx0)  # Resize back to original size if needed
    # Extract cell probabilities and compute masks
    cellprob = yf[..., 2]
    dP = yf[..., :2].transpose((3, 0, 1, 2))
    # masks = cp._compute_masks((batch_size, Ly0, Lx0), dP, cellprob, niter=200)

    return cellprob, dP, yf.permute(3, 0, 1, 2)


def val_iou(
    s_model,
    t_model,
    dataloader,
    loss_func,
    device=torch.device("cpu"),
    bs=1,
    diam=17,
    display=False,
    mask_opt=False,
):

    print("start val!")

    # loss function
    if loss_func == "default":
        # MSE loss + BCEWithLogitsLoss
        criterion = _loss_fn_seg
    elif loss_func == "new":
        criterion = _loss_fn_seg1
    else:
        # Binary Cross Entropy Loss
        criterion = nn.BCEWithLogitsLoss()

    iou_record = []
    dice_record = []
    loss_record = []

    tq = tqdm(total=len(dataloader) * bs)
    tq.set_description("Validation")
    for i, data in enumerate(dataloader):
        # move to GPU if available
        if torch.cuda.is_available() and device.type == "cuda":
            data = data.cuda()

        if display:
            for i, x in enumerate(data):
                plt.imshow(x, cmap="gray")
                plt.title(f"Image n.{i+1}")
                plt.show()

        with torch.no_grad():
            s_model.cp.net.eval()
            s_cellprob, s_dP, s_output = preprocess_and_infer_tensor(
                data,
                s_model.cp,
                diameter=diam,
                batch_size=bs,
                device=device,
            )

        if mask_opt:
            if s_cellprob.ndim == 2:
                student_mask = compute_masks_torch(
                    s_dP,
                    s_cellprob,
                    niter=50,
                    device=device,
                )
            else:
                student_mask = torch.zeros(
                    s_cellprob.shape[0],
                    s_cellprob.shape[1],
                    s_cellprob.shape[2],
                ).to(
                    device
                )  # (bs, w, h)
                for i, x in enumerate(s_cellprob):
                    student_mask = compute_masks_torch(
                        s_dP,
                        s_cellprob,
                        niter=50,
                        device=device,
                    )

        if display:
            for i, x in enumerate(s_cellprob):
                plt.imshow(x > 0.5, cmap="gray")
                plt.title(f"Student mask n.{i+1}")
                plt.show()

        with torch.no_grad():
            t_model.cp.net.eval()
            t_cellprob, t_dP, t_output = preprocess_and_infer_tensor(
                data,
                t_model.cp,
                diameter=25,
                batch_size=bs,
                device=device,
            )

        if mask_opt:
            if t_cellprob.ndim == 2:
                teacher_mask = compute_masks_torch(
                    t_dP,
                    t_cellprob,
                    niter=50,
                    device=device,
                )
            else:
                teacher_mask = torch.zeros(
                    t_cellprob.shape[0],
                    t_cellprob.shape[1],
                    t_cellprob.shape[2],
                ).to(
                    device
                )  # (bs, w, h)
                for i, x in enumerate(t_cellprob):
                    teacher_mask[i] = compute_masks_torch(
                        t_dP[i],
                        t_cellprob[i],
                        niter=50,
                        device=device,
                    )

        if display:
            for i, x in enumerate(t_cellprob):
                plt.imshow(x > 0.5, cmap="gray")
                plt.title(f"Teacher mask n.{i+1}")
                plt.show()

        t_output = t_output.detach().cpu().numpy()
        for i in range(t_output.shape[0]):
            t_output[i, 0], t_output[i, 1], t_output[i, 2] = (
                t_output[i, 2],
                t_output[i, 0].copy(),
                t_output[i, 1].copy(),
            )

        if mask_opt:
            s_bin_mask = (student_mask > 0).long().to(device)  # student binary mask
            t_bin_mask = (teacher_mask > 0).long().to(device)  # teacher binary mask
        else:
            s_bin_mask = (s_cellprob > 0.5).long().to(device)  # student binary mask
            t_bin_mask = (t_cellprob > 0.5).long().to(device)  # teacher binary mask
        correct = (
            (torch.sum((s_bin_mask == 1) & (t_bin_mask == 1)).detach().cpu())
            .detach()
            .cpu()
            .numpy()
        )  # intersection
        union = (
            (torch.sum((s_bin_mask == 1) | (t_bin_mask == 1)).detach().cpu() + 1e-10)
            .detach()
            .cpu()
            .numpy()
        )
        total = (
            torch.sum((s_bin_mask == 1)).detach().cpu().numpy()
            + torch.sum((t_bin_mask == 1)).detach().cpu().numpy()
            + 1e-10
        )
        iou_record.append(100 * correct / union)  # IoU
        dice_record.append(100 * (2 * correct) / (total))  # Dice

        if loss_func == "default":
            loss = criterion(t_output, s_output, device)
        elif loss_func == "new":
            loss = criterion(t_output, s_output, device)
        else:
            loss = criterion(s_cellprob, t_cellprob)

        loss_record.append(loss.item())

        tq.update(bs)

    tq.close()
    iou = np.mean(iou_record)
    dice = np.mean(dice_record)
    loss = np.mean(loss_record)

    print()
    print(f"val accuracy: {iou:.2f}")

    return iou, dice, loss


def train_kd(
    student_model,
    teacher_model,
    learning_rate,
    optimizer,
    loss_func,
    train_dataloader,
    val_dataloader,
    n_epochs,
    bs,
    epoch_start_i,
    device,
    checkp_step,
    val_step,
    save_path,
    diameter=17,
    display=False,
    freeze=None,
    mask_opt=False,
):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn_kd:
        dataloader:
        n_epochs:
        bs:
        epoch_start_i:
        checkp_step:
        save_path:
        diameter:
    """

    writer = SummaryWriter(
        comment=f"_fastcp_D{diameter}_LR{learning_rate:.2e}_BS{bs}_EP{n_epochs}"
    )

    # loss function
    if loss_func == "default":
        # MSE loss + BCEWithLogitsLoss
        criterion = _loss_fn_seg
    elif loss_func == "new":
        criterion = _loss_fn_seg1
    elif loss_func == "cellprob":
        criterion = _loss_fn_seg2
    else:
        # Binary Cross Entropy Loss
        criterion = nn.BCEWithLogitsLoss()

    max_acc = 0
    prev_acc = 0
    counter = 0

    for epoch in range(epoch_start_i, n_epochs):
        lr = poly_lr_scheduler(optimizer, learning_rate, iter=epoch, max_iter=n_epochs)

        # loss and accuracy lists
        loss_record = []
        train_IoU = []
        train_dice = []

        # student model in train mode
        student_model.cp.net.train()
        if freeze:
            # set Batch Normalization to eval() to apply the running stats instead of calculating and using the batch statistics and updating the running stats (https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/2)
            for name, module in student_model.cp.net.named_modules():
                if freeze == "d":
                    # downsample
                    if isinstance(module, nn.BatchNorm2d) and "downsample" in name:
                        module.eval()
                elif freeze == "u":
                    # upsample
                    if isinstance(module, nn.BatchNorm2d) and "upsample" in name:
                        module.eval()
                elif freeze == "t":
                    # all layers, no transposed convolution layers
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
        # teacher model in train mode
        teacher_model.cp.net.eval()

        # Use tqdm for progress bar
        tq = tqdm(total=len(train_dataloader) * bs)
        tq.set_description(
            "epoch %d/%d, lr %.2e, diam %.3f" % (epoch + 1, n_epochs, lr, diameter)
        )

        for i, train_batch in enumerate(
            train_dataloader
        ):  # dataloader should load an unlabelled dataset
            # train_batch.shape = torch.Size([bs, 256, 256])
            optimizer.zero_grad()  # Zero-ing the gradients

            if display:
                for i, x in enumerate(train_batch):
                    plt.imshow(x, cmap="gray")
                    plt.title(f"Image n.{i+1}")
                    plt.show()

            # move to GPU if available
            if torch.cuda.is_available() and device.type == "cuda":
                train_batch = train_batch.cuda()

            do_3D = True if train_batch.ndim > 3 and train_batch.shape[1] > 1 else False

            student_cellprob, student_dP, student_output = preprocess_and_infer_tensor(
                train_batch, student_model.cp, diameter, batch_size=bs, device=device
            )
            if mask_opt:
                if student_cellprob.ndim == 2:
                    student_mask = compute_masks_torch(
                        student_dP,
                        student_cellprob,
                        niter=50,
                        device=device,
                    )
                    """
                    student_mask = compute_masks(
                        student_dP.detach().cpu().numpy(),
                        student_cellprob.detach().cpu().numpy(),
                        niter=50,
                        device=device,
                    )
                    """
                else:
                    """
                    student_mask = torch.zeros(
                        student_cellprob.shape[0],
                        student_cellprob.shape[1],
                        student_cellprob.shape[2],
                    ).to(
                        device
                    )  # (bs, w, h)
                    """
                    student_mask = np.zeros(student_cellprob.shape)
                    for i, x in enumerate(student_cellprob):
                        student_mask[i] = compute_masks(
                            student_dP[i].detach().cpu().numpy(),
                            student_cellprob[i].detach().cpu().numpy(),
                            niter=50,
                            device=device,
                        )
                        """
                        student_mask[i] = compute_masks(
                            student_dP[i].detach().cpu().numpy(),
                            student_cellprob[i].detach().cpu().numpy(),
                            niter=50,
                            device=device,
                        )   # it returns a numpy.ndarray
                        """
                    student_mask = torch.from_numpy(student_mask).to(device)

            if display:
                if student_cellprob.ndim == 2:
                    tiff.imshow(
                        student_cellprob.detach().cpu().numpy() > 0.5, cmap="gray"
                    )
                    plt.title(f"Student cellprob n.1")
                    plt.show()
                else:
                    for i, x in enumerate(student_cellprob):
                        # tiff.imshow(x, cmap="gray")
                        plt.imshow(x.detach().cpu().numpy() > 0, cmap="gray")
                        plt.title(f"Student cellprob n.{i+1}")
                        plt.show()

            # get one batch output from teacher_outputs list
            with torch.no_grad():
                teacher_cellprob, teacher_dP, teacher_output = (
                    preprocess_and_infer_tensor(
                        train_batch,
                        teacher_model.cp,
                        25,
                        batch_size=bs,
                        device=device,
                    )
                )
            if mask_opt:
                if teacher_cellprob.ndim == 2:
                    teacher_mask = compute_masks_torch(
                        teacher_dP,
                        teacher_cellprob,
                        niter=50,
                        device=device,
                    )
                    """
                    teacher_mask = compute_masks(
                        teacher_dP,
                        teacher_cellprob,
                        niter=50,
                        device=device,
                    )   # it returns a numpy.ndarray
                    """
                else:
                    """
                    teacher_mask = torch.zeros(
                        teacher_cellprob.shape[0],
                        teacher_cellprob.shape[1],
                        teacher_cellprob.shape[2],
                    ).to(
                        device
                    )  # (bs, w, h)
                    """
                    teacher_mask = np.zeros(teacher_cellprob.shape)
                    for i, x in enumerate(teacher_cellprob):
                        teacher_mask[i] = compute_masks(
                            teacher_dP[i].detach().cpu().numpy(),
                            teacher_cellprob[i].detach().cpu().numpy(),
                            niter=50,
                            device=device,
                        )
                        """
                        teacher_mask[i] = compute_masks(
                            teacher_dP[i],
                            teacher_cellprob[i],
                            niter=50,
                            device=device,
                        )   # it returns a numpy.ndarray
                        """
                    teacher_mask = torch.from_numpy(teacher_mask).to(device)

            teacher_output = teacher_output.detach().cpu().numpy()
            for i in range(teacher_output.shape[0]):
                teacher_output[i, 0], teacher_output[i, 1], teacher_output[i, 2] = (
                    teacher_output[i, 2],
                    teacher_output[i, 0].copy(),
                    teacher_output[i, 1].copy(),
                )

            if display:
                if teacher_cellprob.ndim == 2:
                    tiff.imshow(
                        teacher_cellprob.detach().cpu().numpy() > 0, cmap="gray"
                    )
                    plt.title(f"Teacher cellprob n.1")
                    plt.show()
                else:
                    for i, x in enumerate(teacher_cellprob):
                        # tiff.imshow(x, cmap="gray")
                        plt.imshow(x.detach().cpu().numpy() > 0, cmap="gray")
                        plt.title(f"Teacher cellprob n.{i+1}")
                        plt.show()
            """teacher_cellprob = teacher_cellprob.detach().cpu()
            if display:
                if teacher_cellprob.ndim == 2:
                    tiff.imshow(teacher_cellprob, cmap="gray")
                    plt.title(f"Teacher mask n.1")
                    plt.show()
                else:
                    for i, x in enumerate(teacher_cellprob):
                        # tiff.imshow(x, cmap="gray")
                        plt.imshow(x > 0, cmap="gray")
                        plt.title(f"Teacher mask n.{i+1}")
                        plt.show()"""
            # teacher_dP Shape: (bs, 2, 256, 256)
            """# Expand cellprob_teacher to match the first dimension
            cellprob_teacher_exp = np.expand_dims(
                teacher_cellprob.detach().cpu().numpy(), axis=1
            )
            # Shape: (bs, 1, 256, 256)

            # Concatenate along the first axis (channel-like dimension)
            lbl = np.concatenate(
                (cellprob_teacher_exp, teacher_dP.detach().cpu().numpy()), axis=1
            )
            # Shape: (bs, 3, 256, 256)"""

            # training accuracy
            if mask_opt:
                s_bin_mask = (student_mask > 0).long().to(device)  # student binary mask
                t_bin_mask = (teacher_mask > 0).long().to(device)  # teacher binary mask
            else:
                s_bin_mask = (
                    (student_cellprob > 0.5).long().to(device)
                )  # student binary mask
                t_bin_mask = (
                    (teacher_cellprob > 0.5).long().to(device)
                )  # teacher binary mask
            correct = (
                (torch.sum((s_bin_mask == 1) & (t_bin_mask == 1)).detach().cpu())
                .detach()
                .cpu()
                .numpy()
            )  # intersection
            union = (
                (
                    torch.sum((s_bin_mask == 1) | (t_bin_mask == 1)).detach().cpu()
                    + 1e-10
                )
                .detach()
                .cpu()
                .numpy()
            )
            total = (
                torch.sum((s_bin_mask == 1)).detach().cpu().numpy()
                + torch.sum((t_bin_mask == 1)).detach().cpu().numpy()
                + 1e-10
            )
            iou = 100 * correct / union  # IoU
            dice = 100 * (2 * correct / total)  # Dice
            train_IoU.append(iou)
            train_dice.append(dice)

            # loss function
            if loss_func == "default":
                loss = criterion(teacher_output, student_output, device)
            elif loss_func == "new" or loss_func == "cellprob":
                loss = criterion(teacher_output, student_output, device)
            else:
                loss = criterion(
                    student_cellprob, (teacher_cellprob > 0.5).to(device).float()
                )
                # loss = criterion(student_cellprob, torch.sigmoid(teacher_cellprob).to(device).float())
            """
            elif loss_func == "ce":
                loss = criterion(s_bin_mask, t_bin_mask)
            elif loss_func == "log_softmax":
                logprobs = F.log_softmax(
                    output_batch.float(), dim=1
                )  # F.softmax(output_batch.float(), dim=1)
                soft_target = F.log_softmax(
                    output_teacher_batch.float(), dim=1
                )  # the teacher provides soft targets (a distribution over all labels)
                target = soft_target  # since we have partially annotated samples, but with dots
                loss = F.kl_div(
                    logprobs, target, reduction="batchmean"
                )  # in this case I can only trust teacher's outputs
            """

            # compute gradients of all variables wrt loss
            loss.backward()

            loss_record.append(loss.item())

            # performs weights updates using calculated gradients
            optimizer.step()

            tq.update(bs)
            tq.set_postfix(loss="%.2f" % loss)

        tq.close()

        # training accuracy
        train_IoU_mean = np.mean(train_IoU)
        train_dice_mean = np.mean(train_dice)
        if loss_func != "dice":
            writer.add_scalar("IoU/train", train_IoU_mean, epoch)
            print(f"train IoU: {train_IoU_mean:.2f}")
        else:
            writer.add_scalar("Dice/train", train_dice_mean, epoch)
            print(f"train Dice: {train_dice_mean:.2f}")

        # loss mean
        loss_train_mean = np.mean(loss_record)
        # loss_list.append(loss_train_mean)
        writer.add_scalar("Loss/train", loss_train_mean, epoch)
        print(f"loss for train : {loss_train_mean:.2f}")

        # Evaluate summaries only once in a while
        if (epoch + 1) % checkp_step == 0:
            # save the model
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            path = os.path.abspath(
                os.path.join(
                    save_path,
                    f"Saved_model_epoch_{epoch+1}_lr{learning_rate:.2e}_d{diameter:.1f}.pth",
                )
            )
            print(f"saving model at epoch n.{epoch+1}")
            torch.save(student_model.cp.net.module.state_dict(), path)

        # Evaluate summaries only once in a while
        if val_step is not None and (epoch + 1) % val_step == 0:
            IoU, dice, loss = val_iou(
                student_model,
                teacher_model,
                val_dataloader,
                loss_func,
                device,
                bs,
                diameter,
                display,
                mask_opt,
            )
            # val_acc_list.append(acc)
            if loss_func != "dice":
                writer.add_scalar("IoU/validation", IoU, epoch)
            else:
                writer.add_scalar("Dice/validation", dice, epoch)
            writer.add_scalar("Loss/validation", loss, epoch)

            if loss_func == "dice":
                acc = dice
            else:
                acc = IoU

            if acc > max_acc:
                max_acc = acc

                path = os.path.abspath(
                    os.path.join(
                        save_path,
                        f"Best_model_epoch_{epoch+1}_lr{learning_rate:.2e}_d{diameter:.1f}.pth",
                    )
                )
                torch.save(student_model.cp.net.module.state_dict(), path)
            else:
                # early stop condition: if the performance is getting really worse, stop it
                if (abs(max_acc - acc) > 5) and epoch != 0:
                    print(f"Early stop condition at epoch n.{epoch+1}")
                    return max_acc
            if abs(prev_acc - acc) < 0.25:
                counter += 1
                if counter == 3:
                    # early stop condition: if we are in a plateau, stop it
                    print(f"Early stop condition at epoch n.{epoch+1}")
                    return max_acc
            else:
                counter = 0
            prev_acc = acc

    return max_acc


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
        s_freeze,
        s_fastcellpose,
        data_aug,
        loss_func,
        display,
        s_down_path,
        s_up_path,
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
        args.s_freeze,
        args.s_fastcellpose,
        args.data_aug,
        args.loss_func,
        args.display,
        args.s_down_path,
        args.s_up_path,
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
        val_dataset = OrgDataset_patches(
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
            size=args.image_size,
            mode="train",
        )
        val_dataset = OrgDataset_slices(
            samples_root,
            augmentation=data_aug,
            num_samples=args.num_samples,
            samples_per_class=args.samples_per_class,
            size=args.image_size,
            mode="val",
        )
    else:
        train_dataset = OrgDataset(
            samples_root,
            augmentation=data_aug,
            num_samples=args.num_samples,
            samples_per_class=args.samples_per_class,
            mode="train",
        )
        val_dataset = OrgDataset(
            samples_root,
            augmentation=data_aug,
            num_samples=args.num_samples,
            samples_per_class=args.samples_per_class,
            mode="val",
        )

    dataloader_train = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        # num_workers=args.num_workers,    # TODO: review
        # pin_memory=False,    # TODO: review
        drop_last=True,
    )
    dataloader_val = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=True,
        # num_workers=args.num_workers,    # TODO: review
        # pin_memory=False,    # TODO: review
        drop_last=True,
    )

    ## model
    # teacher
    t_model = Cellpose(
        model_type="nuclei",
        gpu=True if device.type == "cuda" else False,
        device=device,
        style=not args.t_no_style,
    )
    if t_path:
        # load only weights belonging to layers that are in common between the model and the pretrained model
        t_model.cp.net.load_state_dict(
            torch.load(t_path, map_location=device), strict=False
        )

    # student
    if s_fastcellpose:
        style_on = False
        residual_on = False
        concatenation = False
        model_name = "demoglo_nbase=32_conv=2"
        inference_model_name = "FastCellpose/demo_infer/demoglo_nbase=32_conv=2.pth"
        model_type = "nuclei"
        s_model = FastCellpose(
            gpu=True if device.type == "cuda" else False,
            device=device,
            pretrained_model=None if args.s_init else inference_model_name,
            model_type=model_type,
            net_avg=False,
            style_on=not args.s_no_style,
            concatenation=args.s_concatenation,
            residual_on=args.s_residual_on,
        )
    else:
        s_model = Cellpose(
            model_type="nuclei",
            dilation=s_dilation,
            upsample_compr=s_upsample_compr,
            gpu=True if device.type == "cuda" else False,
            device=device,
            style=not args.s_no_style,
            nbase=args.s_nbase,
            depthwise=args.s_depthwise,
        )

    if s_path:
        # load only weights belonging to layers that are in common between the model and the pretrained model
        s_model.cp.net.load_state_dict(
            torch.load(s_path, map_location=device), strict=False
        )
    if s_down_path or s_up_path:
        weights = s_model.cp.net.state_dict()
        if s_down_path:
            down_d = torch.load(s_down_path, map_location=device)
            weights.update({k: v for k, v in down_d.items() if "downsample" in k})
        if s_up_path:
            up_d = torch.load(s_up_path, map_location=device)
            weights.update({k: v for k, v in up_d.items() if "upsample" in k})
        s_model.cp.net.load_state_dict(weights, strict=False)
    if s_freeze == "d":
        # freeze downsample part
        for name, module in s_model.cp.net.named_modules():
            if "downsample" in name:
                if isinstance(module, nn.BatchNorm2d):
                    if hasattr(module, "weight"):
                        module.weight.requires_grad_(False)
                    if hasattr(module, "bias"):
                        module.bias.requires_grad_(False)
                    module.eval()
                elif isinstance(module, nn.Conv2d):
                    if hasattr(module, "weight"):
                        module.weight.requires_grad_(False)
                    if hasattr(module, "bias"):
                        module.bias.requires_grad_(False)
                elif isinstance(module, nn.Conv3d):
                    if hasattr(module, "weight"):
                        module.weight.requires_grad_(False)
                    if hasattr(module, "bias"):
                        module.bias.requires_grad_(False)
                elif isinstance(module, nn.Linear):
                    if hasattr(module, "weight"):
                        module.weight.requires_grad_(False)
                    if hasattr(module, "bias"):
                        module.bias.requires_grad_(False)
    elif s_freeze == "u":
        # freeze upsample part
        for name, module in s_model.cp.net.named_modules():
            if "upsample" in name:
                if isinstance(module, nn.BatchNorm2d):
                    if hasattr(module, "weight"):
                        module.weight.requires_grad_(False)
                    if hasattr(module, "bias"):
                        module.bias.requires_grad_(False)
                    module.eval()
                elif isinstance(module, nn.Conv2d):
                    if hasattr(module, "weight"):
                        module.weight.requires_grad_(False)
                    if hasattr(module, "bias"):
                        module.bias.requires_grad_(False)
                elif isinstance(module, nn.Conv3d):
                    if hasattr(module, "weight"):
                        module.weight.requires_grad_(False)
                    if hasattr(module, "bias"):
                        module.bias.requires_grad_(False)
                elif isinstance(module, nn.Linear):
                    if hasattr(module, "weight"):
                        module.weight.requires_grad_(False)
                    if hasattr(module, "bias"):
                        module.bias.requires_grad_(False)
                elif isinstance(module, nn.ConvTranspose2d):
                    if hasattr(module, "weight"):
                        module.weight.requires_grad_(False)
                    if hasattr(module, "bias"):
                        module.bias.requires_grad_(False)
                elif isinstance(module, nn.ConvTranspose3d):
                    if hasattr(module, "weight"):
                        module.weight.requires_grad_(False)
                    if hasattr(module, "bias"):
                        module.bias.requires_grad_(False)
    elif s_freeze == "t":
        # freeze convolutional layers and batch normalization ones, not transposed convolutional layers
        for module in s_model.cp.net.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, "weight"):
                    module.weight.requires_grad_(False)
                if hasattr(module, "bias"):
                    module.bias.requires_grad_(False)
                module.eval()
            elif isinstance(module, nn.Conv2d):
                if hasattr(module, "weight"):
                    module.weight.requires_grad_(False)
                if hasattr(module, "bias"):
                    module.bias.requires_grad_(False)
            elif isinstance(module, nn.Conv3d):
                if hasattr(module, "weight"):
                    module.weight.requires_grad_(False)
                if hasattr(module, "bias"):
                    module.bias.requires_grad_(False)
            elif isinstance(module, nn.Linear):
                if hasattr(module, "weight"):
                    module.weight.requires_grad_(False)
                if hasattr(module, "bias"):
                    module.bias.requires_grad_(False)

    ## optimizer
    # build optimizer
    if optim == "rmsprop":
        optimizer = torch.optim.RMSprop(s_model.cp.net.parameters(), lr)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            s_model.cp.net.parameters(), lr, momentum=0.9, weight_decay=1e-4
        )
    elif optim == "adam":
        optimizer = torch.optim.Adam(s_model.cp.net.parameters(), lr)
    else:
        print("not supported optimizer \n")
        return None

    ## load a saved model from start epoch
    if epoch_start_i != 0:
        path = os.path.abspath(
            os.path.join(
                saved_model_dir,
                f"Saved_model_epoch_{epoch_start_i}_lr{lr:.2e}_d{diam:.1f}.pth",
            )
        )
        print(f"loading data from saved model {path}")
        s_model.cp.net.load_state_dict(torch.load(path))

    if torch.cuda.is_available() and device.type == "cuda":
        s_model.cp.net = torch.nn.DataParallel(s_model.cp.net).cuda()
        t_model.cp.net = torch.nn.DataParallel(
            t_model.cp.net
        ).cuda()  # I put it again, before it was uncommented
    elif device.type == "cpu":
        s_model.cp.net = torch.nn.DataParallel(s_model.cp.net)
        t_model.cp.net = torch.nn.DataParallel(
            t_model.cp.net
        )  # I put it again, before it was uncommented

    # Teacher: set requires_grad to False (not necessary, given torch.no_grad())
    for param in t_model.cp.net.parameters():
        param.requires_grad = False

    if not args.eval_only:
        train_kd(
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
            saved_model_dir,
            diam,
            display,
            s_freeze,
            args.mask_opt,
        )

    val_iou(
        s_model,
        t_model,
        dataloader=dataloader_val,
        loss_func=loss_func,
        device=device,
        bs=bs,
        diam=diam,
        display=display,
        mask_opt=args.mask_opt,
    )


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
        default="default",
        help="Loss function: 'default' from train.py (lambda=5), 'new' for train.py loss function with lambda=1. Default: 'new'",
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
        help="STUDENT: model with dilation convolutional layers",
    )
    parser.add_argument(
        "--s_down_path",
        type=str,
        default=None,
        help="STUDENT: Downsample weights to load",
    )
    parser.add_argument(
        "--s_upsample_compr",
        action="store_true",
        help="STUDENT: model with upsample compression (transposed convolution layers)",
    )
    parser.add_argument(
        "--s_up_path", type=str, default=None, help="STUDENT: Upsample weights to load"
    )
    parser.add_argument(
        "--s_freeze",
        type=str,
        default=None,
        help="STUDENT: freeze layers. Options: 'd' for freezing downsample part layers, 'u' for freezing upsample part layers, 't' for freezing all the layers, but not transposed convolution ones. Default: None",
    )
    parser.add_argument(
        "--s_nbase",
        nargs="+",
        default=[32, 64, 128, 256],
        type=int,
        help="STUDENT: Set the number of base feature maps",
    )
    parser.add_argument(
        "--s_depthwise",
        action="store_true",
        help="STUDENT: It replaces standard convolutional layers with depthwise ones",
    )
    parser.add_argument(
        "--s_fastcellpose",
        action="store_true",
        help="STUDENT: It uses FastCellpose with pretrained model as student",
    )
    parser.add_argument(
        "--s_no_style",
        action="store_true",
        help="STUDENT: It does not include style in upsampling computations",
    )
    parser.add_argument(
        "--s_residual_on",
        action="store_true",
        help="STUDENT: It includes conv+upsampling layers. Default: False, conv+transposedconv",
    )
    parser.add_argument(
        "--s_concatenation",
        action="store_true",
        help="STUDENT: It includes concatenation in network computation",
    )
    parser.add_argument(
        "--s_init",
        action="store_true",
        help="STUDENT: Model with random initialized weights, otherwise glomeruli pretrained model is loaded.",
    )
    parser.add_argument(
        "--t_no_style",
        action="store_true",
        help="TEACHER: It does not include style in upsampling computations",
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
        "--mask_opt",
        action="store_true",
        help="Compute accuracy according to mask",
    )

    args = parser.parse_args()

    main(args)
