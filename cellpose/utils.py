import torch.nn as nn
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import pandas as pd
import random
import numbers
import torchvision


def poly_lr_scheduler(
    optimizer, init_lr, iter, lr_decay_iter=1, max_iter=300, power=0.9
):
    """Polynomial decay of learning rate
    :param init_lr is base learning rate
    :param iter is a current iteration
    :param lr_decay_iter how frequently decay occurs, default is 1
    :param max_iter is number of maximum iterations
    :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr * (1 - iter / max_iter) ** power
    optimizer.param_groups[0]["lr"] = lr
    return lr


def dice_loss(pred, target, smooth=1e-10, device=torch.device("cpu")):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W).
        target: Tensor of ground truth (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    pred = (pred > 0.5).long().to(device)
    target = torch.from_numpy(target).to(device)
    target = (target > 0.5).long().to(device)

    # Calculate intersection and union
    intersection = (
        (torch.sum((pred == 1) & (target == 1)).detach().cpu()).detach().cpu().numpy()
    )
    total = (
        torch.sum((pred == 1)).detach().cpu().numpy()
        + torch.sum((target == 1)).detach().cpu().numpy()
        + smooth
    )

    # Compute Dice Coefficient
    dice = (2.0 * intersection) / total

    # Return Dice Loss
    return 1 - dice.mean()
