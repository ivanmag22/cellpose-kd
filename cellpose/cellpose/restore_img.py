import cv2
import tifffile as tiff
import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import torch

from denoise import CellposeDenoiseModel, DenoiseModel

import argparse


def main(args):
    inp, diam = args.input, args.diameter
    model_type = "denoise_cyto3"

    img = tiff.imread(inp)
    # "denoise", "deblur", "upsample"

    """
    model = CellposeDenoiseModel(
        gpu=False, model_type="nuclei", restore_type="deblur_nuclei"
    )
    _, _, _, img_restore = model.eval(img, channels=[0, 0], diameter=diam)
    """

    model = DenoiseModel(gpu=False, model_type="deblur_nuclei")
    img_restore = model.eval(
        img,
        channels=[0, 0],
        diameter=diam,
    )

    tiff.imshow(img, cmap="gray")
    plt.show()

    tiff.imshow(img_restore, cmap="gray")
    plt.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", "-i", type=str, default=None, help="Path for the input file"
    )
    parser.add_argument(
        "--diameter",
        default=17.0,
        type=float,
        help="Value for mean diameter for object (pixel)",
    )
    parser.add_argument(
        "--only_restore",
        action="store_true",
        help="Models will restore the image, without segmenting it",
    )

    args = parser.parse_args()

    main(args)
