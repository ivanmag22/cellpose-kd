import numpy as np
import time, os, sys
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import tifffile as tiff
import skimage.io
import torch

import argparse

from cellpose.models import Cellpose
from cellpose.denoise import DenoiseModel
from cellpose.transforms import convert_image
from cellpose.core import run_net

import random


def main(args):
    (
        device,
        i_path,
        o_path,
        diam,
        model_path,
        dilation,
        upsample_compr,
        avg_pool,
        flow_threshold,
        cellprob_threshold,
        niter,
        down_size,
        no_mask,
        nbase,
        depthwise,
        no_style,
        down_path,
        up_path,
        denoise,
    ) = (
        torch.device(args.device),
        args.input,
        args.output,
        args.diameter,
        args.model_path,
        args.dilation,
        args.upsample_compr,
        args.avg_pool,
        args.flow_threshold,
        args.cellprob_threshold,
        args.niter,
        args.down_size,
        args.no_mask,
        args.nbase,
        args.depthwise,
        args.no_style,
        args.down_path,
        args.up_path,
        args.denoise,
    )

    seed = 0  # 230
    random.seed(seed)
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    assert i_path is not None and (os.path.isfile(i_path) or os.path.isdir(i_path))
    assert (o_path != "." and os.path.isdir(o_path)) or o_path == "."

    filt = ""
    if "gauss" in i_path:
        filt += "_gauss"
    if "opening" in i_path or "open" in i_path:
        filt += "_opening"
    if "median" in i_path:
        filt += "_median"
    if "resized" in i_path:
        filt += "_resized"

    if os.path.isdir(i_path):
        # folder: I take only the first result
        img_path = os.path.join(i_path, os.listdir(i_path)[0])
    else:
        # file
        img_path = i_path
    ext = img_path.split(".")[-1]
    img = np.zeros((1024, 1024))
    if ext == "tif" or ext == "tiff":
        img = tiff.imread(img_path)
        if img.ndim >= 3 and img.shape[-1] == 3:
            # if 3D mask and RGB image
            tp = img.dtype
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(
                tp
            )  # it casts a RGB image into a grayscale one
        img = img.squeeze()
    elif ext == "jpg":
        img = skimage.io.imread(img_path)
    do_3D = True if img.ndim >= 3 and img.shape[-3] > 1 else False

    model = Cellpose(
        gpu=True if device.type == "cuda" else False,
        device=device,
        model_type="nuclei",
        dilation=dilation,
        pooling=not avg_pool,
        upsample_compr=upsample_compr,
        style=not no_style,
        nbase=nbase,
        depthwise=depthwise,
        fastcp=args.fastcp,
    )

    if model_path is not None:
        model.cp.net.load_state_dict(
            torch.load(model_path, map_location=device), strict=False
        )
    else:
        if args.fastcp and args.nbase == [16, 32, 64, 128] and not dilation and not upsample_compr and not depthwise:
            pretrained_path="models/fastcp-pretrained-model/demoglo_nbase=16_conv=2.pth"
            model.cp.net.load_state_dict(
                torch.load(pretrained_path, map_location=device), strict=False
            )

    if down_path or up_path:
        weights = model.cp.net.state_dict()
        if down_path:
            down_d = torch.load(down_path, map_location=device)
            weights.update({k: v for k, v in down_d.items() if "downsample" in k})
        if up_path:
            up_d = torch.load(up_path, map_location=device)
            weights.update({k: v for k, v in up_d.items() if "upsample" in k})
            # weights.update({k: v for k, v in up_d.items()})   # it loads both upsample and output layers
        model.cp.net.load_state_dict(weights, strict=False)

    if denoise:
        cp_den = DenoiseModel(
            gpu=False if device.type == "cpu" else True,
            model_type="denoise_nuclei",
            chan2=False,
            device=device,
        )
        img = cp_den.eval(
            img,
            channels=[0, 0],
            do_3D=do_3D,
            z_axis=0 if do_3D else None,
            diameter=diam,
        )

    torch.use_deterministic_algorithms(
        True
    )  # When enabled, operations will use deterministic algorithms when available, and if only nondeterministic algorithms are available they will throw a RuntimeError when called

    if os.path.isfile(i_path):
        # file
        start = time.time()
        masks, flows, styles, diams = model.eval(
            img,
            diameter=diam,
            do_3D=do_3D,
            z_axis=0 if do_3D else None,
            channels=[0, 0],
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            niter=niter,
            compute_masks=not no_mask,
            down_size=down_size,
        )  # do_3D=False for 2D images, True for 3D images: z_axis=0 for z axis, channels is an optional list to specify channel images of what segment and the nuclear channel, channel_axis to specify the axis which corresponds to the image channels, anisotropy for dividing by 2 the original image in z-axis
        end = time.time()
        print(f"\tElapsed time: {end - start:.2f} s")

        if not no_mask:
            path = ""
            counter = 0
            while True:
                path = f"MASK_{'fastcp_' if args.fastcp else ''}{str(diam)}{filt}{'_d' if dilation else ''}{'_u' if upsample_compr else ''}{'_p' if model_path is not None else ''}{'' if counter==0 else f'_{counter}'}.{ext}"
                if os.path.isfile(os.path.join(o_path, path)):
                    counter += 1
                else:
                    break
            if "tif" in ext:
                # Save mask as 16-bit in case this has to be used for detecting more than 255 objects; tif and png more suitable for 16 bits
                masks = masks.astype(np.uint16)
            elif ext == "jpg":
                # Save flow as 8-bit
                masks = masks.astype(np.uint8)
                for i in range(masks.shape[0]):
                    for j in range(masks.shape[1]):
                        if masks[i, j] != 0:
                            masks[i, j] = 255
            skimage.io.imsave(os.path.join(o_path, path), masks, check_contrast=False)
    else:
        # folder
        paths = os.listdir(i_path)
        n_files = len(paths)
        for i, img_path in enumerate(paths):
            print(f"\t{i+1}/{n_files}: {os.path.split(img_path)[-1]}")

            # reading
            ext = img_path.split(".")[-1]
            img = np.zeros((1024, 1024))
            if ext == "tif" or ext == "tiff":
                img = tiff.imread(os.path.join(i_path, img_path))
                if img.ndim >= 3 and img.shape[-1] == 3:
                    # if 3D mask and RGB image
                    tp = img.dtype
                    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(
                        tp
                    )  # it casts a RGB image into a grayscale one
                img = img.squeeze()
            elif ext == "jpg":
                img = skimage.io.imread(os.path.join(i_path, img_path))

            # inference
            start = time.time()
            masks, flows, styles, diams = model.eval(
                img,
                diameter=diam,
                do_3D=do_3D,
                z_axis=0 if do_3D else None,
                channels=[0, 0],
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                niter=niter,
                compute_masks=not no_mask,
                down_size=down_size,
            )  # do_3D=False for 2D images, True for 3D images: z_axis=0 for z axis, channels is an optional list to specify channel images of what segment and the nuclear channel, channel_axis to specify the axis which corresponds to the image channels, anisotropy for dividing by 2 the original image in z-axis
            end = time.time()
            print(f"\tElapsed time: {end - start:.2f} s")

            if not no_mask:
                path = ""
                counter = 0
                while True:
                    path = f"MASK_{'fastcp_' if args.fastcp else ''}{str(diam)}{filt}{'_d' if dilation else ''}{'_u' if upsample_compr else ''}{'_p' if model_path is not None else ''}{'' if counter==0 else f'_{counter}'}.{ext}"
                    if os.path.isfile(os.path.join(o_path, path)):
                        counter += 1
                    else:
                        break
                if "tif" in ext:
                    # Save mask as 16-bit in case this has to be used for detecting more than 255 objects; tif and png more suitable for 16 bits
                    masks = masks.astype(np.uint16)
                elif ext == "jpg":
                    # Save flow as 8-bit
                    masks = masks.astype(np.uint8)
                    for i in range(masks.shape[0]):
                        for j in range(masks.shape[1]):
                            if masks[i, j] != 0:
                                masks[i, j] = 255
                skimage.io.imsave(
                    os.path.join(o_path, path), masks, check_contrast=False
                )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device where to run the code ('cpu'/'cuda')",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path for the input file or directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=".",
        help="Folder where to save the tiff file. Default: '.', it saves the image where you are executing the code",
    )
    parser.add_argument(
        "--diameter",
        default=17.0,
        type=float,
        help="Value for mean diameter for object (pixel)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="nuclei",
        help="Model type ('nuclei' by default, but there are also 'cyto' etc.) or human-in-the-loop models",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Model to load")
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="Model with dilated convolutional layers (downsample compression)",
    )
    parser.add_argument(
        "--down_path", type=str, default=None, help="Downsample weights to load"
    )
    parser.add_argument(
        "--upsample_compr",
        action="store_true",
        help="Model with upsample compression",
    )
    parser.add_argument(
        "--up_path", type=str, default=None, help="Upsample weights to load"
    )
    parser.add_argument(
        "--nbase",
        nargs="+",
        default=[32, 64, 128, 256],
        type=int,
        help="Set the number of base feature maps",
    )
    parser.add_argument(
        "--no_style",
        action="store_true",
        help="It does not include style in upsampling computations",
    )
    parser.add_argument(
        "--avg_pool",
        action="store_true",
        help="Model with average pooling layers (instead of max pooling layers)",
    )
    parser.add_argument(
        "--flow_threshold",
        default=0.4,
        type=float,
        help="Maximum allowed error of the flows for each mask. Default: 0.4, range: [0.1, 1.1]",
    )
    parser.add_argument(
        "--cellprob_threshold",
        default=0.0,
        type=float,
        help="This threshold determines probability that a detected object is a cell. Default: 0.0, range: [-6, 6]",
    )
    parser.add_argument(
        "--niter",
        default=200,
        type=int,
        help="Number of iterations for gradient computation. Default: 200",
    )
    parser.add_argument(
        "--down_size",
        default=1,
        type=int,
        help="Factor to downsample the vector field (and then to upsample) in order to have a faster mask computation. Default: 1",
    )
    parser.add_argument(
        "--no_mask",
        action="store_true",
        help="It does not compute masks, so only model.cp.net computations",
    )
    parser.add_argument(
        "--depthwise",
        action="store_true",
        help="It replaces standard convolutional layers with depthwise ones",
    )
    parser.add_argument(
        "--fastcp",
        action="store_true",
        help="FastCP will be used",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Model that tries to remove noise from images",
    )

    args = parser.parse_args()

    main(args)
