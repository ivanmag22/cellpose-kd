import numpy as np
import time, os, sys
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import tifffile as tiff
import skimage.io

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

# import torch.optim as optim
# from torch.profiler import record_function, ProfilerActivity
from torchinfo import summary
from thop import profile

# import torch_pruning as tp
from calflops import calculate_flops

import argparse

from cellpose.models import Cellpose


def count_nonzero_params(model):
    total_params = 0
    nonzero_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        nonzero_params += (param != 0).sum().item()

    return nonzero_params, total_params


def main(args):
    model_path, dilation, upsample_compr, nbase, no_style, depthwise, x, y, z = (
        args.model_path,
        args.dilation,
        args.upsample_compr,
        args.nbase,
        args.no_style,
        args.depthwise,
        args.x,
        args.y,
        args.z,
    )

    model = Cellpose(
        gpu=False,
        model_type="nuclei",
        dilation=dilation,
        upsample_compr=upsample_compr,
        nbase=nbase,
        style=not no_style,
        depthwise=depthwise,
        fastcp=args.fastcp,
    )
    net = model.cp.net

    input = (z, 2, x, y)
    t = torch.randn(input)

    print(model.__dict__)  # To see all attributes

    summary(
        net, input, device=torch.device("cpu"), verbose=1
    )  # verbose=2 for full details also for weights

    if model_path or dilation or upsample_compr:
        if model_path:
            net.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu")),
                strict=False,
            )

        # parameters
        params, total = count_nonzero_params(net)

        # macs, params = tp.utils.count_ops_and_params(net, t)
        macs = calculate_flops(
            model=net, input_shape=input, output_as_string=False, output_precision=4
        )[1]

        if z > 1:
            # 3D segmentation
            # measures on net_ortho
            net_ortho = model.cp.net_ortho
            # parameters
            params_ortho, total_ortho = count_nonzero_params(net_ortho)

            # macs, params = tp.utils.count_ops_and_params(net, t)
            # TODO: change input, since we need to consider it in YZ and ZX planes
            macs_ortho = calculate_flops(
                model=net_ortho,
                input_shape=input,
                output_as_string=False,
                output_precision=4,
            )[1]

            params += params_ortho
            total += total_ortho
            macs += macs_ortho

        print(f"MACs:\t{macs}")

        print(f"Parameters:")
        print(f"\tNon-zero parameters: {params}")
        print(f"\tTotal parameters: {total}")
        print(f"\tPruned percentage: {100 * (1 - params / total):.2f}%")
    else:
        macs, params = profile(net, inputs=(t,))
        if z > 1:
            # 3D segmentation
            # measures on net_ortho
            net_ortho = model.cp.net_ortho
            # parameters
            params_ortho, total_ortho = count_nonzero_params(net_ortho)

            # macs, params = tp.utils.count_ops_and_params(net, t)
            # TODO: change input, since we need to consider it in YZ and ZX planes
            macs_ortho = calculate_flops(
                model=net_ortho,
                input_shape=input,
                output_as_string=False,
                output_precision=4,
            )[1]

            params += params_ortho
            total += total_ortho
            macs += macs_ortho
        print(f"MACs:\t{macs}\nParameters:\t{params}")

    return params, macs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--do_3D",
        action="store_true",
        help="It enables computation on 3D segmentation network",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="Model with dilated convolution layers",
    )
    parser.add_argument(
        "--upsample_compr",
        action="store_true",
        help="Model with upsample compression",
    )
    parser.add_argument(
        "--no_style",
        action="store_true",
        help="It does not include style in upsampling computations",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Model path")
    parser.add_argument(
        "--nbase",
        nargs="+",
        default=[32, 64, 128, 256],
        type=int,
        help="Set the number of base feature maps",
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
    parser.add_argument("-x", type=int, default=1024, help="Width")
    parser.add_argument("-y", type=int, default=1024, help="Height")
    parser.add_argument("-z", type=int, default=1, help="Number of slices")

    args = parser.parse_args()

    main(args)
