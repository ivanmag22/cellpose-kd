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
from torchinfo import summary
from torchviz import make_dot  # for computational graph

from cellpose.models import Cellpose

print("Settings: model_type=nuclei, do_3D=True; sample 256x256x5 sized")
print("Cellpose")
model = Cellpose(gpu=False, model_type="nuclei")
"""
x = np.random.rand(5, 256, 256)
y = torch.from_numpy(model.eval(x, diameter=30.0, do_3D=True)[0])
print(y.shape)
# y = model.cp.net(x)[0]
make_dot(y, params=dict(model.cp.net.named_parameters())).render("nn", format="png")
"""
input()
print("Pruned Cellpose")
model.cp.net.load_state_dict(
    torch.load(
        "C:\\Users\\Ivan Magistro\\Desktop\\Implementation\\models\\pruned_model_17.0 - Copy.pth",
        map_location=torch.device("cpu"),
    )
)
for p in [  # conv_x.0 batch norm, conv_x.1 relu, conv_x.2 conv
    "downsample.down.res_down_0.conv.conv_0.0.weight",
    "downsample.down.res_down_0.conv.conv_0.2.weight",
    "downsample.down.res_down_0.conv.conv_1.0.weight",
    "downsample.down.res_down_0.conv.conv_1.2.weight",
    "downsample.down.res_down_1.conv.conv_0.0.weight",
    "downsample.down.res_down_1.conv.conv_0.2.weight",
    "downsample.down.res_down_3.conv.conv_1.0.weight",
    "downsample.down.res_down_3.conv.conv_1.2.weight",
]:
    print(
        p,
        "\t",
        model.cp.net.state_dict()[p],
        f"\tSize: {model.cp.net.state_dict()[p].shape}",
    )
"""
y = torch.from_numpy(model.eval(x, diameter=30.0, do_3D=True)[0])
# y = model.cp.net(x)[0]
make_dot(y, params=dict(model.cp.net.named_parameters())).render("nn", format="png")
"""
input()
print("Dilated Cellpose")
model = Cellpose(gpu=False, model_type="nuclei", dilation=True)
"""
y = torch.from_numpy(model.eval(x, diameter=30.0, do_3D=True)[0])
# y = model.cp.net(x)[0]
make_dot(y, params=dict(model.cp.net.named_parameters())).render("nn", format="png")
"""
input()
print("Pruned Dilated Cellpose")
model.cp.net.load_state_dict(
    torch.load(
        "C:\\Users\\Ivan Magistro\\Desktop\\Implementation\\models\\pruned_model_30.0_d - Copy.pth",
        map_location=torch.device("cpu"),
    )
)
for p in [  # conv_x.0 batch norm, conv_x.1 relu, conv_x.2 conv
    "downsample.down.res_down_0.conv.conv_0.0.weight",
    "downsample.down.res_down_0.conv.conv_0.2.weight",
    "downsample.down.res_down_0.conv.conv_1.0.weight",
    "downsample.down.res_down_0.conv.conv_1.2.weight",
    "downsample.down.res_down_1.conv.conv_0.0.weight",
    "downsample.down.res_down_1.conv.conv_0.2.weight",
    "downsample.down.res_down_3.conv.conv_1.0.weight",
    "downsample.down.res_down_3.conv.conv_1.2.weight",
]:
    print(
        p,
        "\t",
        model.cp.net.state_dict()[p],
        f"\tSize: {model.cp.net.state_dict()[p].shape}",
    )
"""
y = torch.from_numpy(model.eval(x, diameter=30.0, do_3D=True)[0])
# y = model.cp.net(x)[0]
make_dot(y, params=dict(model.cp.net.named_parameters())).render("nn", format="png")
"""
input()

"""
for param_tensor in model.cp.net.state_dict():
    if "weight" in param_tensor:
        # print(param_tensor)
        print(param_tensor, "\t", model.cp.net.state_dict()[param_tensor])
"""
"""
X = np.random.rand(237, 1024, 1024)
_ = model.eval(X, do_3D=True)  # set the weight correctly with a forward call
"""

# summary(model.cp.net, (2, 237, 1024, 1024), verbose=1)
