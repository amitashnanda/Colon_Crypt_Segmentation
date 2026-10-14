"""
Colon Crypt Segmentation (PyTorch) - Check the Weight Conversion

This script verifies the Keras -> PyTorch weight conversion tensor by tensor:
  1. it loads notransfer_torch.pth into the PyTorch port's MaskRCNN (pytorch_mrcnn/mrcnn/model.py) with
     strict key checking (num_batches_tracked buffers excepted), so a layer left at its random
     initialisation would show up;
  2. it re-derives the Keras name of every PyTorch parameter with an independent mapping (not the
     substring rules of convert_weights_to_pytorch.py) and compares the values after the layout
     transpose. The conversion is exact if every tensor is identical and every Keras tensor is used.

Usage (torch_env, from pytorch_mrcnn/):
    python src/weight_conversion/check_weight_conversion.py --h5 models/notransfer.h5 --pth models/notransfer_torch.pth

Author: Amitash Nanda
Date: 2026-10-09
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch

# PyTorch port of the Mask R-CNN layers
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "mrcnn"))  # pytorch_mrcnn/mrcnn/ (the port's model.py and config.py)
import model as torch_model  # noqa: E402
from config import Config  # noqa: E402


# --- Default Settings ---
ALPHA = "abcdefghijklmnopqrstuvwxyz"   # Keras block / branch letters


class Cfg(Config):
    """Configuration that builds the port's MaskRCNN with the shapes of the crypt model."""
    NAME = "check"
    BACKBONE = "resnet50"
    NUM_CLASSES = 3
    GPU_COUNT = 0
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)


def keras_weights(path):
    """
    Read every tensor of a Keras .h5 weights file.

    Args:
        path: Keras weights (.h5)

    Returns:
        Dict "layer/weight" -> ndarray
    """
    out = {}
    with h5py.File(path, "r") as f:
        root = f["model_weights"] if "model_weights" in f else f
        for g in root:
            for sub in root[g]:
                for w in root[g][sub]:
                    out[f"{sub}/{w}"] = np.asarray(root[g][sub][w][()])
    return out


def keras_name(torch_key):
    """
    Independent map: PyTorch parameter name -> (Keras layer, Keras weight).

    Args:
        torch_key: name in the PyTorch state_dict

    Returns:
        Tuple of (Keras layer name, Keras weight name)
    """
    parts = torch_key.split(".")
    leaf = parts[-1]
    bn_map = {"weight": "gamma:0", "bias": "beta:0", "running_mean": "moving_mean:0", "running_var": "moving_variance:0"}
    conv_map = {"weight": "kernel:0", "bias": "bias:0"}
    if torch_key.startswith("fpn.C1.0."):
        return "conv1", conv_map[leaf]
    if torch_key.startswith("fpn.C1.1."):
        return "bn_conv1", bn_map[leaf]
    if parts[1] in ("C2", "C3", "C4", "C5"):
        stage = parts[1][1]
        block = ALPHA[int(parts[2])]
        sub = parts[3]
        if sub == "downsample":
            idx = parts[4]
            return (f"res{stage}{block}_branch1", conv_map[leaf]) if idx == "0" else (f"bn{stage}{block}_branch1", bn_map[leaf])
        n = int(sub[-1])
        branch = ALPHA[n - 1]
        if sub.startswith("conv"):
            return f"res{stage}{block}_branch2{branch}", conv_map[leaf]
        return f"bn{stage}{block}_branch2{branch}", bn_map[leaf]
    fpn = {"P5_conv1": "fpn_c5p5", "P4_conv1": "fpn_c4p4", "P3_conv1": "fpn_c3p3", "P2_conv1": "fpn_c2p2",
           "P5_conv2": "fpn_p5", "P4_conv2": "fpn_p4", "P3_conv2": "fpn_p3", "P2_conv2": "fpn_p2"}
    if parts[0] == "fpn" and parts[1] in fpn:
        return fpn[parts[1]], conv_map[leaf]
    if parts[0] == "rpn":
        return {"conv_shared": "rpn_conv_shared", "conv_class": "rpn_class_raw", "conv_bbox": "rpn_bbox_pred"}[parts[1]], conv_map[leaf]
    if parts[0] == "classifier":
        layer = {"conv1": "mrcnn_class_conv1", "conv2": "mrcnn_class_conv2", "bn1": "mrcnn_class_bn1", "bn2": "mrcnn_class_bn2",
                 "linear_class": "mrcnn_class_logits", "linear_bbox": "mrcnn_bbox_fc"}[parts[1]]
        return layer, (bn_map if parts[1].startswith("bn") else conv_map)[leaf]
    if parts[0] == "mask":
        if parts[1] == "conv5":
            return "mrcnn_mask", conv_map[leaf]
        if parts[1] == "deconv":
            return "mrcnn_mask_deconv", conv_map[leaf]
        return f"mrcnn_mask_{parts[1]}", (bn_map if parts[1].startswith("bn") else conv_map)[leaf]
    raise KeyError(torch_key)


def to_torch_layout(arr, torch_key, shape):
    """
    Keras array -> PyTorch layout: 4-D kernels (H, W, in, out) -> (out, in, H, W), 2-D kernels transposed.

    Args:
        arr: Keras array
        torch_key, shape: name and shape of the PyTorch tensor (not needed by the transform)

    Returns:
        Array in PyTorch layout
    """
    if arr.ndim == 4:
        return arr.transpose(3, 2, 0, 1)  # Keras HWIO (or HW-out-in for Conv2DTranspose) -> PyTorch layout
    if arr.ndim == 2:
        return arr.T
    return arr


def main():
    ap = argparse.ArgumentParser(
        description="Verify the Keras -> PyTorch weight conversion tensor by tensor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--h5", required=True, help="Keras weights, e.g. models/notransfer.h5")
    ap.add_argument("--pth", required=True, help="converted state dict, e.g. models/notransfer_torch.pth")
    args = ap.parse_args()

    # --- 1. Keys and shapes against the PyTorch model ---

    torch_model.set_device("cpu")
    net = torch_model.MaskRCNN(config=Cfg(), model_dir="/tmp")
    sd = torch.load(args.pth, map_location="cpu", weights_only=False)
    expected = {k for k in net.state_dict() if not k.endswith("num_batches_tracked")}
    missing = sorted(expected - set(sd))
    unexpected = sorted(set(sd) - expected)
    print(f"PyTorch model tensors (excl. num_batches_tracked): {len(expected)}; in .pth: {len(sd)}")
    print(f"missing from .pth: {len(missing)} {missing[:5]}")
    print(f"unexpected in .pth: {len(unexpected)} {unexpected[:5]}")
    shape_bad = [k for k in expected & set(sd) if tuple(sd[k].shape) != tuple(net.state_dict()[k].shape)]
    print(f"shape mismatches: {len(shape_bad)} {shape_bad[:5]}")


    # --- 2. Values against the Keras tensors (independent name mapping) ---
    kw = keras_weights(args.h5)
    used = set()
    worst = (0.0, None)
    n_cmp = 0
    for k in sorted(expected):
        layer, w = keras_name(k)
        kk = f"{layer}/{w}"
        if kk not in kw:
            print("  !! no keras tensor for", k, "->", kk)
            continue
        used.add(kk)
        a = to_torch_layout(kw[kk], k, sd[k].shape)
        b = sd[k].numpy()
        if a.shape != b.shape:
            print("  !! shape", k, a.shape, b.shape)
            continue
        d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
        n_cmp += 1
        if d > worst[0]:
            worst = (d, k)
    unused = sorted(set(kw) - used)
    print(f"compared {n_cmp} tensors with the .h5; largest |difference| = {worst[0]:.3g} ({worst[1]})")
    print(f"keras tensors not used by the PyTorch model: {len(unused)} {unused[:6]}")
    print(f"total parameters in .pth: {sum(v.numel() for v in sd.values()):,}")
    ok = not missing and not unexpected and not shape_bad and worst[0] == 0.0 and not unused
    print("RESULT:", "conversion is exact" if ok else "conversion has problems (see above)")


if __name__ == "__main__":
    main()


# Example usage (torch_env, from pytorch_mrcnn/):
#
#   python src/weight_conversion/check_weight_conversion.py --h5 models/notransfer.h5 --pth models/notransfer_torch.pth
#   -> ... RESULT: conversion is exact
