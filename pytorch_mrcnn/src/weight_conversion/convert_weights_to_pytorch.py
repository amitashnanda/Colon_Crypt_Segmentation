"""
Colon Crypt Segmentation (PyTorch) - Keras-to-PyTorch Weight Conversion

This script converts a Matterport Mask R-CNN Keras .h5 (ResNet-50) into a PyTorch state_dict for the
PyTorch port (pytorch_mrcnn/mrcnn/model.py, from multimodallearning/pytorch-mask-rcnn). It is adapted
from the port's convert_from_keras.py, with:
  - ResNet-50 support (6 blocks in C4 instead of 23)
  - h5py 3.x compatibility (weight[()] instead of weight.value)
  - the Keras -> PyTorch layout change of every convolution and dense kernel

Layer names are mapped by ordered substring rules (weight_map.txt lists the Keras tensors), then conv
kernels (kH, kW, Cin, Cout) become (Cout, Cin, kH, kW) and dense kernels (Cin, Cout) become (Cout, Cin).
Check the result with check_weight_conversion.py.

Usage (torch_env, from pytorch_mrcnn/):
    python src/weight_conversion/convert_weights_to_pytorch.py \\
        --keras_model models/notransfer.h5 \\
        --pytorch_model models/notransfer_torch.pth \\
        --backbone resnet50

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import collections

import h5py
import numpy as np
import torch


# --- Default Settings ---
ALPHABET = list("abcdefghijklmnopqrstuvwxyz")   # Keras block / branch letters

# exact renames applied after the substring rules
REPLACE_EXACT = collections.OrderedDict([
    ("conv1.bias", "fpn.C1.0.bias"),
    ("conv1.weight", "fpn.C1.0.weight"),
])


def build_replace_dict(backbone):
    """
    Ordered substring rules that turn Keras tensor names into names of the PyTorch port.

    Args:
        backbone: "resnet50" or "resnet101"

    Returns:
        OrderedDict Keras substring -> PyTorch substring, applied in order
    """
    assert backbone in ("resnet50", "resnet101")
    c4_blocks = {"resnet50": 6, "resnet101": 23}[backbone]

    d = collections.OrderedDict([
        ("beta:0", "bias"),
        ("gamma:0", "weight"),
        ("moving_mean:0", "running_mean"),
        ("moving_variance:0", "running_var"),
        ("bias:0", "bias"),
        ("kernel:0", "weight"),
        ("mrcnn_mask_", "mask."),
        ("mrcnn_mask", "mask.conv5"),
        ("mrcnn_class_", "classifier."),
        ("logits", "linear_class"),
        ("mrcnn_bbox_fc", "classifier.linear_bbox"),
        ("rpn_", "rpn."),
        ("class_raw", "conv_class"),
        ("bbox_pred", "conv_bbox"),
        ("bn_conv1", "fpn.C1.1"),
        ("bn2a_branch1", "fpn.C2.0.downsample.1"),
        ("res2a_branch1", "fpn.C2.0.downsample.0"),
        ("bn3a_branch1", "fpn.C3.0.downsample.1"),
        ("res3a_branch1", "fpn.C3.0.downsample.0"),
        ("bn4a_branch1", "fpn.C4.0.downsample.1"),
        ("res4a_branch1", "fpn.C4.0.downsample.0"),
        ("bn5a_branch1", "fpn.C5.0.downsample.1"),
        ("res5a_branch1", "fpn.C5.0.downsample.0"),
        ("fpn_c2p2", "fpn.P2_conv1"),
        ("fpn_c3p3", "fpn.P3_conv1"),
        ("fpn_c4p4", "fpn.P4_conv1"),
        ("fpn_c5p5", "fpn.P5_conv1"),
        ("fpn_p2", "fpn.P2_conv2.1"),
        ("fpn_p3", "fpn.P3_conv2.1"),
        ("fpn_p4", "fpn.P4_conv2.1"),
        ("fpn_p5", "fpn.P5_conv2.1"),
    ])

    # C2: 3 blocks (a,b,c), C3: 4 blocks (a-d), C4: 6 or 23 blocks, C5: 3 blocks (a-c)
    for stage, n_blocks in [(2, 3), (3, 4), (4, c4_blocks), (5, 3)]:
        for block in range(n_blocks):
            for branch in range(3):
                bn_key = f"bn{stage}{ALPHABET[block]}_branch2{ALPHABET[branch]}"
                res_key = f"res{stage}{ALPHABET[block]}_branch2{ALPHABET[branch]}"
                d[bn_key] = f"fpn.C{stage}.{block}.bn{branch+1}"
                d[res_key] = f"fpn.C{stage}.{block}.conv{branch+1}"

    return d


def load_keras_state_dict(h5_path):
    """
    Read every tensor of a Keras .h5 weights file.

    Args:
        h5_path: Keras weights (.h5)

    Returns:
        OrderedDict "layer_name.weight_name" -> ndarray
    """
    state = collections.OrderedDict()
    with h5py.File(h5_path, mode="r") as f:
        # Matterport typically stores at root; some files wrap in "model_weights"
        root = f["model_weights"] if "model_weights" in f else f
        for group_name, group in root.items():
            if len(group.keys()) == 0:
                continue
            for layer_name, layer in group.items():
                for weight_name, weight in layer.items():
                    state[f"{layer_name}.{weight_name}"] = np.asarray(weight[()])
    return state


def convert(h5_path, backbone):
    """
    Convert a Keras .h5 into a PyTorch state_dict of the port.

    Args:
        h5_path: Keras weights (.h5)
        backbone: "resnet50" or "resnet101"

    Returns:
        Dict PyTorch tensor name -> torch.Tensor
    """
    state = load_keras_state_dict(h5_path)
    replace_dict = build_replace_dict(backbone)

    # Apply substring replacements in order. Each rule may touch many keys.
    for orig, repl in replace_dict.items():
        for key in list(state.keys()):
            if orig in key:
                new_key = key.replace(orig, repl)
                state[new_key] = state.pop(key)

    for orig, repl in REPLACE_EXACT.items():
        if orig in state:
            state[repl] = state.pop(orig)

    # Reshape: Keras Conv (kH,kW,Cin,Cout) -> PyTorch (Cout,Cin,kH,kW)
    # Keras Dense (Cin,Cout) -> PyTorch Linear (Cout,Cin)
    for name in list(state.keys()):
        arr = state[name]
        if arr.ndim == 4:
            state[name] = arr.transpose(3, 2, 0, 1).copy(order="C")
        elif arr.ndim == 2:
            state[name] = arr.transpose(1, 0).copy(order="C")

    return {k: torch.from_numpy(v) for k, v in state.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Matterport Mask R-CNN Keras .h5 into a PyTorch state_dict of the port.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--keras_model", required=True, help="Keras weights, e.g. models/notransfer.h5")
    parser.add_argument("--pytorch_model", required=True, help="state dict to write, e.g. models/notransfer_torch.pth")

    # Model
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet101"], help="backbone of the model")
    args = parser.parse_args()

    sd = convert(args.keras_model, args.backbone)

    # Sanity print
    print(f"Converted {len(sd)} tensors. Sample keys:")
    for k in list(sd.keys())[:10]:
        print(f"  {k:55s} {tuple(sd[k].shape)}")
    print("  ...")
    total = sum(v.numel() for v in sd.values())
    print(f"Total params: {total:,}")

    torch.save(sd, args.pytorch_model)
    print(f"Wrote {args.pytorch_model}")


if __name__ == "__main__":
    main()


# Example usage (torch_env, from pytorch_mrcnn/):
#
#   python src/weight_conversion/convert_weights_to_pytorch.py --keras_model models/notransfer.h5 \
#       --pytorch_model models/notransfer_torch.pth --backbone resnet50
#   python src/weight_conversion/check_weight_conversion.py --h5 models/notransfer.h5 --pth models/notransfer_torch.pth
