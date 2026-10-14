"""
Colon Crypt Segmentation (PyTorch) - TF-Parity Mask R-CNN Engine

This module is the PyTorch Mask R-CNN behind src/inference/my_inference.py. It gives the detections
of the TF/Keras model (Matterport Mask R-CNN, mrcnn/ at the repository root) for the same tile.

The learned layers (ResNet-50, FPN, RPN, classifier and mask heads) come from the PyTorch port
(pytorch_mrcnn/mrcnn/model.py) with the converted weights models/notransfer_torch.pth, which match the
TF weights notransfer.h5 bit for bit (src/weight_conversion/check_weight_conversion.py). The port
follows an older Matterport release whose box handling differs, so everything that is not a learned
layer is re-implemented here, step by step after the TF library (mrcnn/model.py and mrcnn/utils.py):

  * input:        imgaug PadToAspectRatio(1.5, "center") + Resize(768x1024) reproduced exactly
                  (floor/ceil padding with int(0.5*total) split, cv2 INTER_CUBIC)
  * anchors:      generated in pixels, then normalised with norm_boxes ((h-1, w-1) scale, [0,0,1,1] shift)
  * proposals:    stable top-k 6000, deltas x RPN_BBOX_STD_DEV, clip to [0,1], NMS 0.7 in float32 with
                  tf.image.non_max_suppression semantics (including the order in which its priority queue
                  visits exactly equal scores), zero-padded to 1000 like the Keras graph
  * ROI align:    exact tf.image.crop_and_resize (bilinear, extrapolation 0, TF sampling formula),
                  pyramid level from Equation 1 with image_area of the 1024x1024 molded image
  * detections:   refine_detections_graph: argmax class, BBOX_STD_DEV deltas, clip to the normalised
                  window, score >= 0.7, per-class NMS 0.4 on float boxes (no rounding), top-k 512
  * masks:        mask head on the unrounded normalised detection boxes
  * unmolding:    MaskRCNN.unmold_detections: window normalisation, denorm_boxes with np.around,
                  zero-area filter, utils.unmold_mask (skimage resize order 1, threshold 0.5)

GPU convolutions run with TF32 off and deterministic cuDNN.

Usage (imported; see src/inference/my_inference.py):
    from tfparity_maskrcnn import TFParityMaskRCNN, load_and_preprocess
    model = TFParityMaskRCNN("models/notransfer_torch.pth", device="cuda")
    r = model.detect(load_and_preprocess("tile.png"))   # dict with rois, class_ids, scores, masks

Author: Amitash Nanda
Date: 2026-10-09
"""

import math
import os
import sys

import cv2
import numpy as np
import skimage.io
import skimage.transform
import torch
from skimage import img_as_ubyte

# PyTorch port of the Mask R-CNN layers (pytorch_mrcnn/mrcnn/); its modules import each other by plain name
HERE = os.path.dirname(os.path.abspath(__file__))
PT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # pytorch_mrcnn/
sys.path.insert(0, os.path.join(PT_ROOT, "mrcnn"))
import model as torch_model  # noqa: E402  pytorch_mrcnn/mrcnn/model.py (layers and weights only)
from config import Config as _PortConfig  # noqa: E402  pytorch_mrcnn/mrcnn/config.py


# --- Default Settings (values identical to BowlConfig in the TF script src/inference/my_inference.py) ---
IMAGE_MIN_DIM = 768
IMAGE_MAX_DIM = 1024
MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
RPN_ANCHOR_RATIOS = [0.5, 1, 2]
RPN_ANCHOR_STRIDE = 1
BACKBONE_STRIDES = [4, 8, 16, 32, 64]
RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
PRE_NMS_LIMIT = 6000
POST_NMS_ROIS_INFERENCE = 1000
RPN_NMS_THRESHOLD = 0.7
DETECTION_MIN_CONFIDENCE = 0.7
DETECTION_NMS_THRESHOLD = 0.4
DETECTION_MAX_INSTANCES = 512
POOL_SIZE = 7
MASK_POOL_SIZE = 14
NUM_CLASSES = 3
CLASS_NAMES = ["bg", "crypt", "gland"]

F32 = np.float32  # box arithmetic is done in float32, as in the TF graph


class _ModelConfig(_PortConfig):
    """Configuration used only to build the pytorch_mrcnn modules with the right shapes."""
    NAME = "crypts_tfparity"
    BACKBONE = "resnet50"
    NUM_CLASSES = NUM_CLASSES
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = IMAGE_MIN_DIM
    IMAGE_MAX_DIM = IMAGE_MAX_DIM
    RPN_ANCHOR_SCALES = RPN_ANCHOR_SCALES
    DETECTION_MAX_INSTANCES = DETECTION_MAX_INSTANCES
    DETECTION_NMS_THRESHOLD = DETECTION_NMS_THRESHOLD
    DETECTION_MIN_CONFIDENCE = DETECTION_MIN_CONFIDENCE


# ---------------------------------------------------------------------------------------------
# Preprocessing: exact replica of imgaug 0.4.0 PadToAspectRatio(1.5, "center") + Resize(768x1024)
# ---------------------------------------------------------------------------------------------
def pad_to_aspect_ratio_imgaug(img, aspect=1.5):
    """
    Zero-pad an image to the width/height ratio `aspect`, as imgaug 0.4.0 PadToAspectRatio(position="center").

    Args:
        img: H x W (x C) image
        aspect: target width / height

    Returns:
        The padded image (the input itself if it already has that ratio)
    """
    h, w = img.shape[:2]
    if w / h < aspect:
        diff = aspect * h - w
        total = int(np.ceil(diff / 2)) + int(np.floor(diff / 2))
        left = int(0.5 * total)
        return cv2.copyMakeBorder(img, 0, 0, left, total - left, cv2.BORDER_CONSTANT, value=0)
    if w / h > aspect:
        diff = w / aspect - h
        total = int(np.floor(diff / 2)) + int(np.ceil(diff / 2))
        top = int(0.5 * total)
        return cv2.copyMakeBorder(img, top, total - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
    return img


def load_and_preprocess(path):
    """
    Load one tile and give it the input format of the TF script src/inference/my_inference.py.

    Same steps: skimage.io.imread, grey -> RGB, drop the alpha channel, pad to 3:2 and resize to
    1024 x 768 (bicubic).

    Args:
        path: tile file

    Returns:
        768 x 1024 x 3 uint8 RGB image
    """
    img = skimage.io.imread(path)
    if img.ndim < 3:
        img = img_as_ubyte(img)
        img = np.expand_dims(img, 2)
        img = img[:, :, [0, 0, 0]]
    img = img[:, :, :3]
    img = pad_to_aspect_ratio_imgaug(np.ascontiguousarray(img), 1.5)
    return cv2.resize(img, (1024, 768), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------------------------
# numpy copies of mrcnn/utils.py helpers (that module imports tensorflow, so it is not imported)
# ---------------------------------------------------------------------------------------------
def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    Anchors of one pyramid level in pixels (utils.generate_anchors).

    Args:
        scales: anchor side length(s) in pixels
        ratios: width/height ratios
        shape: [height, width] of the feature map
        feature_stride: stride of the feature map in pixels
        anchor_stride: anchor every `anchor_stride` feature-map cells

    Returns:
        [N, 4] anchors (y1, x1, y2, x2)
    """
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    box_centers = np.stack([box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
    return np.concatenate([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride):
    """
    Anchors of all pyramid levels, level by level (utils.generate_pyramid_anchors).

    Args:
        scales: one anchor scale per level
        ratios: width/height ratios
        feature_shapes: [height, width] per level
        feature_strides: stride per level
        anchor_stride: anchor stride in feature-map cells

    Returns:
        [N, 4] anchors (y1, x1, y2, x2) in pixels
    """
    return np.concatenate([generate_anchors(scales[i], ratios, feature_shapes[i], feature_strides[i], anchor_stride)
                           for i in range(len(scales))], axis=0)


def norm_boxes(boxes, shape):
    """
    Pixel boxes -> normalised float32 boxes (utils.norm_boxes).

    Args:
        boxes: [N, 4] boxes (y1, x1, y2, x2) in pixels
        shape: (height, width) of the image

    Returns:
        [N, 4] float32 boxes in normalised coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """
    Normalised boxes -> int32 pixel boxes, rounded with np.around (utils.denorm_boxes).

    Args:
        boxes: [N, 4] normalised boxes
        shape: (height, width) of the image

    Returns:
        [N, 4] int32 boxes in pixels
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def resize_image_square(image):
    """
    Mold the image to the 1024 x 1024 network input: utils.resize_image(mode="square", min_dim=768,
    max_dim=1024, min_scale=False).

    Args:
        image: H x W x 3 image

    Returns:
        Tuple of (1024 x 1024 x 3 image, window (y1, x1, y2, x2) of the image inside it)
    """
    h, w = image.shape[:2]
    scale = max(1, IMAGE_MIN_DIM / min(h, w))
    if round(max(h, w) * scale) > IMAGE_MAX_DIM:
        scale = IMAGE_MAX_DIM / max(h, w)
    if scale != 1:
        image = skimage.transform.resize(image, (round(h * scale), round(w * scale)), order=1, mode="constant",
                                         cval=0, clip=True, preserve_range=True, anti_aliasing=False)
    h, w = image.shape[:2]
    top = (IMAGE_MAX_DIM - h) // 2
    bottom = IMAGE_MAX_DIM - h - top
    left = (IMAGE_MAX_DIM - w) // 2
    right = IMAGE_MAX_DIM - w - left
    image = np.pad(image, [(top, bottom), (left, right), (0, 0)], mode="constant", constant_values=0)
    return image, (top, left, h + top, w + left)


def skimage_resize_bilinear(mask, out_shape):
    """
    utils.resize(mask, shape): skimage.transform.resize(order=1, mode='constant', cval=0, clip=True,
    preserve_range=False, anti_aliasing=False).

    Kept as a function so that the installed scikit-image can be checked against scikit-image 0.15, the
    version of the original TF environment.

    Args:
        mask: 2-D float mask
        out_shape: (height, width) of the result

    Returns:
        Resized float mask
    """
    return skimage.transform.resize(mask, out_shape, order=1, mode="constant", cval=0, clip=True,
                                    preserve_range=False, anti_aliasing=False)


def unmold_mask(mask, bbox, image_shape):
    """
    Paste a 28 x 28 mask-head output into a full-size bool mask (utils.unmold_mask).

    Args:
        mask: 28 x 28 float mask of one detection
        bbox: (y1, x1, y2, x2) of the detection in pixels
        image_shape: shape of the original image

    Returns:
        H x W bool mask (threshold 0.5)
    """
    y1, x1, y2, x2 = bbox
    m = skimage_resize_bilinear(mask, (y2 - y1, x2 - x1))
    m = np.where(m >= 0.5, 1, 0).astype(bool)
    full = np.zeros(image_shape[:2], dtype=bool)
    full[y1:y2, x1:x2] = m
    return full


# ---------------------------------------------------------------------------------------------
# Box operations in float32 with the operation order of the TF graph
# ---------------------------------------------------------------------------------------------
def apply_box_deltas(boxes, deltas):
    """
    Apply (dy, dx, log(dh), log(dw)) refinements to boxes (apply_box_deltas_graph).

    Args:
        boxes: [N, 4] float32 boxes (y1, x1, y2, x2)
        deltas: [N, 4] float32 refinements

    Returns:
        [N, 4] float32 refined boxes
    """
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + F32(0.5) * height
    center_x = boxes[:, 1] + F32(0.5) * width
    center_y = center_y + deltas[:, 0] * height
    center_x = center_x + deltas[:, 1] * width
    height = height * np.exp(deltas[:, 2])
    width = width * np.exp(deltas[:, 3])
    y1 = center_y - F32(0.5) * height
    x1 = center_x - F32(0.5) * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1).astype(np.float32)


def clip_boxes(boxes, window):
    """
    Clip boxes to a window (clip_boxes_graph: y1 = max(min(y1, wy2), wy1), and so on).

    Args:
        boxes: [N, 4] float32 boxes
        window: (wy1, wx1, wy2, wx2)

    Returns:
        [N, 4] float32 clipped boxes
    """
    wy1, wx1, wy2, wx2 = [F32(v) for v in window]
    y1 = np.maximum(np.minimum(boxes[:, 0], wy2), wy1)
    x1 = np.maximum(np.minimum(boxes[:, 1], wx2), wx1)
    y2 = np.maximum(np.minimum(boxes[:, 2], wy2), wy1)
    x2 = np.maximum(np.minimum(boxes[:, 3], wx2), wx1)
    return np.stack([y1, x1, y2, x2], axis=1).astype(np.float32)


def _heap_pop_order(scores):
    """
    Order in which a libstdc++ std::priority_queue ordered by score only pops the boxes.

    This is the visiting order of TF builds whose NMS comparator has no index tie-break. It only differs
    from score-then-index order when scores are exactly equal.

    Args:
        scores: [N] float32 scores

    Returns:
        List of the N indices in pop order
    """
    s = scores.tolist()
    heap = []
    # push: sift the new element up while its parent has a lower score
    for i in range(len(s)):
        heap.append(i)
        hole = len(heap) - 1
        parent = (hole - 1) // 2
        while hole > 0 and s[heap[parent]] < s[i]:
            heap[hole] = heap[parent]
            hole = parent
            parent = (hole - 1) // 2
        heap[hole] = i
    # pop: libstdc++ __adjust_heap (move the hole to a leaf, then push the last element back up)
    out = []
    while heap:
        out.append(heap[0])
        last = heap.pop()
        if not heap:
            break
        n = len(heap)
        hole, second = 0, 0
        while second < (n - 1) // 2:
            second = 2 * (second + 1)
            if s[heap[second]] < s[heap[second - 1]]:
                second -= 1
            heap[hole] = heap[second]
            hole = second
        if (n & 1) == 0 and second == (n - 2) // 2:
            second = 2 * (second + 1)
            heap[hole] = heap[second - 1]
            hole = second - 1
        parent = (hole - 1) // 2
        while hole > 0 and s[heap[parent]] < s[last]:
            heap[hole] = heap[parent]
            hole = parent
            parent = (hole - 1) // 2
        heap[hole] = last
    return out


def tf_non_max_suppression(boxes, scores, max_output_size, iou_threshold, ties="heap"):
    """
    tf.image.non_max_suppression (CPU kernel).

    Greedy selection in score order with IoU in float32; a box is suppressed when its IoU with a kept box
    is > threshold; boxes with non-positive area never suppress or get suppressed.

    Args:
        boxes: [N, 4] boxes
        scores: [N] scores
        max_output_size: maximum number of boxes kept
        iou_threshold: IoU above which a box is suppressed
        ties: "heap" (TF's priority-queue order for equal scores) or "index"

    Returns:
        int64 indices of the kept boxes, in selection order
    """
    boxes = np.asarray(boxes, np.float32)
    scores = np.asarray(scores, np.float32)
    n = len(scores)
    if n == 0 or max_output_size <= 0:
        return np.zeros((0,), np.int64)
    ymin = np.minimum(boxes[:, 0], boxes[:, 2])
    xmin = np.minimum(boxes[:, 1], boxes[:, 3])
    ymax = np.maximum(boxes[:, 0], boxes[:, 2])
    xmax = np.maximum(boxes[:, 1], boxes[:, 3])
    area = (ymax - ymin) * (xmax - xmin)
    if ties == "heap":
        order = _heap_pop_order(scores)
    else:
        order = np.lexsort((np.arange(n), -scores))
    thr = F32(iou_threshold)
    out_n = min(max_output_size, n)
    sel = np.empty(out_n, np.int64)
    k = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in order:
            if k >= out_n:
                break
            if k and area[i] > 0:
                s = sel[:k]
                ih = np.maximum(np.minimum(ymax[i], ymax[s]) - np.maximum(ymin[i], ymin[s]), F32(0))
                iw = np.maximum(np.minimum(xmax[i], xmax[s]) - np.maximum(xmin[i], xmin[s]), F32(0))
                inter = ih * iw
                iou = inter / ((area[i] + area[s]) - inter)
                iou = np.where(area[s] > 0, iou, F32(0))
                if np.any(iou > thr):
                    continue
            sel[k] = i
            k += 1
    return sel[:k]


def tf_top_k_indices(values, k):
    """
    tf.nn.top_k(sorted=True) indices: descending values, equal values keep the lower index first.

    Args:
        values: [N] values
        k: number of indices to return

    Returns:
        Indices of the k largest values
    """
    order = np.lexsort((np.arange(len(values)), -np.asarray(values, np.float32)))
    return order[:k]


# ---------------------------------------------------------------------------------------------
# tf.image.crop_and_resize (bilinear, extrapolation_value=0), exact sampling formula
# ---------------------------------------------------------------------------------------------
def tf_crop_and_resize(fmap, boxes, crop):
    """
    Bilinear crops of one feature map, as tf.image.crop_and_resize.

    Args:
        fmap: torch float32 feature map [C, H, W]
        boxes: numpy float32 [N, 4] normalised boxes
        crop: output side length

    Returns:
        torch tensor [N, C, crop, crop]
    """
    C, H, W = fmap.shape
    N = len(boxes)
    if N == 0:
        return fmap.new_zeros((0, C, crop, crop))
    # sampling positions (TF formula) and the four neighbours of each
    b = boxes.astype(np.float32)
    y1, x1, y2, x2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    hm1, wm1, cm1 = F32(H - 1), F32(W - 1), F32(crop - 1)
    height_scale = (y2 - y1) * hm1 / cm1
    width_scale = (x2 - x1) * wm1 / cm1
    steps = np.arange(crop, dtype=np.float32)
    in_y = y1[:, None] * hm1 + steps[None, :] * height_scale[:, None]
    in_x = x1[:, None] * wm1 + steps[None, :] * width_scale[:, None]
    valid = ((in_y >= 0) & (in_y <= hm1))[:, :, None] & ((in_x >= 0) & (in_x <= wm1))[:, None, :]
    fy, fx = np.floor(in_y), np.floor(in_x)
    y_lerp, x_lerp = (in_y - fy).astype(np.float32), (in_x - fx).astype(np.float32)
    top = np.clip(fy, 0, H - 1).astype(np.int64)
    bot = np.clip(np.ceil(in_y), 0, H - 1).astype(np.int64)
    left = np.clip(fx, 0, W - 1).astype(np.int64)
    right = np.clip(np.ceil(in_x), 0, W - 1).astype(np.int64)
    dev = fmap.device
    flat = fmap.reshape(C, H * W)

    def gather(rows, cols):
        idx = torch.from_numpy((rows[:, :, None] * W + cols[:, None, :]).reshape(-1)).to(dev)
        return flat.index_select(1, idx).reshape(C, N, crop, crop)

    # bilinear interpolation; samples outside the map get the extrapolation value 0
    tl, tr, bl, br = gather(top, left), gather(top, right), gather(bot, left), gather(bot, right)
    xl = torch.from_numpy(x_lerp).to(dev)[None, :, None, :]
    yl = torch.from_numpy(y_lerp).to(dev)[None, :, :, None]
    t = tl + (tr - tl) * xl
    bm = bl + (br - bl) * xl
    out = t + (bm - t) * yl
    out = torch.where(torch.from_numpy(valid).to(dev)[None], out, torch.zeros((), device=dev, dtype=out.dtype))
    return out.permute(1, 0, 2, 3).contiguous()


def roi_pyramid_levels(boxes, image_shape=(IMAGE_MAX_DIM, IMAGE_MAX_DIM)):
    """
    PyramidROIAlign level of each box (Equation 1 of the FPN paper, normalised coordinates).

    Args:
        boxes: [N, 4] float32 normalised boxes
        image_shape: shape of the molded image

    Returns:
        [N] int64 levels in 2..5
    """
    h = boxes[:, 2] - boxes[:, 0]
    w = boxes[:, 3] - boxes[:, 1]
    image_area = F32(image_shape[0] * image_shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.sqrt(h * w) / (F32(224.0) / np.sqrt(image_area))
        lvl = np.log(x) / np.log(F32(2.0))
        lvl = np.round(lvl)
    lvl = np.where(np.isfinite(lvl), lvl, -1e9)  # log2(0) = -inf -> level 2, as in the TF graph
    return np.clip(4 + lvl, 2, 5).astype(np.int64)


def pyramid_roi_align(boxes, feature_maps, crop):
    """
    ROI align over the feature pyramid (PyramidROIAlign).

    Args:
        boxes: numpy float32 [N, 4] normalised boxes
        feature_maps: [P2, P3, P4, P5], each torch [1, C, H, W]
        crop: output side length

    Returns:
        torch tensor [N, C, crop, crop], in the order of `boxes`
    """
    levels = roi_pyramid_levels(boxes)
    C = feature_maps[0].shape[1]
    out = feature_maps[0].new_zeros((len(boxes), C, crop, crop))
    for i, level in enumerate(range(2, 6)):
        ix = np.where(levels == level)[0]
        if len(ix):
            out[torch.from_numpy(ix).to(out.device)] = tf_crop_and_resize(feature_maps[i][0], boxes[ix], crop)
    return out


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------
class TFParityMaskRCNN:
    """
    Mask R-CNN inference with the PyTorch layers and the TF post-processing.

    Args:
        weights: converted state dict (models/notransfer_torch.pth)
        device: "cuda" or "cpu"
        nms_ties: "heap" (TF's order for exactly equal scores, default) or "index"
    """

    def __init__(self, weights, device="cuda", nms_ties="heap"):
        self.device = torch.device(device)
        self.nms_ties = nms_ties
        # deterministic GPU convolutions in full float32
        if self.device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch_model.set_device(self.device)
        # network with the converted weights (every tensor must fit)
        net = torch_model.MaskRCNN(config=_ModelConfig(), model_dir=os.path.join(PT_ROOT, "logs"))  # nothing is written there
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.endswith("num_batches_tracked")]
        if missing or unexpected:
            raise RuntimeError(f"weights do not fit the model: missing={missing[:5]} unexpected={unexpected[:5]}")
        self.net = net.to(self.device).eval()
        # anchors of the 1024 x 1024 molded image, normalised as in the TF graph
        shapes = [[int(math.ceil(IMAGE_MAX_DIM / s)), int(math.ceil(IMAGE_MAX_DIM / s))] for s in BACKBONE_STRIDES]
        anchors_px = generate_pyramid_anchors(RPN_ANCHOR_SCALES, RPN_ANCHOR_RATIOS, shapes, BACKBONE_STRIDES, RPN_ANCHOR_STRIDE)
        self.anchors = norm_boxes(anchors_px, (IMAGE_MAX_DIM, IMAGE_MAX_DIM))

    # ---- network pieces ----------------------------------------------------------------------
    @torch.no_grad()
    def _backbone_rpn(self, molded):
        """
        ResNet-50 + FPN and the RPN head on the molded image.

        Args:
            molded: 1024 x 1024 x 3 float32 image (mean pixel subtracted)

        Returns:
            Tuple of ([P2, P3, P4, P5] feature maps, RPN probabilities [A, 2], RPN deltas [A, 4])
        """
        x = torch.from_numpy(molded.transpose(2, 0, 1)[None].copy()).to(self.device)
        p2, p3, p4, p5, p6 = self.net.fpn(x)
        outs = [self.net.rpn(p) for p in (p2, p3, p4, p5, p6)]
        rpn_probs = torch.cat([o[1] for o in outs], dim=1)[0]
        rpn_bbox = torch.cat([o[2] for o in outs], dim=1)[0]
        return [p2, p3, p4, p5], rpn_probs.cpu().numpy(), rpn_bbox.cpu().numpy()

    def proposals(self, rpn_probs, rpn_bbox):
        """
        ProposalLayer: top 6000 anchors, refined and clipped, NMS 0.7, zero-padded to 1000 ROIs.

        Args:
            rpn_probs: [A, 2] RPN probabilities
            rpn_bbox: [A, 4] RPN deltas

        Returns:
            [1000, 4] float32 normalised ROIs
        """
        scores = rpn_probs[:, 1].astype(np.float32)
        deltas = (rpn_bbox * RPN_BBOX_STD_DEV.astype(np.float32)).astype(np.float32)
        ix = tf_top_k_indices(scores, min(PRE_NMS_LIMIT, len(scores)))
        boxes = apply_box_deltas(self.anchors[ix], deltas[ix])
        boxes = clip_boxes(boxes, [0, 0, 1, 1])
        keep = tf_non_max_suppression(boxes, scores[ix], POST_NMS_ROIS_INFERENCE, RPN_NMS_THRESHOLD, self.nms_ties)
        rois = np.zeros((POST_NMS_ROIS_INFERENCE, 4), np.float32)
        rois[:len(keep)] = boxes[keep]
        return rois

    @torch.no_grad()
    def classifier(self, feature_maps, rois):
        """
        Classifier head (fpn_classifier_graph) on the ROIs.

        Args:
            feature_maps: [P2, P3, P4, P5]
            rois: [R, 4] float32 normalised ROIs

        Returns:
            Tuple of (class probabilities [R, NUM_CLASSES], box deltas [R, NUM_CLASSES, 4])
        """
        c = self.net.classifier
        x = pyramid_roi_align(rois, feature_maps, POOL_SIZE)
        x = c.relu(c.bn1(c.conv1(x)))
        x = c.relu(c.bn2(c.conv2(x)))
        x = x.view(-1, c.fc_layers_size)
        logits = c.linear_class(x)
        probs = torch.softmax(logits, dim=1)
        deltas = c.linear_bbox(x).view(x.shape[0], -1, 4)
        return probs.cpu().numpy(), deltas.cpu().numpy()

    def refine_detections(self, rois, probs, deltas, window):
        """
        DetectionLayer (refine_detections_graph): best class, refined box, confidence and per-class NMS.

        Args:
            rois: [R, 4] float32 normalised ROIs
            probs: [R, NUM_CLASSES] class probabilities
            deltas: [R, NUM_CLASSES, 4] box deltas
            window: normalised window of the image inside the molded image

        Returns:
            [D, 6] float32 detections (y1, x1, y2, x2, class_id, score), best score first
        """
        n = len(rois)
        class_ids = np.argmax(probs, axis=1)
        class_scores = probs[np.arange(n), class_ids].astype(np.float32)
        deltas_specific = deltas[np.arange(n), class_ids].astype(np.float32)
        refined = apply_box_deltas(rois, (deltas_specific * BBOX_STD_DEV.astype(np.float32)).astype(np.float32))
        refined = clip_boxes(refined, window)
        # foreground detections with score >= DETECTION_MIN_CONFIDENCE
        keep = np.intersect1d(np.where(class_ids > 0)[0], np.where(class_scores >= F32(DETECTION_MIN_CONFIDENCE))[0])
        pre_ids, pre_scores, pre_rois = class_ids[keep], class_scores[keep], refined[keep]
        # NMS per class, classes in order of first appearance
        _, first = np.unique(pre_ids, return_index=True)
        nms_keep = []
        for cid in pre_ids[np.sort(first)]:
            ixs = np.where(pre_ids == cid)[0]
            ck = tf_non_max_suppression(pre_rois[ixs], pre_scores[ixs], DETECTION_MAX_INSTANCES, DETECTION_NMS_THRESHOLD, self.nms_ties)
            nms_keep.append(keep[ixs[ck]])
        nms_keep = np.concatenate(nms_keep) if nms_keep else np.zeros((0,), np.int64)
        keep = np.intersect1d(keep, nms_keep)
        # best DETECTION_MAX_INSTANCES by score
        top = tf_top_k_indices(class_scores[keep], min(len(keep), DETECTION_MAX_INSTANCES))
        keep = keep[top]
        return np.concatenate([refined[keep], class_ids[keep].astype(np.float32)[:, None],
                               class_scores[keep][:, None]], axis=1).astype(np.float32)

    @torch.no_grad()
    def mask_head(self, feature_maps, boxes):
        """
        Mask head (build_fpn_mask_graph) on the detection boxes.

        Args:
            feature_maps: [P2, P3, P4, P5]
            boxes: [D, 4] float32 normalised detection boxes

        Returns:
            [D, 28, 28, NUM_CLASSES] float32 masks, laid out as in Keras
        """
        m = self.net.mask
        x = pyramid_roi_align(boxes, feature_maps, MASK_POOL_SIZE)
        x = m.relu(m.bn1(m.conv1(m.padding(x))))
        x = m.relu(m.bn2(m.conv2(m.padding(x))))
        x = m.relu(m.bn3(m.conv3(m.padding(x))))
        x = m.relu(m.bn4(m.conv4(m.padding(x))))
        x = m.relu(m.deconv(x))
        x = torch.sigmoid(m.conv5(x))
        return x.permute(0, 2, 3, 1).cpu().numpy()  # [N, 28, 28, num_classes] as in Keras

    @staticmethod
    def unmold_detections(detections, mrcnn_mask, original_image_shape, image_shape, window):
        """
        MaskRCNN.unmold_detections: detections of the molded image -> boxes and masks of the original image.

        Args:
            detections: [D, 6] detections (y1, x1, y2, x2, class_id, score), zero rows at the end
            mrcnn_mask: [D, 28, 28, NUM_CLASSES] masks
            original_image_shape: shape of the input image
            image_shape: shape of the molded image
            window: (y1, x1, y2, x2) of the image inside the molded image, in pixels

        Returns:
            Tuple of (boxes [N, 4] int32, class_ids [N], scores [N], masks [H, W, N] bool)
        """
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]
        # boxes relative to the window, then in pixels of the original image
        window = norm_boxes(np.array(window), image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1
        ww = wx2 - wx1
        scale = np.array([wh, ww, wh, ww])
        boxes = np.divide(boxes - shift, scale)
        boxes = denorm_boxes(boxes, original_image_shape[:2])
        # drop detections with zero area
        exclude = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude.shape[0] > 0:
            boxes = np.delete(boxes, exclude, axis=0)
            class_ids = np.delete(class_ids, exclude, axis=0)
            scores = np.delete(scores, exclude, axis=0)
            masks = np.delete(masks, exclude, axis=0)
            N = class_ids.shape[0]
        full = [unmold_mask(masks[i], boxes[i], original_image_shape) for i in range(N)]
        full = np.stack(full, axis=-1) if full else np.empty(original_image_shape[:2] + (0,), dtype=bool)
        return boxes, class_ids, scores, full

    # ---- full detection ------------------------------------------------------------------------
    def detect(self, image, return_stages=False):
        """
        Detect crypts and glands on one tile (MaskRCNN.detect for one image).

        Args:
            image: 768 x 1024 x 3 uint8 RGB tile (output of load_and_preprocess)
            return_stages: also return the output of every stage (for the comparison with TF)

        Returns:
            Dict with rois [N, 4], class_ids [N], scores [N], masks [H, W, N] (and stages if requested)
        """
        molded, window = resize_image_square(image)
        molded = (molded.astype(np.float32) - MEAN_PIXEL).astype(np.float32)
        image_shape = molded.shape
        window_norm = (np.array(window, np.float32) - np.array([0, 0, 1, 1], np.float32)) / \
            (np.array([image_shape[0], image_shape[1]] * 2, np.float32) - F32(1.0))  # norm_boxes_graph
        feature_maps, rpn_probs, rpn_bbox = self._backbone_rpn(molded)
        rois = self.proposals(rpn_probs, rpn_bbox)
        probs, deltas = self.classifier(feature_maps, rois)
        detections = self.refine_detections(rois, probs, deltas, window_norm)
        mrcnn_mask = self.mask_head(feature_maps, detections[:, :4]) if len(detections) else \
            np.zeros((0, 28, 28, NUM_CLASSES), np.float32)
        boxes, class_ids, scores, masks = self.unmold_detections(detections, mrcnn_mask, image.shape, image_shape, window)
        out = {"rois": boxes, "class_ids": class_ids, "scores": scores, "masks": masks}
        if return_stages:
            out["stages"] = {"rois": rois, "mrcnn_class": probs, "mrcnn_bbox": deltas, "detections": detections,
                             "mrcnn_mask": mrcnn_mask, "rpn_probs": rpn_probs, "rpn_bbox": rpn_bbox}
        return out
