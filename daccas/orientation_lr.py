"""
orientation_lr.py
=================
Shared utilities for the ConvNeXt-L + one-vs-rest logistic regression image
orientation approach (Amjoud & Amrouch, IEEE Access 2022), adapted to
90-degree rotation CAPTCHAs.

Pipeline (exactly the paper's design):
    frozen ConvNeXt-L (ImageNet)  ->  global-average-pooled 1536-d feature
    ->  one-vs-rest logistic regression over 4 classes {0, 90, 180, 270}

Only 90-degree multiples are used. Rotations are generated with torch.rot90,
so they are lossless (no interpolation, no resampling artifacts).

Three crop modes control how the circular nature of the real-world CAPTCHAs is
handled:

    "circular"             train: full square with a circular mask applied
                           infer: the real (already circular) image as-is
    "inscribed_infer_only" train: full uncropped square
                           infer: inscribed square (side = S/sqrt2) of the circle
    "inscribed_both"       train: inscribed square of the (circularly masked) square
                           infer: inscribed square of the circle

Note on "inscribed_both": the inscribed square sits entirely inside the
inscribed circle (its corners touch the circle), so it never contains masked
pixels. The circular mask is therefore a pixel no-op there -- the only real
difference vs "inscribed_infer_only" is the TRAINING field of view (center crop
vs full square). Kept as specified on purpose.
"""

from pathlib import Path
import math

import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as TF
from PIL import Image

SQRT2 = math.sqrt(2.0)
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# class index -> rotation applied (counter-clockwise). index 0 == upright.
CLASS_ANGLES = (0, 90, 180, 270)
FEATURE_DIM = 1536  # ConvNeXt-L last stage channels

IMG_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
CROP_MODES = ("circular", "inscribed_infer_only", "inscribed_both", "baseline")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def center_square(img: torch.Tensor) -> torch.Tensor:
    """Center-crop a (C,H,W) tensor to the largest centered square.

    Done before rotation so that a 90-degree rotation maps the crop region onto
    itself (rotation-consistent). Avoids aspect-ratio distortion from resizing a
    non-square image to a square."""
    _, h, w = img.shape
    s = min(h, w)
    return TF.center_crop(img, [s, s])


def inscribed_square(img: torch.Tensor) -> torch.Tensor:
    """Center-crop a square (C,S,S) tensor to its inscribed square (side S/sqrt2).

    This is the largest axis-aligned square fitting inside the inscribed circle,
    so it contains none of the black/white circular corners."""
    s = img.shape[-1]
    crop = int(s / SQRT2)
    return TF.center_crop(img, [crop, crop])


def circular_mask(img: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """Set every pixel outside the inscribed circle of a square tensor to `fill`
    (default black, matching Dread's transparent->black corners)."""
    s = img.shape[-1]
    device = img.device
    yy, xx = torch.meshgrid(
        torch.arange(s, device=device),
        torch.arange(s, device=device),
        indexing="ij",
    )
    c = (s - 1) / 2.0
    r = s / 2.0
    inside = ((yy - c) ** 2 + (xx - c) ** 2) <= r ** 2
    out = img.clone()
    out[:, ~inside] = fill
    return out


def rot90_k(img: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate (C,H,W) by k*90 deg counter-clockwise. Lossless."""
    return torch.rot90(img, k % 4, dims=(-2, -1))


def prep_train_square(sq: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply the mode-specific geometry to a (rotated) training source square."""
    if mode == "circular":
        return circular_mask(sq)
    if mode == "inscribed_infer_only" or mode == "baseline":
        return sq
    if mode == "inscribed_both":
        return inscribed_square(circular_mask(sq))  # mask is a pixel no-op here
    raise ValueError(f"unknown mode: {mode}")


def prep_infer_square(img_sq: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply the mode-specific geometry to a real-world (already circular) square.

    For "circular": re-apply the black mask so the corners match training. This
    is idempotent for Dread (already black corners) and normalizes Pitch's
    white/black corners to black. Assumes the CAPTCHA's circle is the inscribed
    circle of the square frame (true for Dread/Pitch)."""
    if mode == "circular" or mode == "baseline":
        return circular_mask(img_sq)       # enforce black corners to match training
    return inscribed_square(img_sq)        # both inscribed modes infer on the inner square


def to_model_tensor(sq: torch.Tensor) -> torch.Tensor:
    """square uint8 (C,S,S) -> normalized float (C,224,224)."""
    x = sq.to(torch.float32) / 255.0
    x = TF.resize(x, [INPUT_SIZE, INPUT_SIZE], antialias=True)
    x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return x


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_rgb(path) -> torch.Tensor:
    """Load an image as a (3,H,W) uint8 tensor. RGBA is composited onto black."""
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    arr = np.array(img, copy=True)  # (H,W,3) uint8, writable
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def iter_image_paths(root, suffixes=IMG_SUFFIXES):
    root = Path(root)
    paths = [p for p in root.rglob("*") if p.suffix.lower() in suffixes]
    paths.sort()
    return paths


# --------------------------------------------------------------------------- #
# Feature extractor (frozen ConvNeXt-L)
# --------------------------------------------------------------------------- #
def build_extractor(device, weights_path: str | None = None):
    """Frozen ConvNeXt-L. Pretrained weights download from download.pytorch.org
    (blocked in some sandboxes; works on the HPC). Pass weights_path to load a
    local .pth instead."""
    if weights_path:
        model = torchvision.models.convnext_large(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        model = torchvision.models.convnext_large(
            weights=torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_features(model, batch: torch.Tensor) -> torch.Tensor:
    """batch (N,3,224,224) -> (N,1536).

    features -> avgpool -> LayerNorm2d -> flatten. This is exactly the tensor
    that feeds ConvNeXt's final fully-connected layer, which we drop (GAP->1536,
    i.e. "we eliminated the fully connected layers")."""
    x = model.features(batch)
    x = model.avgpool(x)
    x = model.classifier[0](x)   # LayerNorm2d
    x = model.classifier[1](x)   # Flatten -> (N,1536)
    return x


@torch.no_grad()
def features_for_split(model, image_paths, mode, device, batch_size=32,
                       progress_every=200, eval_geometry=False):
    """For each source image: center-square, generate 4 lossless rotations, apply
    a geometry for `mode`, extract GAP features.

    eval_geometry=False -> training geometry (prep_train_square).
    eval_geometry=True  -> inference geometry (prep_infer_square), i.e. exactly
        what real-world images undergo. Use this for the synthetic val/test
        splits so their accuracy is a faithful proxy for real-world performance.
        Only 'inscribed_infer_only' actually differs (train=full square,
        eval=inscribed square); for 'circular' and 'inscribed_both' the two
        geometries coincide.

    Returns X float32 [N*4, 1536], y int64 [N*4] with labels in {0,1,2,3}."""
    prep = prep_infer_square if eval_geometry else prep_train_square
    feats, labels = [], []
    buf, buf_lab = [], []

    def flush():
        if not buf:
            return
        batch = torch.stack(buf).to(device)
        f = extract_features(model, batch).cpu().numpy().astype(np.float32)
        feats.append(f)
        labels.extend(buf_lab)
        buf.clear(); buf_lab.clear()

    n = len(image_paths)
    for i, p in enumerate(image_paths):
        try:
            sq = center_square(load_rgb(p))
        except Exception as e:
            print(f"  skip {p}: {e}")
            continue
        for k in range(4):
            r = prep(rot90_k(sq, k), mode)
            buf.append(to_model_tensor(r))
            buf_lab.append(k)
            if len(buf) >= batch_size:
                flush()
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{n} images")
    flush()

    X = np.concatenate(feats, 0) if feats else np.zeros((0, FEATURE_DIM), np.float32)
    y = np.asarray(labels, np.int64)
    return X, y


@torch.no_grad()
def feature_for_realworld(model, path, mode, device):
    """Single real-world (already circular) image -> (1536,) feature under the
    mode's inference geometry. Used for the group max-P(upright) protocol."""
    sq = center_square(load_rgb(path))
    sq = prep_infer_square(sq, mode)
    x = to_model_tensor(sq).unsqueeze(0).to(device)
    return extract_features(model, x).cpu().numpy().astype(np.float32)[0]