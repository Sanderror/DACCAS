"""Image Rotation (Default) solver -- ConvNeXt-L features + logistic regression.

Reproduces the inference path from IMAGE_ROTATION_DEFAULT.ipynb /
eval_realworld.py / orientation_lr.py:

  load image -> center square -> circular mask (crop_mode "circular")
  -> resize 224 + ImageNet-normalise -> frozen ConvNeXt-L global-avg-pooled
  1536-d feature -> StandardScaler+OneVsRest LogisticRegression (C=10, from the
  joblib) -> probability of the upright class (index 0).

A *higher* P(upright) means the variant is more likely already upright, so the
result advertises `objective = "maximize"`.

Note: ConvNeXt-L ImageNet weights are needed for the feature extractor. They are
downloaded from download.pytorch.org by default (needs internet). If that is
blocked, pass `convnext_weights_path` pointing to a local
convnext_large-...pth.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import BaseSolver, ImageInput, load_pil
from .. import orientation_lr as olr


def _pil_to_rgb_uint8_tensor(img: Image.Image) -> torch.Tensor:
    """PIL image -> (3,H,W) uint8 tensor, RGBA composited onto black.

    Mirrors orientation_lr.load_rgb but for an already-open PIL image."""
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    arr = np.array(img, copy=True)  # (H,W,3) uint8
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class ImageRotationDefaultSolver(BaseSolver):
    HANDLES = ("Image Rotation (Default)", "Image Rotation (normal)")

    def __init__(self, joblib_path: str, meta_path: str, device: str = "cpu",
                 convnext_weights_path: str | None = None):
        super().__init__(device=device)
        self.joblib_path = joblib_path
        self.meta_path = meta_path
        self.convnext_weights_path = convnext_weights_path
        self.clf = None
        self.extractor = None
        self.meta = None
        self.mode = "circular"
        self.proba_col = 0

    def _load(self) -> None:
        import joblib

        self.meta = json.loads(Path(self.meta_path).read_text())
        self.mode = self.meta.get("crop_mode", "circular")
        upright_idx = self.meta.get("upright_class_index", 0)
        upright_label = self.meta.get("classes", [0, 1, 2, 3])[upright_idx]

        self.clf = joblib.load(self.joblib_path)
        self.proba_col = list(self.clf.classes_).index(upright_label)

        self.extractor = olr.build_extractor(
            torch.device(self.device), self.convnext_weights_path
        )

    @torch.no_grad()
    def _feature(self, image: ImageInput) -> np.ndarray:
        img = load_pil(image)
        sq = olr.center_square(_pil_to_rgb_uint8_tensor(img))
        sq = olr.prep_infer_square(sq, self.mode)          # circular mask
        x = olr.to_model_tensor(sq).unsqueeze(0).to(self.device)
        feat = olr.extract_features(self.extractor, x)
        return feat.cpu().numpy().astype(np.float32)       # (1, 1536)

    def solve(self, image: ImageInput) -> dict:
        self.load()
        X = self._feature(image)
        p_upright = float(self.clf.predict_proba(X)[0, self.proba_col])
        return {
            "class": "Image Rotation (Default)",
            "solver": "image_rotation_default",
            "p_upright": p_upright,
            "objective": "maximize",   # highest P(upright) across variants is best
            "crop_mode": self.mode,
        }
