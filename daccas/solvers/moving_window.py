"""Moving Window solver -- ResNet-34 single-character classifier.

From MOVING_WINDOW.ipynb. The best model is a torchvision ResNet-34 with its
final fc replaced by Linear(512, 32) (32 character classes). Preprocessing
exactly mirrors the notebook's `transform_function`:

    increase_contrast(factor=1.25) -> Resize((224,224)) -> ToTensor
    -> Normalize(ImageNet mean/std)

The model outputs logits over 32 classes; argmax gives the class index, which is
mapped back to a character via an ordered charset.

IMPORTANT -- class ordering
---------------------------
In the notebook the mapping was built as:
    label_to_idx = {label: idx for idx, label in
                    enumerate(sorted(df_clean["label"].unique()))}
so class index i corresponds to the i-th entry of the *sorted unique training
labels*. That exact 32-character ordering is data-dependent and is NOT contained
in the .pth weights. You must supply it (see `charsets/moving_window_charset.txt`
or pass `classes=`). Until the correct list is supplied the solver still runs and
returns the class index, but the character it prints is only as correct as the
charset you give it.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import ImageEnhance
from torchvision import transforms, models

from .base import BaseSolver, ImageInput, load_pil

NUM_CLASSES = 32
IMG_SIZE = 224
CONTRAST_FACTOR = 1.25
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _increase_contrast(img):
    return ImageEnhance.Contrast(img).enhance(CONTRAST_FACTOR)


def build_transform(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Lambda(_increase_contrast),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_charset(path: str) -> list[str]:
    """One entry per line, in class-index order. Blank lines are ignored,
    but a line that is a single space is treated as the space character."""
    out = []
    with open(path, "r") as f:
        for line in f:
            ch = line.rstrip("\n")
            if ch == "":
                continue
            out.append(ch)
    return out


class MovingWindowSolver(BaseSolver):
    HANDLES = ("Moving Window",)

    def __init__(self, weights_path: str, device: str = "cpu",
                 classes: list[str] | None = None,
                 charset_path: str | None = None,
                 img_size: int = IMG_SIZE):
        super().__init__(device=device)
        self.weights_path = weights_path
        self.img_size = img_size
        self.model = None
        self.transform = build_transform(img_size)

        if classes is not None:
            self.classes = list(classes)
        elif charset_path is not None:
            self.classes = load_charset(charset_path)
        else:
            self.classes = None  # will fall back to index strings

        if self.classes is not None and len(self.classes) != NUM_CLASSES:
            raise ValueError(
                f"Moving Window expects {NUM_CLASSES} classes, "
                f"got {len(self.classes)}."
            )

    def _load(self) -> None:
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        state = torch.load(self.weights_path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval().to(self.device)
        self.model = model

    @torch.no_grad()
    def solve(self, image: ImageInput) -> dict:
        self.load()
        img = load_pil(image).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(probs.argmax().item())
        conf = float(probs[idx].item())

        if self.classes is not None:
            char = self.classes[idx]
        else:
            char = str(idx)

        return {
            "class": "Moving Window",
            "solver": "moving_window",
            "character": char,
            "class_index": idx,
            "confidence": conf,
            "charset_provided": self.classes is not None,
        }
