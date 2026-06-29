"""Text solver -- TDA transformer for text CAPTCHAs.

One class serves all four text variants; they differ only in weights and
charset:

    Text (Gregwar)  GREGWAR_BEST_MODEL.pth        charset_36 (37 classes incl. null)
    Text (Mobicms)  MOBICMS_BEST_MODEL.pth        charset_36
    Text (King)     KING_FINETUNE_BEST_MODEL.pth  charset_36
    Text (Other)    GENERAL_BEST_MODEL.pth        charset_62 (63 classes incl. null)

Inference mirrors TEXT_SOLVER.ipynb (cell 17):
    image -> RGB -> resize (128 x 32, BILINEAR) -> ToTensor
    -> Normalize(ImageNet) -> CaptchaTDAModel -> argmax over the sequence
    -> CharsetMapper.get_text(...)
"""
from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms as T

from .base import BaseSolver, ImageInput, load_pil
from ..textmodel import CaptchaTDAModel, CharsetMapper

IMG_H = 32
IMG_W = 128
MAX_LENGTH = 26
D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_QE_LAYERS = 3
DIM_FEEDFORWARD = 2048
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TextSolver(BaseSolver):
    """Generic text-CAPTCHA solver, configured per variant via weights+charset."""

    def __init__(self, class_name: str, weights_path: str, charset_path: str,
                 device: str = "cpu", max_length: int = MAX_LENGTH):
        super().__init__(device=device)
        self.class_name = class_name
        self.weights_path = weights_path
        self.charset_path = charset_path
        self.max_length = max_length
        self.model = None
        self.charset = None
        self._to_tensor = T.ToTensor()
        self._normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        # declare what this instance handles (used for registration)
        self.HANDLES = (class_name,)

    def _load(self) -> None:
        self.charset = CharsetMapper(self.charset_path, max_length=self.max_length)
        model = CaptchaTDAModel(
            num_classes=self.charset.num_classes,
            max_length=self.max_length,
            d_model=D_MODEL, nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_qe_layers=NUM_QE_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=0.0,
        )
        ckpt = torch.load(self.weights_path, map_location=self.device)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval().to(self.device)
        self.model = model

    @torch.no_grad()
    def solve(self, image: ImageInput) -> dict:
        self.load()
        img = load_pil(image).convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
        x = self._normalize(self._to_tensor(img)).unsqueeze(0).to(self.device)
        logits = self.model(x)
        pred_ids = logits.argmax(dim=-1)[0]
        text = self.charset.get_text(pred_ids)
        return {
            "class": self.class_name,
            "solver": "text",
            "text": text,
        }
