"""Image Rotation (Special) solver -- Sobel seam score.

From IMAGE_ROTATION_SPECIAL.ipynb (`seam_score_sobel`). The whole-image Sobel
gradient magnitude is computed, then only pixels in a thin annular band around
the seam between the fixed inner disc and the rotating outer ring are summed.
A *lower* score means a smoother seam, i.e. better ring alignment.

Geometry (from the notebook):
    SEAM_R = 40          radius of the inner circle, in pixels
    DELTA  = 5           shell half-thickness around the seam (chosen value)
    band   = { SEAM_R - DELTA <= R <= SEAM_R + DELTA }

This solver needs no trained weights -- it is a pure image-processing score.
Because the objective is to *minimise* the score, the result advertises
`objective = "minimize"` so the (future) multi-variant comparison knows to pick
the lowest-scoring rotation.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import sobel

from .base import BaseSolver, ImageInput, load_pil

SEAM_R = 40
DEFAULT_DELTA = 5


def _build_radius(h: int, w: int) -> np.ndarray:
    cx, cy = w / 2, h / 2
    Y, X = np.ogrid[:h, :w]
    return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)


def seam_score_sobel(img_gray: np.ndarray, R: np.ndarray,
                     seam_r: int = SEAM_R, delta: int = DEFAULT_DELTA) -> float:
    """Sum of Sobel gradient magnitude inside the seam band. Lower = better."""
    sx = sobel(img_gray, axis=1)
    sy = sobel(img_gray, axis=0)
    sobel_magnitude = np.sqrt(sx ** 2 + sy ** 2)
    score_mask = (R >= seam_r - delta) & (R <= seam_r + delta)
    return float(sobel_magnitude[score_mask].sum())


class ImageRotationSpecialSolver(BaseSolver):
    HANDLES = ("Image Rotation (Special)", "Image Rotation (special)")

    def __init__(self, device: str = "cpu",
                 seam_r: int = SEAM_R, delta: int = DEFAULT_DELTA):
        super().__init__(device=device)
        self.seam_r = seam_r
        self.delta = delta

    def _load(self) -> None:
        # nothing to load; pure image processing
        return None

    def solve(self, image: ImageInput) -> dict:
        self.load()
        gray = np.array(load_pil(image).convert("L"), dtype=np.float32)
        R = _build_radius(*gray.shape)
        score = seam_score_sobel(gray, R, seam_r=self.seam_r, delta=self.delta)
        return {
            "class": "Image Rotation (Special)",
            "solver": "image_rotation_special",
            "sobel_score": score,
            "objective": "minimize",   # lowest score across variants is best aligned
            "seam_r": self.seam_r,
            "delta": self.delta,
        }
