"""Image Rotation (Special) solver -- seam stitching + Sobel seam score.

Two stages:

1. Preprocessing (`stitch_seam`) -- closes the black gap ring between the inner
   disc and the outer ring so the two contents meet at a single seam:
     * the inner disc (radius `inner_r`, default 40 on the 100x100 captcha) is
       isolated with a circular mask;
     * the remaining outer ring is shrunk by `scale` (default 0.93), pulling its
       inner edge inward to meet the disc;
     * the *circular* part of the inner disc (its black corners excluded) is
       stitched back onto the centre of the shrunk ring.

2. Scoring (`seam_score_sobel`) -- whole-image Sobel gradient magnitude summed
   over a thin annular band around the seam (radius `seam_r` +/- `delta`). A
   *lower* score means a smoother seam, i.e. better ring alignment, so the result
   advertises `objective = "minimize"`.

After stitching, the seam sits at exactly `inner_r` (= 40) from the centre, which
is why `seam_r` stays 40. No trained weights are needed.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import sobel

from .base import BaseSolver, ImageInput, load_pil

SEAM_R = 40
DEFAULT_DELTA = 1
INNER_R = 40           # inner-disc radius on the raw 100x100 captcha
OUTER_SCALE = 0.93     # shrink factor that closes the black gap


def _circle_mask(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    Y, X = np.ogrid[:h, :w]
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2


def _build_radius(h: int, w: int) -> np.ndarray:
    cx, cy = w / 2.0, h / 2.0
    Y, X = np.ogrid[:h, :w]
    return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)


def stitch_seam(gray: np.ndarray, inner_r: int = INNER_R,
                scale: float = OUTER_SCALE,
                resample: int = Image.BILINEAR) -> np.ndarray:
    """Close the black gap: shrink the outer ring and re-stitch the inner disc.

    `gray` is a 2-D uint8 array. Returns a 2-D uint8 array (the outer-ring size,
    e.g. 93x93 for a 100x100 input at scale 0.93)."""
    arr = np.asarray(gray)
    H, W = arr.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    inner_mask = _circle_mask(H, W, cx, cy, inner_r)

    # outer ring = image with the inner disc blanked out
    outer = arr.copy()
    outer[inner_mask] = 0

    # shrink the outer ring so its inner edge moves in to meet the disc
    new_w, new_h = int(round(W * scale)), int(round(H * scale))
    outer_small = np.asarray(
        Image.fromarray(outer).resize((new_w, new_h), resample)
    ).copy()

    # stitch the circular inner disc (corners excluded) onto the ring centre
    ocx, ocy = new_w / 2.0, new_h / 2.0
    dst_mask = _circle_mask(new_h, new_w, ocx, ocy, inner_r)
    ys, xs = np.where(dst_mask)
    src_x = np.round(xs - ocx + cx).astype(int)
    src_y = np.round(ys - ocy + cy).astype(int)
    ok = (src_x >= 0) & (src_x < W) & (src_y >= 0) & (src_y < H)
    outer_small[ys[ok], xs[ok]] = arr[src_y[ok], src_x[ok]]
    return outer_small


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
                 seam_r: int = SEAM_R, delta: int = DEFAULT_DELTA,
                 preprocess: bool = True,
                 inner_r: int = INNER_R, scale: float = OUTER_SCALE):
        super().__init__(device=device)
        self.seam_r = seam_r
        self.delta = delta
        self.preprocess = preprocess
        self.inner_r = inner_r
        self.scale = scale

    def _load(self) -> None:
        # nothing to load; pure image processing
        return None

    def solve(self, image: ImageInput) -> dict:
        self.load()
        gray = np.array(load_pil(image).convert("L"), dtype=np.uint8)
        if self.preprocess:
            gray = stitch_seam(gray, inner_r=self.inner_r, scale=self.scale)
        gray = gray.astype(np.float32)
        R = _build_radius(*gray.shape)
        score = seam_score_sobel(gray, R, seam_r=self.seam_r, delta=self.delta)
        return {
            "class": "Image Rotation (Special)",
            "solver": "image_rotation_special",
            "sobel_score": score,
            "objective": "minimize",   # lowest score across variants is best aligned
            "preprocessed": self.preprocess,
            "seam_r": self.seam_r,
            "delta": self.delta,
        }
