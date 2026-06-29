"""Base class for all DACCAS solvers.

The DACCAS design treats the solver library as a set of interchangeable,
independently loadable modules behind a common interface. Every solver:

  * declares the canonical class name(s) it handles (`HANDLES`),
  * lazily loads its model/weights on first use (`load`),
  * exposes a single `solve(image)` entry point returning a structured dict.

This is what makes the library additive: a new CAPTCHA type is supported by
dropping in a new BaseSolver subclass and registering it, with no change to the
dispatcher or the existing solvers, and no retraining of anything else.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

# Anything DACCAS can turn into an image: a file path, a PIL image, raw image
# bytes (e.g. the PNG returned by a browser screenshot), or a numpy array.
ImageInput = Union[str, Path, Image.Image, bytes, bytearray, "np.ndarray"]


def load_pil(image: ImageInput) -> Image.Image:
    """Normalise any supported input to a PIL image (mode left as-is).

    Accepts:
      * PIL.Image                -> returned as-is
      * str / pathlib.Path       -> opened from disk
      * bytes / bytearray        -> decoded (e.g. a browser screenshot PNG)
      * numpy.ndarray            -> wrapped (H x W, or H x W x 3/4, uint8)
    Making bytes a first-class input is what keeps capture tool-agnostic: any
    automation tool that can return screenshot bytes can feed DACCAS directly.
    """
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(image)))
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    return Image.open(image)


class BaseSolver:
    """Abstract solver. Subclasses implement `_load` and `solve`."""

    #: canonical class label(s) (as produced by the classifier) this solver handles
    HANDLES: tuple[str, ...] = ()

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._loaded = False

    # -- lazy loading ------------------------------------------------------- #
    def load(self) -> "BaseSolver":
        """Load weights once. Safe to call repeatedly."""
        if not self._loaded:
            self._load()
            self._loaded = True
        return self

    def _load(self) -> None:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    # -- inference ---------------------------------------------------------- #
    def solve(self, image: ImageInput) -> dict:  # pragma: no cover
        raise NotImplementedError
