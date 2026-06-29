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

from pathlib import Path
from typing import Union

from PIL import Image

ImageInput = Union[str, Path, Image.Image]


def load_pil(image: ImageInput) -> Image.Image:
    """Accept a path or a PIL image and return a PIL image (mode left as-is)."""
    if isinstance(image, Image.Image):
        return image
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
