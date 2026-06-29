"""Vendored text-CAPTCHA model code (TDA transformer).

Files copied verbatim from the training repo, with internal imports made
relative so they work as a package:
    model.py        -> CaptchaTDAModel (the architecture)
    resnet_tda.py   -> ResNet45 + TDA backbone
    tda.py          -> Triplet Deep Attention module
    dataset.py      -> CharsetMapper (label <-> text), preprocessing reference
"""
from .model import CaptchaTDAModel
from .dataset import CharsetMapper

__all__ = ["CaptchaTDAModel", "CharsetMapper"]
