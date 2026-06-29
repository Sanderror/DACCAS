"""DACCAS -- Dark-web Adaptive CAPTCHA Classification And Solving."""
from .daccas import DACCAS
from .classifier import DACCASClassifier
from . import capture

__all__ = ["DACCAS", "DACCASClassifier", "capture"]
__version__ = "0.1.0"
