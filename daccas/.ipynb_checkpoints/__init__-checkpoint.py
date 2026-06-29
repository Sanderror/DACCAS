"""DACCAS -- Dark-web Adaptive CAPTCHA Classification And Solving."""
from .daccas import DACCAS
from .classifier import DACCASClassifier

__all__ = ["DACCAS", "DACCASClassifier"]
__version__ = "0.1.0"
