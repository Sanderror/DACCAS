"""DACCAS -- Dark-web Adaptive CAPTCHA Classification And Solving.

Top-level object exposing the two-step pipeline:

    daccas = DACCAS(models_dir="models", charsets_dir="charsets")
    cls = daccas.Classify(image)            # -> {'class', 'confidence', ...}
    out = daccas.Solve(image, cls["class"]) # -> structured solver result

The classifier acts as the dispatcher: it predicts one of nine classes, and
Solve routes the image to the matching solver in the library. Solvers lazy-load
their weights the first time they are used, so constructing DACCAS is cheap and
you only pay for the solvers you actually call.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .classifier import DACCASClassifier
from .solvers.base import ImageInput
from .solvers import (
    OpenCircleSolver,
    ImageRotationSpecialSolver,
    ImageRotationDefaultSolver,
    MovingWindowSolver,
    TextSolver,
)

# default weight / charset filenames (as delivered)
DEFAULT_FILES = {
    "classifier": "CLASSIFICATION_MODEL.pt",
    "open_circle": "OPEN_CIRCLE_BEST_MODEL.pt",
    "rotation_default_joblib": "IMAGE_ROTATION_DEFAULT_MODEL.joblib",
    "rotation_default_meta": "IMAGE_ROTATION_DEFAULT_META.json",
    "moving_window": "MOVING_WINDOW_MODEL.pth",
    "gregwar": "GREGWAR_BEST_MODEL.pth",
    "mobicms": "MOBICMS_BEST_MODEL.pth",
    "king": "KING_FINETUNE_BEST_MODEL.pth",
    "general": "GENERAL_BEST_MODEL.pth",
}
DEFAULT_CHARSETS = {
    "charset_36": "charset_36.txt",
    "charset_62": "charset_62.txt",
    "moving_window": "moving_window_charset.txt",
}

NO_SOLVER_MESSAGE = (
    "No solver exists for this CAPTCHA class. The dispatcher routed the image to "
    "'No Solver', meaning DACCAS recognised it as a type outside the current "
    "solver library and deliberately declined to attempt a solution."
)


class DACCAS:
    def __init__(
        self,
        models_dir: str = "models",
        charsets_dir: str = "charsets",
        device: str = "cpu",
        convnext_weights_path: Optional[str] = None,
        moving_window_charset: Optional[str] = None,
    ):
        self.models_dir = Path(models_dir)
        self.charsets_dir = Path(charsets_dir)
        self.device = device

        def m(key):
            return str(self.models_dir / DEFAULT_FILES[key])

        def c(key):
            return str(self.charsets_dir / DEFAULT_CHARSETS[key])

        # --- dispatcher / classifier --------------------------------------- #
        self.classifier = DACCASClassifier(m("classifier"), device=device)

        # --- moving-window charset (the one data-dependent ordering) ------- #
        mw_charset = moving_window_charset or c("moving_window")
        mw_charset = mw_charset if os.path.exists(mw_charset) else None

        # --- solver library ------------------------------------------------ #
        # Each text variant is its own TextSolver instance (weights + charset).
        self.text_solvers = {
            "Text (Gregwar)": TextSolver("Text (Gregwar)", m("gregwar"), c("charset_36"), device),
            "Text (Mobicms)": TextSolver("Text (Mobicms)", m("mobicms"), c("charset_36"), device),
            "Text (King)":    TextSolver("Text (King)",    m("king"),    c("charset_36"), device),
            "Text (Other)":   TextSolver("Text (Other)",   m("general"), c("charset_62"), device),
        }

        self.registry: dict[str, object] = {
            "Open Circle": OpenCircleSolver(m("open_circle"), device=device),
            "Image Rotation (Special)": ImageRotationSpecialSolver(device=device),
            "Image Rotation (Default)": ImageRotationDefaultSolver(
                m("rotation_default_joblib"), m("rotation_default_meta"),
                device=device, convnext_weights_path=convnext_weights_path,
            ),
            "Moving Window": MovingWindowSolver(
                m("moving_window"), device=device, charset_path=mw_charset,
            ),
            **self.text_solvers,
        }

    # ----------------------------------------------------------------------- #
    # Step 1: classification (dispatch)
    # ----------------------------------------------------------------------- #
    def Classify(self, image: ImageInput) -> dict:
        """Predict the CAPTCHA class with the YOLOv8 classification model."""
        return self.classifier.classify(image)

    # ----------------------------------------------------------------------- #
    # Step 2: solving (route to the matching solver)
    # ----------------------------------------------------------------------- #
    def Solve(self, image: ImageInput, captcha_class: str) -> dict:
        """Route `image` to the solver for `captcha_class` and return its result."""
        if captcha_class == "No Solver":
            return {
                "class": "No Solver",
                "solver": None,
                "solved": False,
                "message": NO_SOLVER_MESSAGE,
            }

        solver = self.registry.get(captcha_class)
        if solver is None:
            return {
                "class": captcha_class,
                "solver": None,
                "solved": False,
                "message": f"Unknown class '{captcha_class}'; no solver registered.",
            }

        return solver.solve(image)

    # ----------------------------------------------------------------------- #
    # Convenience: classify then solve in one call
    # ----------------------------------------------------------------------- #
    def run(self, image: ImageInput) -> dict:
        cls = self.Classify(image)
        result = self.Solve(image, cls["class"])
        return {"classification": cls, "result": result}
