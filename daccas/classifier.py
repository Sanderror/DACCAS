"""DACCAS classifier (the dispatcher).

Wraps the YOLOv8 classification model (CLASSIFICATION_MODEL.pt). The model's own
class names are the filesystem-safe folder names used during training
(e.g. "text_gregwar"); we map them back to the canonical human-readable labels
used throughout DACCAS (e.g. "Text (Gregwar)").
"""
from __future__ import annotations

from .solvers.base import ImageInput, load_pil

IMG_SIZE = 224

# safe folder name (what the YOLO model emits) -> canonical DACCAS class label
SAFE_TO_CLASS = {
    "text_gregwar": "Text (Gregwar)",
    "text_mobicms": "Text (Mobicms)",
    "text_king": "Text (King)",
    "text_other": "Text (Other)",
    "moving_window": "Moving Window",
    "open_circle": "Open Circle",
    "img_rotation_normal": "Image Rotation (Default)",
    "img_rotation_special": "Image Rotation (Special)",
    "no_solver": "No Solver",
}


class DACCASClassifier:
    def __init__(self, weights_path: str, device: str = "cpu",
                 imgsz: int = IMG_SIZE):
        self.weights_path = weights_path
        self.device = device
        self.imgsz = imgsz
        self.model = None

    def load(self) -> "DACCASClassifier":
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
        return self

    def classify(self, image: ImageInput) -> dict:
        """Return {'class', 'confidence', 'raw_name'} for the top-1 prediction."""
        self.load()
        img = load_pil(image).convert("RGB")
        r = self.model.predict(
            source=img, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]
        top1 = int(r.probs.top1)
        raw_name = r.names[top1]
        label = SAFE_TO_CLASS.get(raw_name, raw_name)
        return {
            "class": label,
            "confidence": float(r.probs.top1conf),
            "raw_name": raw_name,
        }
