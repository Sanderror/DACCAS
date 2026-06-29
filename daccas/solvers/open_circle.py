"""Open Circle solver.

Runs the YOLOv8 detector (OPEN_CIRCLE_BEST_MODEL.pt) on the CAPTCHA and returns
the centre (x, y) of the most confident predicted bounding box. The training
notebook used imgsz=480 and a prediction confidence threshold of 0.25, so we
mirror that here.
"""
from __future__ import annotations

from .base import BaseSolver, ImageInput, load_pil

IMG_SIZE = 480
CONF_THRESHOLD = 0.25


class OpenCircleSolver(BaseSolver):
    HANDLES = ("Open Circle",)

    def __init__(self, weights_path: str, device: str = "cpu",
                 imgsz: int = IMG_SIZE, conf: float = CONF_THRESHOLD):
        super().__init__(device=device)
        self.weights_path = weights_path
        self.imgsz = imgsz
        self.conf = conf
        self.model = None

    def _load(self) -> None:
        from ultralytics import YOLO
        self.model = YOLO(self.weights_path)

    def solve(self, image: ImageInput) -> dict:
        self.load()
        img = load_pil(image).convert("RGB")
        results = self.model.predict(
            source=img, imgsz=self.imgsz, conf=self.conf,
            device=self.device, verbose=False,
        )
        r = results[0]
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            return {
                "class": "Open Circle",
                "solver": "open_circle",
                "found": False,
                "x": None, "y": None, "confidence": None,
                "message": "No open-circle bounding box detected.",
            }

        # pick the most confident box
        confs = boxes.conf.tolist()
        best_i = max(range(len(confs)), key=lambda i: confs[i])
        # xyxy -> centre
        x1, y1, x2, y2 = boxes.xyxy[best_i].tolist()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        all_boxes = [
            {
                "x": (b[0] + b[2]) / 2.0,
                "y": (b[1] + b[3]) / 2.0,
                "confidence": c,
                "xyxy": list(b),
            }
            for b, c in zip(boxes.xyxy.tolist(), confs)
        ]

        return {
            "class": "Open Circle",
            "solver": "open_circle",
            "found": True,
            "x": cx,
            "y": cy,
            "confidence": confs[best_i],
            "n_detections": len(confs),
            "all_boxes": all_boxes,
        }
