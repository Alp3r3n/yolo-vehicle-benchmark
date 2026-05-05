import time
import torch
from ultralytics import YOLO

from config import VEHICLE_CLASSES


class VehicleDetector:
    """Loads a YOLO model and runs vehicle-class-filtered inference."""

    def __init__(self, model_path: str, device: str):
        # Fall back to CPU if CUDA is requested but not available
        if device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
            device = "cpu"

        self.device = device
        print(f"Selected device: {self.device}")

        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, frame, conf: float):
        """
        Run inference on a single BGR frame.

        Returns:
            detections (list[dict]): Filtered vehicle detections with label, confidence, bbox.
            inference_ms (float): Time taken for this frame in milliseconds.
        """
        start = time.perf_counter()
        results = self.model(frame, conf=conf, device=self.device, verbose=False)
        inference_ms = (time.perf_counter() - start) * 1000.0

        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(
                    {
                        "label": VEHICLE_CLASSES[cls_id],
                        "confidence": float(box.conf[0]),
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        return detections, inference_ms
