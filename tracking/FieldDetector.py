from ultralytics import YOLO
import numpy as np


class FieldDetector:

    def __init__(self, model_path, device):

        self.model = YOLO(model_path)
        self.model.to(device)

    def detect_keypoints(self, frame):

        result = self.model.predict(
            frame,
            conf=0.25,
            verbose=False
        )[0]

        if (
            result.keypoints is None
            or len(result.keypoints.xy) == 0
        ):
            return None

        pts = result.keypoints.xy[0]

        return pts.cpu().numpy()