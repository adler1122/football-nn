from ultralytics import YOLO
import numpy as np


class FieldDetector:
    """
    Detects pitch keypoints in a single camera frame using a YOLO
    pose / keypoint model trained on football field lines.

    Returns
    -------
    detect_keypoints() → (N, 2) float32 numpy array of (x, y) pixel
    coordinates, or *None* when no keypoints are found.
    """

    def __init__(self, model_path: str, device: str) -> None:
        self.model = YOLO(model_path)
        self.model.to(device)

    def detect_keypoints(self, frame: np.ndarray) -> np.ndarray | None:

        result = self.model.predict(
            frame,
            conf=0.25,
            verbose=False,
        )[0]

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return None

        pts = result.keypoints.xy[0]

        if len(pts) == 0:
            return None

        keypoints = pts.cpu().numpy()  # (N, 2) float32


        valid_mask = ~((keypoints[:, 0] == 0) & (keypoints[:, 1] == 0))
        keypoints = keypoints[valid_mask]

        if len(keypoints) < 4:
            return None

        return keypoints

    def detect_keypoints_with_confidence(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:

        result = self.model.predict(
            frame,
            conf=0.25,
            verbose=False,
        )[0]

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return None

        pts = result.keypoints.xy[0].cpu().numpy()      # (N, 2)
        conf = result.keypoints.conf[0].cpu().numpy()   # (N,)

        valid_mask = (conf > 0.0) & ~((pts[:, 0] == 0) & (pts[:, 1] == 0))
        pts = pts[valid_mask]
        conf = conf[valid_mask]

        if len(pts) < 4:
            return None

        return pts, conf