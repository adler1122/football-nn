from ultralytics import YOLO
import numpy as np


_CONF_MIN = 0.5


class FieldDetector:
    """
    Detects pitch keypoints in a camera frame using a YOLO pose model
    trained on football field lines.
    """

    def __init__(self, model_path: str, device: str) -> None:
        self.model = YOLO(model_path)
        self.model.to(device)

    def detect_keypoints(self, frame: np.ndarray) -> "np.ndarray | None":
        """
        Returns (N, 2) float32 array of valid keypoint pixel coords, or None.
        Only keypoints with confidence >= _CONF_MIN are returned.
        """
        result = self.model.predict(frame, conf=0.25, verbose=False)[0]

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return None

        pts  = result.keypoints.xy[0].cpu().numpy()   # (32, 2)
        conf = result.keypoints.conf
        conf = conf[0].cpu().numpy() if conf is not None else np.zeros(len(pts))

        
        valid = (conf >= _CONF_MIN) & ~((pts[:, 0] == 0) & (pts[:, 1] == 0))

        if valid.sum() < 4:
            return None


        filtered = pts.copy()
        filtered[~valid] = 0
        return filtered

    def detect_keypoints_with_confidence(
        self, frame: np.ndarray
    ) -> "tuple[np.ndarray, np.ndarray] | None":
        """
        Returns (keypoints, confidences) preserving all 32 index positions.
        Invalid keypoints have conf=0.0 and coords=(0,0).
        Returns None if fewer than 4 keypoints pass the threshold.
        """
        result = self.model.predict(frame, conf=0.25, verbose=False)[0]

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return None

        pts  = result.keypoints.xy[0].cpu().numpy()   # (32, 2)
        conf = result.keypoints.conf
        conf = conf[0].cpu().numpy() if conf is not None else np.zeros(len(pts))

        
        valid = (conf >= _CONF_MIN) & ~((pts[:, 0] == 0) & (pts[:, 1] == 0))

        if valid.sum() < 4:
            return None

        pts_out  = pts.copy()
        conf_out = conf.copy()
        pts_out[~valid]  = 0
        conf_out[~valid] = 0.0

        return pts_out, conf_out