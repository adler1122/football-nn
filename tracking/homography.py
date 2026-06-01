import cv2
import numpy as np

PITCH_KEYPOINT_COORDS = np.array([
    # idx  description
    [   0,   0],   #  0  top-left corner
    [ 165,   0],   #  1  top, left penalty-area (outer)
    [ 280,   0],   #  2  top, left goal-area (outer)
    [ 280,   0],   #  3  top, left goal-area (inner)
    [ 165,   0],   #  4  top, left penalty-area (inner)
    [ 525,   0],   #  5  top, centre

    [   0, 680],   #  6  bottom-left corner
    [ 165, 680],   #  7  bottom, left penalty-area (outer)
    [ 280, 680],   #  8  bottom, left goal-area (outer)

    [   0, 220],   #  9  left touchline — top of left penalty box
    [ 165, 220],   # 10  top-left corner of left penalty box
    [ 165, 460],   # 11  bottom-left corner of left penalty box
    [   0, 460],   # 12  left touchline — bottom of left penalty box

    [   0, 290],   # 13  left touchline — top of left goal area
    [ 280, 290],   # 14  top-right of left goal area
    [ 280, 390],   # 15  bottom-right of left goal area
    [   0, 390],   # 16  left touchline — bottom of left goal area

    [1050, 220],   # 17  right touchline — top of right penalty box
    [ 885, 220],   # 18  top-right corner of right penalty box
    [ 885, 460],   # 19  bottom-right corner of right penalty box
    [1050, 460],   # 20  right touchline — bottom of right penalty box

    [1050, 290],   # 21  right touchline — top of right goal area
    [ 770, 290],   # 22  top-left of right goal area
    [ 770, 390],   # 23  bottom-left of right goal area

    [1050,   0],   # 24  top-right corner
    [ 885,   0],   # 25  top, right penalty-area (outer)
    [ 770,   0],   # 26  top, right goal-area (outer)
    [ 770, 680],   # 27  bottom, right goal-area (outer)
    [ 885, 680],   # 28  bottom, right penalty-area (outer)
    [1050, 680],   # 29  bottom-right corner

    [ 525, 340],   # 30  centre spot
    [ 525, 680],   # 31  bottom, centre
], dtype=np.float32)
# fmt: on

# Keypoints whose confidence is below this threshold are ignored.
_KPT_CONF_THRESHOLD: float = 0.5


class HomographyMapper:

    PITCH_W: int = 1050
    PITCH_H: int = 680

    def __init__(self) -> None:
        self.H: np.ndarray | None = None


    def compute(
        self,
        image_keypoints: np.ndarray,
        confidences: np.ndarray | None = None,
    ) -> "np.ndarray | None":

        if image_keypoints is None:
            return None

        image_keypoints = np.asarray(image_keypoints, dtype=np.float32)
        n = len(image_keypoints)

        if n == 0:
            return None

        
        if confidences is not None:
            confidences = np.asarray(confidences, dtype=np.float32)
            valid = (confidences >= _KPT_CONF_THRESHOLD)
        else:
            
            valid = ~((image_keypoints[:, 0] == 0) & (image_keypoints[:, 1] == 0))

        
        max_idx = min(n, len(PITCH_KEYPOINT_COORDS))
        valid_mask = valid[:max_idx]

        src = image_keypoints[:max_idx][valid_mask]         # camera points
        dst = PITCH_KEYPOINT_COORDS[:max_idx][valid_mask]   # known pitch points

        n_valid = len(src)
        if n_valid < 4:
            print(
                f"[HomographyMapper] Only {n_valid} valid keypoints "
                "(need >= 4). Homography not computed."
            )
            return None

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        if H is None:
            print("[HomographyMapper] findHomography returned None.")
            return None

        inliers = int(mask.sum()) if mask is not None else 0
        print(
            f"[HomographyMapper] H computed — "
            f"{inliers}/{n_valid} inliers "
            f"(from {n_valid} valid / {n} total keypoints)"
        )

        if inliers < 4:
            print("[HomographyMapper] Too few RANSAC inliers — H rejected.")
            return None

        return H

    def transform(self, point: "tuple[int, int]") -> "tuple[int, int] | None":

        if self.H is None:
            return None

        p = np.array(
            [[[float(point[0]), float(point[1])]]],
            dtype=np.float32,
        )
        mapped = cv2.perspectiveTransform(p, self.H)

        return (
            int(mapped[0][0][0]),
            int(mapped[0][0][1]),
        )

    def transform_many(
        self,
        points: "list[tuple[int, int]]",
    ) -> "list[tuple[int, int] | None]":
        """Transform a list of points in one call."""
        return [self.transform(pt) for pt in points]