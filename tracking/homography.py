import cv2
import numpy as np



# Measurements used:
#   Penalty area depth : 16.5m = 165px
#   Goal area depth    :  5.5m =  55px
#   Penalty area width : 40.3m = 403px  → y: 138 to 541  (centred on 340)
#   Goal area width    : 18.3m = 183px  → y: 248 to 431  (centred on 340)
#   Penalty spot       : 11.0m = 110px  from goal line
#   Centre circle r    :  9.15m = 91px
#   Penalty arc r      :  9.15m = 91px  from penalty spot

#  0  top-left corner
#  1  top-left corner of left 18-box on touchline
#  2  top-left corner of left 6-box on touchline
#  3  bottom-left corner of left 6-box on touchline
#  4  bottom-left corner of left 18-box on touchline
#  5  bottom-left corner
#  6  top-right corner of left 6-box (inside pitch)
#  7  bottom-right corner of left 6-box (inside pitch)
#  8  left penalty spot
#  9  top-right corner of left 18-box (inside pitch)
# 10  top of penalty arc next to left 18-box (bulges toward centre)
# 11  bottom of penalty arc next to left 18-box
# 12  bottom-right corner of left 18-box (inside pitch)
# 13  top of halfway line (outside touchline)
# 14  halfway line — top intersection with centre circle
# 15  halfway line — bottom intersection with centre circle
# 16  bottom of halfway line (outside touchline)
# 17  top-left corner of right 18-box (inside pitch)
# 18  top of penalty arc next to right 18-box
# 19  bottom of penalty arc next to right 18-box
# 20  bottom-left corner of right 18-box (inside pitch)
# 21  right penalty spot
# 22  top-left corner of right 6-box (inside pitch)
# 23  bottom-left corner of right 6-box (inside pitch)
# 24  top-right corner
# 25  top-right corner of right 18-box on touchline
# 26  top-right corner of right 6-box on touchline
# 27  bottom-right corner of right 6-box on touchline
# 28  bottom-right corner of right 18-box on touchline
# 29  bottom-right corner
# 30  middle of left penalty arc (leftmost point)
# 31  middle of right penalty arc (rightmost point)


_PEN_Y1  = 138   # (680 - 403) // 2
_PEN_Y2  = 541   # 138 + 403
_GOAL_Y1 = 248   # (680 - 183) // 2
_GOAL_Y2 = 431   # 248 + 183


_L_ARC_X = 110 + 91   # 201
_R_ARC_X = 940 - 91   # 849


_L_ARC_Y_OFFSET = int((91**2 - (165 - 110)**2) ** 0.5)  # 72
_L_ARC_TOP    = 340 - _L_ARC_Y_OFFSET   # 268  — top of arc where it exits box
_L_ARC_BOTTOM = 340 + _L_ARC_Y_OFFSET   # 412  — bottom of arc where it exits box
_R_ARC_TOP    = 340 - _L_ARC_Y_OFFSET   # 268
_R_ARC_BOTTOM = 340 + _L_ARC_Y_OFFSET   # 412

PITCH_KEYPOINT_COORDS = np.array([
    #   x      y
    [    0,    0],   #  0  top-left corner
    [    0,  138],   #  1  top-left of left 18-box (touchline)
    [    0,  248],   #  2  top-left of left 6-box (touchline)
    [    0,  431],   #  3  bottom-left of left 6-box (touchline)
    [    0,  541],   #  4  bottom-left of left 18-box (touchline)
    [    0,  680],   #  5  bottom-left corner
    [   55,  248],   #  6  top-right of left 6-box (inside pitch)
    [   55,  431],   #  7  bottom-right of left 6-box (inside pitch)
    [  110,  340],   #  8  left penalty spot
    [  165,  138],   #  9  top-right of left 18-box (inside pitch)
    [  165, _L_ARC_TOP],    # 10  top of left penalty arc (where it leaves box)
    [  165, _L_ARC_BOTTOM], # 11  bottom of left penalty arc (where it leaves box)
    [  165,  541],   # 12  bottom-right of left 18-box (inside pitch)
    [  525,    0],   # 13  top of halfway line (touchline)
    [  525,  249],   # 14  halfway line — top of centre circle
    [  525,  431],   # 15  halfway line — bottom of centre circle
    [  525,  680],   # 16  bottom of halfway line (touchline)
    [  885,  138],   # 17  top-left of right 18-box (inside pitch)
    [  885, _R_ARC_TOP],    # 18  top of right penalty arc
    [  885, _R_ARC_BOTTOM], # 19  bottom of right penalty arc
    [  885,  541],   # 20  bottom-left of right 18-box (inside pitch)
    [  940,  340],   # 21  right penalty spot
    [  995,  248],   # 22  top-left of right 6-box (inside pitch)
    [  995,  431],   # 23  bottom-left of right 6-box (inside pitch)
    [ 1050,    0],   # 24  top-right corner
    [ 1050,  138],   # 25  top-right of right 18-box (touchline)
    [ 1050,  248],   # 26  top-right of right 6-box (touchline)
    [ 1050,  431],   # 27  bottom-right of right 6-box (touchline)
    [ 1050,  541],   # 28  bottom-right of right 18-box (touchline)
    [ 1050,  680],   # 29  bottom-right corner
    [_L_ARC_X, 340], # 30  left penalty arc midpoint (leftmost point)
    [_R_ARC_X, 340], # 31  right penalty arc midpoint (rightmost point)
], dtype=np.float32)



_KPT_CONF_THRESHOLD: float = 0.5


class HomographyMapper:
    """
    Maps image pixel coordinates to 2-D pitch canvas coordinates using
    a perspective homography estimated from up to 32 labelled keypoints.

    Pitch canvas: 1050 x 680 px.

    Usage
    -----
    mapper = HomographyMapper()
    mapper.H = mapper.compute(image_keypoints, confidences)
    px, py  = mapper.transform((foot_x, foot_y))
    """

    PITCH_W: int = 1050
    PITCH_H: int = 680

    def __init__(self) -> None:
        self.H: np.ndarray | None = None

    def compute(
        self,
        image_keypoints: np.ndarray,
        confidences: np.ndarray | None = None,
    ) -> "np.ndarray | None":
        """
        Estimate homography from camera space to pitch canvas.

        Parameters
        ----------
        image_keypoints : (N, 2) float32 — pixel coords from FieldDetector
        confidences     : (N,) float32  — per-keypoint confidence, optional

        Returns
        -------
        (3, 3) float32 homography matrix or None on failure.
        """
        if image_keypoints is None:
            return None

        image_keypoints = np.asarray(image_keypoints, dtype=np.float32)
        n = len(image_keypoints)
        if n == 0:
            return None

        
        if confidences is not None:
            confidences = np.asarray(confidences, dtype=np.float32)
            valid = confidences >= _KPT_CONF_THRESHOLD
        else:
            valid = ~((image_keypoints[:, 0] == 0) & (image_keypoints[:, 1] == 0))

        max_idx    = min(n, len(PITCH_KEYPOINT_COORDS))
        valid_mask = valid[:max_idx]

        src = image_keypoints[:max_idx][valid_mask]
        dst = PITCH_KEYPOINT_COORDS[:max_idx][valid_mask]

        if len(src) < 4:
            print(f"[HomographyMapper] Only {len(src)} valid keypoints — need >= 4.")
            return None

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        if H is None:
            print("[HomographyMapper] findHomography returned None.")
            return None

        inliers = int(mask.sum()) if mask is not None else 0
        print(
            f"[HomographyMapper] H computed — "
            f"{inliers}/{len(src)} inliers "
            f"({n} total keypoints from model)"
        )

        if inliers < 4:
            print("[HomographyMapper] Too few inliers — H rejected.")
            return None

        return H

    def transform(self, point: "tuple[int, int]") -> "tuple[int, int] | None":
        """Map a single camera-space point to pitch canvas coords."""
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
        self, points: "list[tuple[int, int]]"
    ) -> "list[tuple[int, int] | None]":
        return [self.transform(pt) for pt in points]