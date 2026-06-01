import cv2
import numpy as np



''' 32-keypoint pitch coordinate map , might be wrong tho , check it '''

# Derived by visually inspecting the keypoint debug image against the
# actual pitch markings. Coordinates are in the canonical pitch canvas:
#   width  = 1050 px  (~105 m, 1 m = 10 px)
#   height =  680 px  (~ 68 m)
#
# Left side  = x near 0
# Right side = x near 1050
# Top        = y near 0  (touchline visible at top of broadcast camera)
# Bottom     = y near 680
#
# Keypoints confirmed from debug image:
#
#  idx   visible location in image

#   0    not visible (off frame - top left corner)
#   1    not visible (off frame)
#   2    not visible (off frame)
#   3    left touchline, mid-height area
#   4    left touchline, upper area
#   5    centre spot (bottom half of circle)
#   6    not visible
#   7    not visible
#   8    left touchline, upper-mid
#   9    left touchline, top area
#  10    left touchline, just below 9
#  11    left touchline, lower area
#  12    left touchline, bottom area
#  13    top touchline, left of centre
#  14    top of centre circle (left tangent)
#  15    bottom of centre circle (left tangent)
#  16    not visible
#  17    right penalty area, top
#  18    right side, two locations (top penalty + bottom right)
#  19    not visible
#  20    not visible
#  21    right touchline, mid
#  22    top touchline, right penalty area
#  23    not visible
#  24    top touchline, far right
#  25    top touchline, right of centre-right
#  26    not visible
#  27    not visible
#  28    not visible
#  29    not visible
#  30    left penalty spot
#  31    centre circle, right tangent (bottom half)

PITCH_KEYPOINT_COORDS = np.array([
    #  x     y      idx   location
    [   0,   0],  #  0   top-left corner
    [ 165,   0],  #  1   top touchline, left penalty area outer
    [ 220,   0],  #  2   top touchline, left goal area outer
    [   0, 408],  #  3   left touchline, bottom of left penalty area
    [   0, 272],  #  4   left touchline, top of left penalty area
    [ 525, 408],  #  5   centre spot (detected in bottom-circle area)
    [   0, 680],  #  6   bottom-left corner
    [ 165, 680],  #  7   bottom touchline, left penalty area outer
    [   0, 340],  #  8   left touchline, halfway height
    [   0, 204],  #  9   left touchline, upper area (top of penalty box)
    [   0, 272],  # 10   left touchline, just inside penalty area top
    [ 165, 476],  # 11   left penalty box, bottom-left corner
    [   0, 476],  # 12   left touchline, bottom of penalty area
    [ 525,   0],  # 13   top touchline, centre
    [ 433, 340],  # 14   centre circle, left tangent (top)
    [ 433, 340],  # 15   centre circle, left tangent (bottom)
    [   0, 476],  # 16   left touchline, goal area bottom
    [ 885, 204],  # 17   right penalty box, top-right corner
    [ 885, 272],  # 18   right penalty box inner top
    [ 885, 476],  # 19   right penalty box, bottom-right corner
    [1050, 476],  # 20   right touchline, bottom of right penalty area
    [1050, 340],  # 21   right touchline, mid height
    [ 885,   0],  # 22   top touchline, right penalty area outer
    [ 830,   0],  # 23   top touchline, right goal area outer
    [1050,   0],  # 24   top-right corner
    [ 700,   0],  # 25   top touchline, right of centre
    [ 830, 680],  # 26   bottom touchline, right goal area
    [ 885, 680],  # 27   bottom touchline, right penalty area outer
    [1050, 680],  # 28   bottom-right corner
    [ 525, 680],  # 29   bottom touchline, centre
    [ 110, 340],  # 30   left penalty spot
    [ 617, 340],  # 31   centre circle, right tangent
], dtype=np.float32)


# Keypoints whose confidence is below this are ignored
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
        confidences     : (N,)  float32 — per-keypoint confidence, optional

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

        # Build validity mask
        if confidences is not None:
            confidences = np.asarray(confidences, dtype=np.float32)
            valid = confidences >= _KPT_CONF_THRESHOLD
        else:
            valid = ~((image_keypoints[:, 0] == 0) & (image_keypoints[:, 1] == 0))

        max_idx = min(n, len(PITCH_KEYPOINT_COORDS))
        valid_mask = valid[:max_idx]

        src = image_keypoints[:max_idx][valid_mask]
        dst = PITCH_KEYPOINT_COORDS[:max_idx][valid_mask]

        # Remove duplicate dst coordinates — they confuse RANSAC
        _, unique_idx = np.unique(dst, axis=0, return_index=True)
        src = src[unique_idx]
        dst = dst[unique_idx]

        n_valid = len(src)
        if n_valid < 4:
            print(
                f"[HomographyMapper] Only {n_valid} unique valid keypoints "
                "(need >= 4). H not computed."
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