import cv2
import numpy as np



_PEN_Y1  = 138
_PEN_Y2  = 541
_GOAL_Y1 = 248
_GOAL_Y2 = 431
_L_ARC_TOP    = 268
_L_ARC_BOTTOM = 412

PITCH_KEYPOINT_COORDS = np.array([
    [    0,    0],   #  0  top-left corner
    [    0,  138],   #  1  top-left of left 18-box (touchline)
    [    0,  248],   #  2  top-left of left 6-box (touchline)
    [    0,  431],   #  3  bottom-left of left 6-box (touchline)
    [    0,  541],   #  4  bottom-left of left 18-box (touchline)
    [    0,  680],   #  5  bottom-left corner
    [   55,  248],   #  6  top-right of left 6-box (inside)
    [   55,  431],   #  7  bottom-right of left 6-box (inside)
    [  110,  340],   #  8  left penalty spot
    [  165,  138],   #  9  top-right of left 18-box (inside)
    [  165,  268],   # 10  top of left penalty arc
    [  165,  412],   # 11  bottom of left penalty arc
    [  165,  541],   # 12  bottom-right of left 18-box (inside)
    [  525,    0],   # 13  top of halfway line (touchline)
    [  525,  249],   # 14  halfway line — top of centre circle
    [  525,  431],   # 15  halfway line — bottom of centre circle
    [  525,  680],   # 16  bottom of halfway line (touchline)
    [  885,  138],   # 17  top-left of right 18-box (inside)
    [  885,  268],   # 18  top of right penalty arc
    [  885,  412],   # 19  bottom of right penalty arc
    [  885,  541],   # 20  bottom-left of right 18-box (inside)
    [  940,  340],   # 21  right penalty spot
    [  995,  248],   # 22  top-left of right 6-box (inside)
    [  995,  431],   # 23  bottom-left of right 6-box (inside)
    [ 1050,    0],   # 24  top-right corner
    [ 1050,  138],   # 25  top-right of right 18-box (touchline)
    [ 1050,  248],   # 26  top-right of right 6-box (touchline)
    [ 1050,  431],   # 27  bottom-right of right 6-box (touchline)
    [ 1050,  541],   # 28  bottom-right of right 18-box (touchline)
    [ 1050,  680],   # 29  bottom-right corner
    [  201,  340],   # 30  left penalty arc midpoint
    [  849,  340],   # 31  right penalty arc midpoint
], dtype=np.float32)

_KPT_CONF_THRESHOLD: float = 0.5
_RANSAC_THRESHOLD:   float = 30.0
_MIN_INLIERS:        int   = 4


class HomographyMapper:
    """
    maps image pixel coordinates to 2-D pitch canvas coordinates.

    maintains a base homography H computed from field keypoints, and
    refines it every frame using optical flow on pitch pixels — so
    camera pans, tilts, and zooms are compensated automatically.

    Usage
    
    mapper = HomographyMapper()
    mapper.H = mapper.compute(image_keypoints, confidences)

    # Every frame:
    mapper.update_with_optical_flow(prev_frame, curr_frame)
    px, py = mapper.transform((foot_x, foot_y))
    """

    PITCH_W: int = 1050
    PITCH_H: int = 680

    # Optical flow parameters
    _LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    _FEATURE_PARAMS = dict(
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7,
    )


    _REFRESH_INTERVAL = 30


    _MAX_FLOW_CORRECTION = 80.0

    def __init__(self) -> None:
        self.H: np.ndarray | None = None          # current best H
        self._H_base: np.ndarray | None = None    # H from keypoint detection
        self._prev_gray: np.ndarray | None = None # previous frame (grayscale)
        self._prev_pts: np.ndarray | None = None  # tracked feature points
        self._frame_count: int = 0



    def compute(
        self,
        image_keypoints: np.ndarray,
        confidences: np.ndarray | None = None,
    ) -> "np.ndarray | None":
        """
        Estimate homography from camera space to pitch canvas using
        detected field keypoints.
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

        n_valid = len(src)
        if n_valid < _MIN_INLIERS:
            print(f"[HomographyMapper] Only {n_valid} valid keypoints — need >= {_MIN_INLIERS}.")
            return None


        H_ransac, mask = cv2.findHomography(src, dst, cv2.RANSAC, _RANSAC_THRESHOLD)
        inliers = int(mask.sum()) if mask is not None else 0
        print(f"[HomographyMapper] RANSAC: {inliers}/{n_valid} inliers")

        if H_ransac is not None and inliers >= max(_MIN_INLIERS, n_valid // 2):
            inlier_mask = mask.ravel().astype(bool)
            H_final, _ = cv2.findHomography(src[inlier_mask], dst[inlier_mask], 0)
            if H_final is not None and self._sanity_check(H_final):
                print(f"[HomographyMapper] H from RANSAC inliers ({inliers} pts).")
                return H_final


        print("[HomographyMapper] Falling back to least-squares.")
        H_ls, _ = cv2.findHomography(src, dst, 0)
        if H_ls is not None and self._sanity_check(H_ls):
            print(f"[HomographyMapper] H from least-squares ({n_valid} pts).")
            return H_ls

        print("[HomographyMapper] H computation failed.")
        return None



    def update_with_optical_flow(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> None:
        """
        refine self.H using sparse optical flow between prev_frame and
        curr_frame to compensate for camera movement.

        call this every frame BEFORE calling transform().

        parameters
        ----------
        prev_frame : BGR frame from the previous video frame
        curr_frame : BGR frame from the current video frame
        """
        if self.H is None:
            return

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        
        if (
            self._prev_gray is None
            or self._prev_pts is None
            or self._frame_count % self._REFRESH_INTERVAL == 0
        ):
            self._prev_pts = self._detect_pitch_features(prev_frame)
            self._prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        if self._prev_pts is None or len(self._prev_pts) < 8:
            self._prev_gray = curr_gray
            self._frame_count += 1
            return

        
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            curr_gray,
            self._prev_pts,
            None,
            **self._LK_PARAMS,
        )

        if curr_pts is None:
            self._prev_gray = curr_gray
            self._frame_count += 1
            return

        
        good_prev = self._prev_pts[status.ravel() == 1]
        good_curr = curr_pts[status.ravel() == 1]

        if len(good_prev) < 8:
           
            self._prev_pts = None
            self._prev_gray = curr_gray
            self._frame_count += 1
            return


        H_motion, motion_mask = cv2.findHomography(
            good_prev, good_curr, cv2.RANSAC, 3.0
        )

        if H_motion is None:
            self._prev_gray = curr_gray
            self._frame_count += 1
            return


        correction = np.linalg.norm(H_motion - np.eye(3))
        if correction > self._MAX_FLOW_CORRECTION:
            print(f"[HomographyMapper] Optical flow correction too large ({correction:.1f}) — skipped.")
            self._prev_gray = curr_gray
            self._frame_count += 1
            return


        try:
            H_motion_inv = np.linalg.inv(H_motion)
            H_updated = self.H @ H_motion_inv

            if self._sanity_check(H_updated):
                self.H = H_updated
        except np.linalg.LinAlgError:
            pass


        self._prev_gray = curr_gray
        self._prev_pts  = good_curr.reshape(-1, 1, 2)
        self._frame_count += 1



    def transform(self, point: "tuple[int, int]") -> "tuple[int, int] | None":
        """map a single camera-space point to pitch canvas coords."""
        if self.H is None:
            return None
        p = np.array(
            [[[float(point[0]), float(point[1])]]],
            dtype=np.float32,
        )
        mapped = cv2.perspectiveTransform(p, self.H)
        return (int(mapped[0][0][0]), int(mapped[0][0][1]))

    def transform_many(
        self, points: "list[tuple[int, int]]"
    ) -> "list[tuple[int, int] | None]":
        return [self.transform(pt) for pt in points]



    def _detect_pitch_features(
        self, frame: np.ndarray
    ) -> "np.ndarray | None":
        """
        detect good features to track on the pitch area of the frame.

        create a mask that focuses on the green grass area to avoid
        tracking players, crowd, or scoreboard elements — only stable
        pitch features (line edges, grass texture) are tracked.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape


        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))


        kernel = np.ones((15, 15), np.uint8)
        mask   = cv2.erode(mask, kernel, iterations=1)

        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self._FEATURE_PARAMS)
        return pts

    def _sanity_check(self, H: np.ndarray) -> bool:
        """check H is not degenerate and maps pitch to a sensible image region."""
        if H is None:
            return False
        if abs(np.linalg.det(H)) < 1e-10:
            return False

        corners = np.array([
            [[0.,    0.]],
            [[1050., 0.]],
            [[1050., 680.]],
            [[0.,    680.]],
        ], dtype=np.float32)

        try:
            H_inv  = np.linalg.inv(H)
            mapped = cv2.perspectiveTransform(corners, H_inv)
        except Exception:
            return False

        if not np.all(np.isfinite(mapped)):
            return False

        xs = mapped[:, 0, 0]
        ys = mapped[:, 0, 1]
        if (xs.max() - xs.min()) < 100 or (ys.max() - ys.min()) < 100:
            return False

        return True