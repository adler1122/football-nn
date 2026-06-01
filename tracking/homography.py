import cv2
import numpy as np


class HomographyMapper:

    def __init__(self):

        self.pitch_points = np.array([
            [0,0],
            [1050,0],
            [1050,680],
            [0,680]
        ], dtype=np.float32)

    def compute(self, image_keypoints):

        if image_keypoints is None:
            return None

        image_keypoints = np.asarray(
            image_keypoints,
            dtype=np.float32
        )

        if len(image_keypoints) < 4:
            return None

        src = image_keypoints[:4]

        dst = self.pitch_points

        H, _ = cv2.findHomography(
            src,
            dst
        )

        return H

    def transform(self, point, H):

        if H is None:
            return None

        p = np.array(
            [[[point[0], point[1]]]],
            dtype=np.float32
        )

        mapped = cv2.perspectiveTransform(
            p,
            H
        )

        return (
            int(mapped[0][0][0]),
            int(mapped[0][0][1])
        )