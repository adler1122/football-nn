import cv2
import numpy as np


class HomographyMapper:

    def __init__(self):

        self.image_points = np.float32([
            [120, 110],
            [1180, 110],
            [1260, 650],
            [50, 650]
        ])

        self.pitch_points = np.float32([
            [0, 0],
            [1050, 0],
            [1050, 680],
            [0, 680]
        ])

        self.H = cv2.getPerspectiveTransform(
            self.image_points,
            self.pitch_points
        )

    def transform(self, point):

        p = np.array(
            [[[point[0], point[1]]]],
            dtype=np.float32
        )

        mapped = cv2.perspectiveTransform(
            p,
            self.H
        )

        return (
            int(mapped[0][0][0]),
            int(mapped[0][0][1])
        )