import cv2
import numpy as np


class PitchDrawer:

    def __init__(self):

        self.width = 1050
        self.height = 680

    def create_pitch(self):

        pitch = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8
        )

        # grass stripes
        for i in range(0, self.width, 80):

            color = (
                (40, 140, 40)
                if (i // 80) % 2 == 0
                else (30, 120, 30)
            )

            cv2.rectangle(
                pitch,
                (i, 0),
                (i + 80, self.height),
                color,
                -1
            )

        # outer boundary
        cv2.rectangle(
            pitch,
            (0, 0),
            (self.width - 1, self.height - 1),
            (255, 255, 255),
            3
        )

        # halfway line
        cv2.line(
            pitch,
            (self.width // 2, 0),
            (self.width // 2, self.height),
            (255, 255, 255),
            2
        )

        # center circle
        cv2.circle(
            pitch,
            (self.width // 2, self.height // 2),
            80,
            (255, 255, 255),
            2
        )

        # center spot
        cv2.circle(
            pitch,
            (self.width // 2, self.height // 2),
            4,
            (255, 255, 255),
            -1
        )

        return pitch