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

        pitch[:] = (0, 120, 0)

        cv2.rectangle(
            pitch,
            (0, 0),
            (self.width - 1, self.height - 1),
            (255, 255, 255),
            3
        )

        cv2.line(
            pitch,
            (self.width // 2, 0),
            (self.width // 2, self.height),
            (255, 255, 255),
            2
        )

        cv2.circle(
            pitch,
            (self.width // 2, self.height // 2),
            80,
            (255, 255, 255),
            2
        )

        return pitch