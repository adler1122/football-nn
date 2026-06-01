import cv2
import numpy as np


class PitchDrawer:


    WIDTH  = 1050
    HEIGHT = 680
    _L_PEN_X  = 165
    _L_GA_X   = 220   
    _PEN_Y1   = 138   
    _PEN_Y2   = 541   
    _GA_Y1    = 248   
    _GA_Y2    = 431   
    _L_SPOT_X = 110   
    _R_SPOT_X = 940   

    WHITE = (255, 255, 255)
    LINE_W = 2

    def __init__(self) -> None:
        self.width  = self.WIDTH
        self.height = self.HEIGHT

    def create_pitch(self) -> np.ndarray:
        """Return a freshly drawn 1050 × 680 BGR pitch image."""

        pitch = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)

        
        for i in range(0, self.WIDTH, 80):
            color = (40, 140, 40) if (i // 80) % 2 == 0 else (30, 120, 30)
            cv2.rectangle(pitch, (i, 0), (i + 80, self.HEIGHT), color, -1)

        W = self.WHITE
        LW = self.LINE_W

        
        cv2.rectangle(pitch, (0, 0), (self.WIDTH - 1, self.HEIGHT - 1), W, LW + 1)

        
        mid_x = self.WIDTH // 2
        cv2.line(pitch, (mid_x, 0), (mid_x, self.HEIGHT), W, LW)

        
        centre = (mid_x, self.HEIGHT // 2)
        cv2.circle(pitch, centre, 92, W, LW)
        cv2.circle(pitch, centre, 4,  W, -1)

        
        cv2.rectangle(
            pitch,
            (0,              self._PEN_Y1),
            (self._L_PEN_X,  self._PEN_Y2),
            W, LW,
        )

        
        cv2.rectangle(
            pitch,
            (0,             self._GA_Y1),
            (self._L_GA_X,  self._GA_Y2),
            W, LW,
        )

        
        cv2.circle(pitch, (self._L_SPOT_X, self.HEIGHT // 2), 4, W, -1)

       
        cv2.ellipse(
            pitch,
            (self._L_SPOT_X, self.HEIGHT // 2),
            (92, 92),
            0,
            -53, 53,   
            W, LW,
        )

        
        cv2.rectangle(
            pitch,
            (self.WIDTH - self._L_PEN_X, self._PEN_Y1),
            (self.WIDTH,                  self._PEN_Y2),
            W, LW,
        )

        
        cv2.rectangle(
            pitch,
            (self.WIDTH - self._L_GA_X, self._GA_Y1),
            (self.WIDTH,                 self._GA_Y2),
            W, LW,
        )

        
        cv2.circle(pitch, (self._R_SPOT_X, self.HEIGHT // 2), 4, W, -1)

        
        cv2.ellipse(
            pitch,
            (self._R_SPOT_X, self.HEIGHT // 2),
            (92, 92),
            0,
            127, 233,
            W, LW,
        )

        
        goal_h = 73   
        goal_y1 = (self.HEIGHT - goal_h) // 2
        goal_y2 = goal_y1 + goal_h
        goal_d  = 20   

        cv2.rectangle(pitch, (0,                    goal_y1), (goal_d, goal_y2), W, LW)
        cv2.rectangle(pitch, (self.WIDTH - goal_d,  goal_y1), (self.WIDTH, goal_y2), W, LW)

        return pitch