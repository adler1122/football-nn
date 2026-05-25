from typing import List

import numpy as np
import cv2

def read_video(video_path: str) -> List[np.ndarray]:
    """
    Docstring for read_video
    
    :param video_path: Description
    :type video_path: str
    :return: Description
    :rtype: Tuple[List[ndarray[_AnyShape, dtype[Any]]], float]
    """

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        frames.append(frame)
    return frames

def save_video(frames: List[np.ndarray],video_path: str) -> None:
    """
    Docstring for save_video
    
    :param frames: Description
    :type frames: List[np.ndarray]
    :param video_path: Description
    :type video_path: str
    """

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 24, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()
