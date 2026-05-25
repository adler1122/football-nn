from dataclasses import dataclass, field
from typing import List, Tuple
import torch

@dataclass
class Config:
    """
    Docstring for Config
    """
    input_video_path: str = "./input_video.mp4"
    output_video_path: str = "./output_video.avi"

    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

    colors: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [
            (0, 215, 255),   # referees/coaches – amber
            (50, 205, 50),   # team 1 – green
            (60, 20, 220),   # team 2 – deep red
        ]
    )

    class Train:
        """
        Docstring for Train
        """
        batch_size: int = 8
        yolo_base_model = 'yolov8s.pt'
        train_data_path = './data/football-players-detection'
        imgsz = 640
        conf: float = 0.1
        cache_dir: str = "cache"
        use_cache: bool = True

    class Analyzer:
        """
        Docstring for Analyzer
        """

        player_model_path = 'models/player_best.pt'
