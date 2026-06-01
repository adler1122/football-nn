from dataclasses import dataclass, field
from typing import Dict, Tuple
import torch


@dataclass
class Config:


    input_video_path: str = "./test.mp4"
    output_video_path: str = "./output_video.avi"

    device: str = "mps" if torch.backends.mps.is_available() else "cpu"


    colors: Dict[int, Tuple[int, int, int]] = field(
        default_factory=lambda: {
            0: (0,   215, 255),   # yellow  unknown / referee fallback
            1: (50,  205,  50),   # green   team 1
            2: (60,   20, 220),   # red     team 2
        }
    )

    class Train:
        batch_size: int = 8

        player_base_model: str = "yolov8s.pt"
        field_base_model: str  = "yolov8s-pose.pt"

        player_data_path: str = "./data/football-players-detection"
        field_data_path: str  = "./data/football-field-detection"

        imgsz: int = 640
        conf: float = 0.1

        cache_dir: str = "cache"
        use_cache: bool = True

    class Analyzer:
        player_model_path: str = "models/player_best.pt"
        field_model_path: str  = "models/field_best.pt"