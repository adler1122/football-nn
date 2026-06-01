from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from typing import List, Dict, Any

from .homography import HomographyMapper
from .pitch import PitchDrawer


class Tracker:

    def __init__(self, model_path: str, device: str) -> None:
        self.model = YOLO(model_path)
        self.model.to(device)
        self.tracker = sv.ByteTrack()
        self.mapper = HomographyMapper()
        self.pitch_drawer = PitchDrawer()


    def __detect(
        self,
        frames: List[np.ndarray],
        conf: float,
        batch_size: int = 16,
    ) -> List[Any]:
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i: i + batch_size]
            results = self.model.predict(batch, conf=conf)
            detections += results
        return detections


    def track_detections(
        self, frames: List[np.ndarray]
    ) -> Dict[str, List[Dict[int, Dict[str, Any]]]]:

        raw_detections = self.__detect(frames, conf=0.3, batch_size=20)

        objects: Dict[str, list] = {
            "ball":        [],
            "players":     [],
            "goalkeepers": [],   
            "others":      [],   
        }

        for idx, detection in enumerate(raw_detections):
            class_names: Dict[int, str] = detection.names

            detection_sv = sv.Detections.from_ultralytics(detection)

            
            objects["ball"].append({})
            objects["players"].append({})
            objects["goalkeepers"].append({})
            objects["others"].append({})

            
            for det in detection_sv:
                cls_id   = det[3]
                bbox     = det[0]
                cls_name = class_names.get(cls_id, "")

                if cls_name == "ball":
                    
                    objects["ball"][idx][1] = {
                        "bounding_box": bbox.tolist()   
                    }
                    break   

           
            tracked = self.tracker.update_with_detections(detection_sv)

            for det in tracked:
                cls_id   = det[3]
                track_id = det[4]
                bbox     = det[0].tolist()
                cls_name = class_names.get(cls_id, "")

                match cls_name:
                    case "player":
                        objects["players"][idx][track_id] = {
                            "bounding_box": bbox
                        }
                    case "goalkeeper":

                        objects["goalkeepers"][idx][track_id] = {
                            "bounding_box": bbox
                        }
                    case _:
                        objects["others"][idx][track_id] = {
                            "bounding_box": bbox
                        }

        return objects


    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2       = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width    = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rect_w = 40
        rect_h = 20
        x1_rect = x_center - rect_w // 2
        x2_rect = x_center + rect_w // 2
        y1_rect = (y2 - rect_h // 2) + 15
        y2_rect = (y2 + rect_h // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,      y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color,  cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    draw_traingle = draw_triangle



    def draw_birds_eye_view(
        self,
        frame: np.ndarray,
        player_dict: dict,
        ball_dict: dict,
        goalkeeper_dict: dict | None = None,
    ) -> np.ndarray:

        pitch = self.pitch_drawer.create_pitch()

    
        for track_id, player in player_dict.items():
            foot = get_foot_position(player["bounding_box"])
            mapped = self.mapper.transform(foot)   
            if mapped is None:
                continue

            x = max(0, min(HomographyMapper.PITCH_W - 1, mapped[0]))
            y = max(0, min(HomographyMapper.PITCH_H - 1, mapped[1]))

            color = player.get("team_color", (0, 0, 255))
            cv2.circle(pitch, (x, y), 12, color,       -1)
            cv2.circle(pitch, (x, y), 12, (255, 255, 255), 2)

    
        if goalkeeper_dict:
            for track_id, gk in goalkeeper_dict.items():
                foot = get_foot_position(gk["bounding_box"])
                mapped = self.mapper.transform(foot)
                if mapped is None:
                    continue

                x = max(0, min(HomographyMapper.PITCH_W - 1, mapped[0]))
                y = max(0, min(HomographyMapper.PITCH_H - 1, mapped[1]))

                color = gk.get("team_color", (180, 0, 180))
                cv2.circle(pitch, (x, y), 12, color,       -1)
                cv2.circle(pitch, (x, y), 12, (255, 255, 255), 2)

        
        for _, ball in ball_dict.items():
            center = get_center_of_bbox(ball["bounding_box"])
            mapped = self.mapper.transform(center)
            if mapped is None:
                continue

            bx = max(0, min(HomographyMapper.PITCH_W - 1, mapped[0]))
            by = max(0, min(HomographyMapper.PITCH_H - 1, mapped[1]))

            cv2.circle(pitch, (bx, by), 8, (0, 255, 255), -1)
            cv2.circle(pitch, (bx, by), 8, (255, 255, 255), 2)

      
        mini = cv2.resize(pitch, (600, 380))
        h, w = mini.shape[:2]
        cv2.rectangle(mini, (0, 0), (w - 1, h - 1), (0, 0, 0), 4)

        ox = frame.shape[1] - w - 20
        oy = frame.shape[0] - h - 20

        overlay = frame.copy()
        overlay[oy: oy + h, ox: ox + w] = mini
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        return frame



    def draw_annotations(
        self,
        video_frames: List[np.ndarray],
        tracks: Dict[str, list],
    ) -> List[np.ndarray]:

        output_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict     = tracks["players"][frame_num]
            ball_dict       = tracks["ball"][frame_num]
            referee_dict    = tracks["others"][frame_num]
            goalkeeper_dict = tracks.get("goalkeepers", [{}] * len(video_frames))[frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bounding_box"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bounding_box"], (0, 0, 255))

            # Draw goalkeepers
            for track_id, gk in goalkeeper_dict.items():
                color = gk.get("team_color", (180, 0, 180))
                frame = self.draw_ellipse(frame, gk["bounding_box"], color, track_id)

            # Draw referees / others
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bounding_box"], (0, 255, 255))

            # Draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bounding_box"], (0, 255, 0))

            # Draw bird's-eye minimap
            frame = self.draw_birds_eye_view(
                frame,
                player_dict,
                ball_dict,
                goalkeeper_dict,
            )

            output_frames.append(frame)

        return output_frames




def get_center_of_bbox(bbox) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox) -> float:
    return bbox[2] - bbox[0]


def get_foot_position(bbox) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def measure_distance(p1, p2) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(p1, p2) -> tuple[float, float]:
    return p1[0] - p2[0], p1[1] - p2[1]