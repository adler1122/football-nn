from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from typing import List, Dict, Any

class Tracker:
    """
    Docstring for Tracker
    """
    def __init__(self, model_path: str, device: str):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.tracker: sv.ByteTrack = sv.ByteTrack()

    def __detect(self, frames: List[np.ndarray], conf: float,batch_size: int = 16) -> List[Any]:
        """
        Docstring for detect
        
        :param self: Description
        :param frames: Description
        :type frames: List[np.ndarray]
        :param batch_size: Description
        :type batch_size: int
        :return: Description
        :rtype: List[Any]
        """

        detections = []
        for i in range(0 , len(frames), batch_size):
            batch = frames[i:i+batch_size]
            result = self.model.predict(batch, conf=conf)
            detections += result

        return detections

    def track_detections(self, frames: List[np.ndarray]) -> Dict[str, List[Dict[int, Dict[str, Any]]]]:
        """
        Docstring for track_detections
        
        :param self: Description
        :param frames: Description
        :type frames: List[np.ndarray]
        :return: Description
        :rtype: Dict[str, List[Dict[int, Dict[str, Any]]]]
        """           

        detections = self.__detect(frames,0.1, 20)

        objects = {
            'ball': [],
            'players': [],
            'others': [], # refree or coaches
        }

        for idx, detection in enumerate(detections):
            classficiations: Dict[int, str] = detection.names

            detection_sv = sv.Detections.from_ultralytics(detection)

            objects['players'].append(dict())
            objects['ball'].append(dict())
            objects['others'].append(dict())

            for det in detection_sv:
                classification = det[3]
                bounding_box = det[0]
                class_name = classficiations.get(classification, "")

                if class_name == 'ball':
                    objects['ball'][idx][1] = {"bounding_box": bounding_box}

            detection_frame_tracks = self.tracker.update_with_detections(detection_sv)

            for det in detection_frame_tracks:
                classification = det[3]
                track_id = det[4]
                bounding_box = det[0].tolist()

                class_name = classficiations.get(classification, "")

                match class_name:
                    case "player":
                        objects['players'][idx][track_id] = {"bounding_box": bounding_box}
                    case "goalkeeper":
                        objects['players'][idx][track_id] = {"bounding_box": bounding_box}
                    case _:
                        objects['others'][idx][track_id] = {"bounding_box": bounding_box}

        return objects
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_annotations(self,video_frames, tracks):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["others"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bounding_box"], color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bounding_box"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bounding_box"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bounding_box"],(0,255,0))

            output_video_frames.append(frame)

        return output_video_frames



def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def measure_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0], p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)