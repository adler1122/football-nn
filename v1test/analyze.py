from config import Config
import utils

from tracking import Tracker , FieldDetector
from assigner import Assigner


def run_analyzer(args, config: Config):

    video_frames = utils.read_video(config.input_video_path)
    field_detector = FieldDetector(
    config.Analyzer.field_model_path,
    config.device
    )
    tracker = Tracker(
        config.Analyzer.player_model_path,
        config.device
    )

    tracks = tracker.track_detections(video_frames)

    assigner = Assigner()

   
    assigner.init_teams(video_frames[0], tracks["players"][0])

    
    for frame_num, player_track in enumerate(tracks["players"]):

        frame = video_frames[frame_num]

        for track_id , track in player_track.items():

            pid = assigner.get_or_create_id(
                frame,
                track["bounding_box"],
                track_id
            )

            if pid is None:
                continue

            team = assigner.get_team(pid)

            track["global_id"] = pid
            track["team"] = team
            track["team_color"] = config.colors[team]

        
        if "goalkeepers" in tracks and frame_num < len(tracks["goalkeepers"]):

            for _, gk in tracks["goalkeepers"][frame_num].items():

                pid = assigner.get_or_create_id(
                    frame,
                    gk["bounding_box"]
                )

                if pid is None:
                    continue

                team = assigner.get_team(pid)

                gk["global_id"] = pid
                gk["team"] = team
                gk["team_color"] = config.colors[team]

    output_frames = tracker.draw_annotations(video_frames, tracks)

    utils.save_video(output_frames, config.output_video_path)

    print("Saved:", config.output_video_path)