from config import Config
import utils

from tracking import Tracker, FieldDetector
from assigner import Assigner


def run_analyzer(args, config: Config):

    video_frames = utils.read_video(
        config.input_video_path
    )

    field_detector = FieldDetector(
        config.Analyzer.field_model_path,
        config.device
    )

    tracker = Tracker(
        config.Analyzer.player_model_path,
        config.device
    )



    first_frame = video_frames[0]

    field_points = field_detector.detect_keypoints(
        first_frame
    )

    if field_points is not None:

        H = tracker.mapper.compute(
            field_points
        )

        tracker.mapper.H = H

        print(
            f"Detected {len(field_points)} field keypoints"
        )

    else:

        tracker.mapper.H = None

        print(
            "WARNING: No field keypoints detected"
        )



    tracks = tracker.track_detections(
        video_frames
    )

    assigner = Assigner()

    assigner.init_teams(
        video_frames[0],
        tracks["players"][0]
    )



    for frame_num, player_track in enumerate(
        tracks["players"]
    ):

        frame = video_frames[frame_num]

        for track_id, track in player_track.items():

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

        # Goalkeepers
        if (
            "goalkeepers" in tracks
            and frame_num < len(tracks["goalkeepers"])):

            for track_id, gk in tracks["goalkeepers"][frame_num].items():

                pid = assigner.get_or_create_id(
                    frame,
                    gk["bounding_box"],
                    track_id)

                if pid is None:
                    continue

                team = assigner.get_team(pid)

                gk["global_id"] = pid
                gk["team"] = team
                gk["team_color"] = config.colors[team]



    output_frames = tracker.draw_annotations(video_frames,tracks)

    utils.save_video(output_frames,config.output_video_path)

    print("Saved:",config.output_video_path)