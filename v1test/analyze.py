from config import Config
import utils

from tracking import Tracker, FieldDetector
from assigner import Assigner


def run_analyzer(args, config: Config) -> None:
    """
    Full analysis pipeline:
      1. Read video frames.
      2. Detect field keypoints on the first frame → compute homography H.
      3. Track players, goalkeepers, ball, and others across all frames.
      4. Assign every player / goalkeeper to a team via jersey colour.
      5. Annotate frames and write output video.
    """
    # the problem with homography might be the step 2

    video_frames = utils.read_video(config.input_video_path)
    print(f"Loaded {len(video_frames)} frames from {config.input_video_path}")


    field_detector = FieldDetector(
        config.Analyzer.field_model_path,
        config.device,
    )

    tracker = Tracker(
        config.Analyzer.player_model_path,
        config.device,
    )

    first_frame = video_frames[0]


    kpt_result = field_detector.detect_keypoints_with_confidence(first_frame)

    if kpt_result is not None:
        field_points, field_confidences = kpt_result
        H = tracker.mapper.compute(field_points, confidences=field_confidences)
    else:

        field_points = field_detector.detect_keypoints(first_frame)
        H = tracker.mapper.compute(field_points) if field_points is not None else None

    tracker.mapper.H = H

    if H is not None:
        n_pts = len(field_points) if field_points is not None else 0
        print(f"Homography computed from {n_pts} keypoints.")
    else:
        print("WARNING: Homography failed — bird's-eye view will be disabled.")


    tracks = tracker.track_detections(video_frames)


    assigner = Assigner()



    bootstrap_players = {}

    for frame_idx in range(min(30, len(video_frames))):

        for track_id, player in tracks["players"][frame_idx].items():

            if track_id not in bootstrap_players:
                bootstrap_players[track_id] = player

    assigner.init_teams(
        video_frames[0],
        bootstrap_players)
    

    def resolve_color(team: int) -> tuple:
        return config.colors.get(team, config.colors.get(0, (128, 128, 128)))


    for frame_num, player_track in enumerate(tracks["players"]):
        frame = video_frames[frame_num]

        for track_id, track in player_track.items():
            pid = assigner.get_or_create_id(
                frame,
                track["bounding_box"],
                track_id,
            )
            if pid is None:
                continue

            team = assigner.get_team(pid)   

            track["global_id"]  = pid
            track["team"]       = team
            track["team_color"] = resolve_color(team)


    for frame_num, gk_track in enumerate(tracks["goalkeepers"]):
        frame = video_frames[frame_num]

        for track_id, gk in gk_track.items():
            pid = assigner.get_or_create_id(
                frame,
                gk["bounding_box"],
                track_id,
            )
            if pid is None:
                continue

            team = assigner.get_team(pid)

            gk["global_id"]  = pid
            gk["team"]       = team
            gk["team_color"] = resolve_color(team)


    output_frames = tracker.draw_annotations(video_frames, tracks)
    utils.save_video(output_frames, config.output_video_path)

    print(f"Saved: {config.output_video_path}")