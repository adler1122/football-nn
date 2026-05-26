from config import Config
import utils

from tracking import Tracker
from assigner import Assigner


def run_analyzer(args, config: Config):

    video_frames = utils.read_video(config.input_video_path)

    tracker = Tracker(
        config.Analyzer.player_model_path,
        config.device
    )

    tracks = tracker.track_detections(video_frames)

    assigner = Assigner()

    # -----------------------------
    # TRAIN ONCE (FIXED PROPERLY)
    # -----------------------------
    assigner.assign_team(video_frames, tracks)

    # -----------------------------
    # ASSIGN TEAMS
    # -----------------------------
    for frame_num, player_track in enumerate(tracks["players"]):

        frame = video_frames[frame_num]

        for player_id, track in player_track.items():

            team = assigner.get_player_team(
                frame,
                track["bounding_box"],
                player_id
            )

            track["team"] = team
            track["team_color"] = config.colors[team]

        # -------------------------
        # GOALKEEPERS
        # -------------------------
        if "goalkeepers" in tracks and frame_num < len(tracks["goalkeepers"]):

            gk_track = tracks["goalkeepers"][frame_num]

            for gk_id, gk in gk_track.items():

                gx = (gk["bounding_box"][0] + gk["bounding_box"][2]) / 2
                gy = gk["bounding_box"][3]

                best_team = 1
                best_dist = float("inf")

                for pid, player in player_track.items():

                    pb = player["bounding_box"]

                    px = (pb[0] + pb[2]) / 2
                    py = pb[3]

                    dist = (gx - px) ** 2 + (gy - py) ** 2

                    if dist < best_dist:
                        best_dist = dist
                        best_team = player.get("team", 1)

                gk["team"] = best_team
                gk["team_color"] = config.colors[best_team]

    # -----------------------------
    # OUTPUT
    # -----------------------------
    output_frames = tracker.draw_annotations(video_frames, tracks)

    utils.save_video(output_frames, config.output_video_path)

    print("Saved:", config.output_video_path)