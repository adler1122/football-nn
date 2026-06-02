from config import Config
import utils

from tracking import Tracker, FieldDetector
from assigner import Assigner
import numpy as np


def _compute_best_homography(field_detector, frames, mapper):
    """
    Scan every 10 frames across the entire video, building a composite
    keypoint set that keeps the highest-confidence detection seen so far
    for each of the 32 indices.

    Stops early if all 32 keypoints have conf >= 0.5.

    Returns
    -------
    H              : (3,3) homography matrix or None
    best_frame_idx : frame index that triggered the final H computation
    n_good         : number of keypoints with conf >= 0.5 used
    """
    CONF_MIN      = 0.5
    SAMPLE_EVERY  = 10
    N_KPT         = 32

    # Composite arrays — best detection seen so far per keypoint index
    best_pts  = np.zeros((N_KPT, 2), dtype=np.float32)
    best_conf = np.zeros(N_KPT,      dtype=np.float32)

    best_H          = None
    best_n_good     = 0
    best_frame_idx  = 0

    total_frames = len(frames)
    scanned      = 0

    for i in range(0, total_frames, SAMPLE_EVERY):
        result = field_detector.detect_keypoints_with_confidence(frames[i])
        if result is None:
            continue

        pts, conf = result   # both shape (32,) after FieldDetector filtering
        scanned += 1

        # Update composite: keep whichever detection has higher confidence
        for idx in range(min(N_KPT, len(pts))):
            if conf[idx] > best_conf[idx]:
                best_conf[idx] = conf[idx]
                best_pts[idx]  = pts[idx]

        n_good = int((best_conf >= CONF_MIN).sum())

        # Try to compute H from the current composite keypoint set
        H = mapper.compute(best_pts, confidences=best_conf)

        if H is not None and n_good > best_n_good:
            best_H         = H
            best_n_good    = n_good
            best_frame_idx = i

        print(
            f"  [scan] frame {i:04d} — "
            f"{n_good}/32 keypoints >= 0.5 conf"
        )

        # Early exit: all 32 keypoints confidently detected
        if n_good == N_KPT:
            print(f"  [scan] All 32 keypoints found — stopping early.")
            break

    print(
        f"Scanned {scanned} frames — best H from frame {best_frame_idx} "
        f"({best_n_good}/32 high-confidence keypoints)."
    )

    return best_H, best_frame_idx, best_n_good


def run_analyzer(args, config: Config) -> None:
    """
    Full analysis pipeline:
      1. Read video frames.
      2. Scan every 10 frames to build composite best-confidence keypoints → H.
      3. Track players, goalkeepers, ball, and others across all frames.
      4. Assign every player / goalkeeper to a team via jersey colour.
      5. Annotate frames and write output video.
    """

    # ------------------------------------------------------------------
    # 1. Load video
    # ------------------------------------------------------------------
    video_frames = utils.read_video(config.input_video_path)
    print(f"Loaded {len(video_frames)} frames from {config.input_video_path}")

    # ------------------------------------------------------------------
    # 2. Field detection → best composite homography
    # ------------------------------------------------------------------
    field_detector = FieldDetector(
        config.Analyzer.field_model_path,
        config.device,
    )

    tracker = Tracker(
        config.Analyzer.player_model_path,
        config.device,
    )

    print("Scanning video for best keypoint detections...")
    H, best_idx, n_good = _compute_best_homography(
        field_detector,
        video_frames,
        tracker.mapper,
    )

    tracker.mapper.H = H

    if H is not None:
        print(f"Homography ready — {n_good}/32 keypoints contributed.")
    else:
        print("WARNING: Homography failed — bird's-eye view will be disabled.")

    # ------------------------------------------------------------------
    # 3. Track all objects across the full video
    # ------------------------------------------------------------------
    tracks = tracker.track_detections(video_frames)

    # ------------------------------------------------------------------
    # 4. Team assignment
    # ------------------------------------------------------------------
    assigner = Assigner()

    # Bootstrap from first 30 frames
    bootstrap_players = {}
    for frame_idx in range(min(30, len(video_frames))):
        for track_id, player in tracks["players"][frame_idx].items():
            if track_id not in bootstrap_players:
                bootstrap_players[track_id] = player

    assigner.init_teams(video_frames[0], bootstrap_players)

    def resolve_color(team: int) -> tuple:
        return config.colors.get(team, config.colors.get(0, (128, 128, 128)))

    # --- Players ---
    for frame_num, player_track in enumerate(tracks["players"]):
        frame = video_frames[frame_num]
        for track_id, track in player_track.items():
            pid = assigner.get_or_create_id(
                frame, track["bounding_box"], track_id,
            )
            if pid is None:
                continue
            team = assigner.get_team(pid)
            track["global_id"]  = pid
            track["team"]       = team
            track["team_color"] = resolve_color(team)

    # --- Goalkeepers ---
    for frame_num, gk_track in enumerate(tracks["goalkeepers"]):
        frame = video_frames[frame_num]
        for track_id, gk in gk_track.items():
            pid = assigner.get_or_create_id(
                frame, gk["bounding_box"], track_id,
            )
            if pid is None:
                continue
            team = assigner.get_team(pid)
            gk["global_id"]  = pid
            gk["team"]       = team
            gk["team_color"] = resolve_color(team)

    # ------------------------------------------------------------------
    # 5. Annotate and save
    # ------------------------------------------------------------------
    output_frames = tracker.draw_annotations(video_frames, tracks)
    utils.save_video(output_frames, config.output_video_path)
    print(f"Saved: {config.output_video_path}")