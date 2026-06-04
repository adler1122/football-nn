from config import Config
import utils

from tracking import Tracker, FieldDetector
from tracking.homography import HomographyMapper
from assigner import Assigner
import numpy as np
import cv2


H_RECOMPUTE_EVERY = 90   # roughly every 3 seconds at 30fps


def _build_composite_H(field_detector, frames, mapper, sample_every=10):
    """
    Scan frames to build a composite best-confidence keypoint set
    and compute the best possible H from it.
    Returns (H, best_pts, best_conf).
    """
    N_KPT     = 32
    best_pts  = np.zeros((N_KPT, 2), dtype=np.float32)
    best_conf = np.zeros(N_KPT,      dtype=np.float32)

    for i in range(0, len(frames), sample_every):
        result = field_detector.detect_keypoints_with_confidence(frames[i])
        if result is None:
            continue
        pts, conf = result
        for idx in range(min(N_KPT, len(pts))):
            if conf[idx] > best_conf[idx]:
                best_conf[idx] = conf[idx]
                best_pts[idx]  = pts[idx]

        n_good = int((best_conf >= 0.5).sum())
        print(f"  [scan] frame {i:04d} — {n_good}/32 keypoints >= 0.5 conf")

        if n_good == N_KPT:
            print("  [scan] All 32 found — stopping early.")
            break

    H = mapper.compute(best_pts, confidences=best_conf)
    return H, best_pts, best_conf


def _recompute_H(field_detector, frame, mapper):
    """
    Recompute H from a single frame. Returns new H or None.
    """
    result = field_detector.detect_keypoints_with_confidence(frame)
    if result is None:
        return None
    pts, conf = result
    return mapper.compute(pts, confidences=conf)


def _interpolate_H(H_prev, H_next, alpha):
    """
    Linearly interpolate between two homography matrices.
    alpha=0 → H_prev, alpha=1 → H_next.
    """
    return (1.0 - alpha) * H_prev + alpha * H_next


def run_analyzer(args, config: Config) -> None:
    """
    Full analysis pipeline:
      1. Read video frames.
      2. Scan full video to build best composite H.
      3. Track players, goalkeepers, ball, and others.
      4. Assign teams via jersey colour.
      5. Annotate frames — H is recomputed every H_RECOMPUTE_EVERY frames
         and smoothly interpolated between recomputations so minimap
         positions never jump.
      6. Save output video.
    """


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

    print("Scanning video for best keypoint detections...")
    H_base, _, _ = _build_composite_H(
        field_detector, video_frames, tracker.mapper
    )

    tracker.mapper.H = H_base

    if H_base is not None:
        print("Base homography ready.")
    else:
        print("WARNING: Homography failed — bird's-eye view disabled.")


    # compute H at fixed keyframes across the video, then during
    # annotation we interpolate smoothly between consecutive keyframe H's.
    # This gives smooth minimap movement within each interval and
    # graceful position correction at each keyframe.
    print("Pre-computing homography keyframes...")

    keyframe_indices = list(range(0, len(video_frames), H_RECOMPUTE_EVERY))
    keyframe_H = {}

    for kf_idx in keyframe_indices:
        H_new = _recompute_H(
            field_detector,
            video_frames[kf_idx],
            tracker.mapper,
        )
        # If recomputation fails, keep the previous valid H
        if H_new is not None:
            keyframe_H[kf_idx] = H_new
            print(f"  keyframe {kf_idx:04d}: H recomputed.")
        else:
            # find the most recent valid H
            prev_valid = [k for k in keyframe_H if k < kf_idx]
            if prev_valid:
                keyframe_H[kf_idx] = keyframe_H[max(prev_valid)]
                print(f"  keyframe {kf_idx:04d}: H recomputation failed — keeping previous.")
            elif H_base is not None:
                keyframe_H[kf_idx] = H_base
            else:
                print(f"  keyframe {kf_idx:04d}: no valid H available.")

    # set initial H
    if keyframe_H:
        tracker.mapper.H = keyframe_H[keyframe_indices[0]]


    tracks = tracker.track_detections(video_frames)


    # team assignment

    assigner = Assigner()

    bootstrap_players = {}
    for frame_idx in range(min(30, len(video_frames))):
        for track_id, player in tracks["players"][frame_idx].items():
            if track_id not in bootstrap_players:
                bootstrap_players[track_id] = player

    assigner.init_teams(video_frames[0], bootstrap_players)

    def resolve_color(team: int) -> tuple:
        return config.colors.get(team, config.colors.get(0, (128, 128, 128)))

    for frame_num, player_track in enumerate(tracks["players"]):
        frame = video_frames[frame_num]
        for track_id, track in player_track.items():
            pid = assigner.get_or_create_id(frame, track["bounding_box"], track_id)
            if pid is None:
                continue
            team = assigner.get_team(pid)
            track["global_id"]  = pid
            track["team"]       = team
            track["team_color"] = resolve_color(team)

    for frame_num, gk_track in enumerate(tracks["goalkeepers"]):
        frame = video_frames[frame_num]
        for track_id, gk in gk_track.items():
            pid = assigner.get_or_create_id(frame, gk["bounding_box"], track_id)
            if pid is None:
                continue
            team = assigner.get_team(pid)
            gk["global_id"]  = pid
            gk["team"]       = team
            gk["team_color"] = resolve_color(team)


    output_frames = []
    n = len(video_frames)
    gk_tracks = tracks.get("goalkeepers", [{} for _ in range(n)])

    sorted_keyframes = sorted(keyframe_H.keys())

    for frame_num, frame in enumerate(video_frames):
        frame = frame.copy()

        if keyframe_H:
            prev_kf = max((k for k in sorted_keyframes if k <= frame_num), default=sorted_keyframes[0])
            next_kf_candidates = [k for k in sorted_keyframes if k > frame_num]

            if next_kf_candidates:
                next_kf = next_kf_candidates[0]
                interval = next_kf - prev_kf
                alpha = (frame_num - prev_kf) / interval if interval > 0 else 0.0
                tracker.mapper.H = _interpolate_H(
                    keyframe_H[prev_kf],
                    keyframe_H[next_kf],
                    alpha,
                )
            else:
                
                tracker.mapper.H = keyframe_H[prev_kf]


        if frame_num > 0:
            tracker.mapper.update_with_optical_flow(
                video_frames[frame_num - 1],
                video_frames[frame_num],
            )

        player_dict     = tracks["players"][frame_num]
        ball_dict       = tracks["ball"][frame_num]
        referee_dict    = tracks["others"][frame_num]
        goalkeeper_dict = gk_tracks[frame_num]

        # players
        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = tracker.draw_ellipse(frame, player["bounding_box"], color, track_id)
            if player.get("has_ball", False):
                frame = tracker.draw_triangle(frame, player["bounding_box"], (0, 0, 255))

        # goalkeepers
        for track_id, gk in goalkeeper_dict.items():
            color = gk.get("team_color", (180, 0, 180))
            frame = tracker.draw_ellipse(frame, gk["bounding_box"], color, track_id)

        # referees
        for _, referee in referee_dict.items():
            frame = tracker.draw_ellipse(frame, referee["bounding_box"], (0, 255, 255))

        # ball
        for _, ball in ball_dict.items():
            frame = tracker.draw_triangle(frame, ball["bounding_box"], (0, 255, 0))

        # minimap
        frame = tracker.draw_birds_eye_view(
            frame, player_dict, ball_dict, goalkeeper_dict
        )

        output_frames.append(frame)


    utils.save_video(output_frames, config.output_video_path)
    print(f"Saved: {config.output_video_path}")