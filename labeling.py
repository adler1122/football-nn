"""
Keypoint Dump Tool
==================
Saves one annotated image per sampled frame so you can go through
them manually and write down what each index points to on the pitch.

Usage
-----
    python dump_keypoints.py

Output
------
    keypoint_frames/frame_0000.jpg
    keypoint_frames/frame_0005.jpg
    ...  (one image every SAMPLE_EVERY frames)

Each image has:
  - Green dot  on every detected keypoint
  - Red index number next to each dot
  - Confidence score in small text below the number
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────
VIDEO_PATH    = "./test.mp4"
MODEL_PATH    = "./models/field_best.pt"
OUTPUT_DIR    = "./keypoint_frames"
SAMPLE_EVERY  = 30       # save one frame every N frames
MAX_FRAMES    = 20       # max number of output images
CONF          = 0.25
# ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    model = YOLO(MODEL_PATH)

    print("Opening video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: cannot open video.")
        return

    saved       = 0
    frame_idx   = 0

    while saved < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_EVERY != 0:
            frame_idx += 1
            continue

        # Run field model
        result = model.predict(frame, conf=CONF, verbose=False)[0]

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            print(f"  frame {frame_idx:04d} — no keypoints, skipping")
            frame_idx += 1
            continue

        pts  = result.keypoints.xy[0].cpu().numpy()
        conf = result.keypoints.conf
        conf = conf[0].cpu().numpy() if conf is not None else np.ones(len(pts))

        vis = frame.copy()
        detected = 0

        for i, (x, y) in enumerate(pts):
            if x == 0 and y == 0:
                continue

            cx, cy = int(x), int(y)
            detected += 1

            # Dot
            cv2.circle(vis, (cx, cy), 7, (0, 255, 0), -1)
            cv2.circle(vis, (cx, cy), 7, (255, 255, 255), 2)

            # Index number — large, red
            cv2.putText(
                vis, str(i),
                (cx + 10, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 0, 255), 2,
            )

            # Confidence — small, yellow
            cv2.putText(
                vis, f"{conf[i]:.2f}",
                (cx + 10, cy + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 220, 255), 1,
            )

        # Frame info top-left
        info = f"frame {frame_idx:04d}  |  {detected} keypoints detected"
        cv2.rectangle(vis, (0, 0), (500, 32), (0, 0, 0), -1)
        cv2.putText(vis, info, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        out_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path, vis)
        print(f"  saved: {out_path}  ({detected} keypoints)")

        saved     += 1
        frame_idx += 1

    cap.release()
    print(f"\nDone — {saved} images saved to ./{OUTPUT_DIR}/")
    print("Go through them and note which index lands on which pitch marking.")


if __name__ == "__main__":
    main()