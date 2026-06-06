# Football Video Analytics System

A computer vision pipeline that processes broadcast football match videos to detect and track players, assign team membership, and generate a real-time bird's-eye tactical view of the pitch.

Built for the Applied Neural Network course — KNTU, Spring 2026.  
Instructor: Maryam Abdolali

---

## What the system does

Given a raw football broadcast video, the system outputs an annotated video that shows:

- Every player, goalkeeper, and referee tracked with a stable ID across all frames
- Each player color-coded by team (determined automatically from jersey color)
- The ball tracked and highlighted
- A real-time bird's-eye minimap in the corner showing all player and ball positions on a top-down pitch diagram

---

## How it works — the full pipeline

### Stage 1 — Read the video
The video is loaded frame by frame into memory using OpenCV. All subsequent stages operate on this list of frames.

### Stage 2 — Field keypoint detection
A YOLO pose model trained on football pitch images scans the video to detect up to 32 specific landmark points on the pitch — corners, penalty box corners, penalty spots, centre circle intersections, and so on. These are the reference points used to build the coordinate transformation.

To get the best possible set of keypoints, the system scans one frame every 10 frames across the entire video and keeps the highest-confidence detection seen for each of the 32 indices. This composite set is more reliable than any single frame.

### Stage 3 — Homography computation
A homography is a mathematical matrix (3×3) that maps any pixel coordinate in the camera image to its corresponding location on a standard top-down pitch canvas (1050×680 pixels, matching FIFA's 105m×68m pitch at 10px per metre).

The system computes this matrix from the detected keypoints using `cv2.findHomography` with RANSAC (which automatically rejects noisy or incorrect keypoints). If RANSAC finds too few inliers, it falls back to a least-squares fit on all valid keypoints.

The homography is recomputed every 90 frames from fresh keypoint detections and smoothly interpolated between recomputations — so the minimap updates gradually rather than jumping. Between recomputations, optical flow tracks pitch surface features (grass texture, line edges) to compensate for any camera pan, tilt, or zoom.

### Stage 4 — Player detection and tracking
A second YOLO model detects four object classes in every frame:
- `player` — outfield players
- `goalkeeper` — goalkeepers (tracked separately so team assignment handles their different kit)
- `referee` — referees and coaches
- `ball` — the ball

ByteTrack assigns stable IDs to players across frames. Once a player gets an ID, it keeps that ID even through brief occlusions or when players cross paths.

The ball is handled separately — it is not passed through ByteTrack because it moves too fast and is too small for reliable tracking. It is always stored with ID 1.

### Stage 5 — Team assignment
For each detected player, the system crops the upper-centre portion of the bounding box (the jersey torso area, avoiding arms and background). This crop is converted to HSV color space, background pixels are filtered out, and the mean HSV color is computed as a feature vector.

KMeans clustering (k=2) groups all players into two teams based on jersey color. The clustering is initialized from players visible in the first 30 frames, then applied to every new player that appears.

Every 10 frames, each player's team assignment is re-verified against the cluster model using their current (smoothed) color feature. This corrects mistakes caused by ByteTrack occasionally reassigning an old ID to a different physical player.

### Stage 6 — Visualization and output
For each frame the system draws:
- An ellipse at the feet of each player, colored by team
- A numbered label showing the track ID
- A triangle above the ball
- A cyan ellipse around referees

The bird's-eye minimap is rendered by:
1. Creating a pitch canvas with all standard FIFA markings (penalty boxes, goal areas, centre circle, penalty arcs, etc.)
2. Transforming each player's foot position through the current homography matrix to get their pitch canvas coordinate
3. Drawing a colored dot at that location
4. Compositing the minimap onto the bottom-right corner of the frame with slight transparency

---

## Project structure

```
football_analytics/
├── main.py                  # Entry point — CLI for train/analyze commands
├── config.py                # All settings: paths, colors, device
├── requirements.txt
│
├── models/
│   ├── player_best.pt       # Trained YOLO detection model
│   └── field_best.pt        # Trained YOLO pose model for pitch keypoints
│
├── tracking/
│   ├── __init__.py
│   ├── tracker.py           # Detection, ByteTrack, drawing
│   ├── FieldDetector.py     # Pitch keypoint detection
│   ├── homography.py        # Homography computation + optical flow
│   └── pitch.py             # Pitch canvas drawing
│
├── assigner/
│   ├── __init__.py
│   └── assigner.py          # KMeans jersey color team assignment
│
├── utils/
│   ├── __init__.py
│   └── video_utils.py       # read_video, save_video
│
├── v1test/
│   ├── __init__.py
│   ├── analyze.py           # Full pipeline orchestration
|   ├── train_firld.py        
│   └── train_player.py
│
│
└── tests/
    ├── test_geometry.py     # Bounding box helper tests
    ├── test_homography.py   # Homography mapper tests
    └── test_assigner.py     # Team assigner tests
```

---

## Models

The system uses two custom-trained YOLO models:

| Model | Architecture | Task | Classes |
|-------|-------------|------|---------|
| `player_best.pt` | YOLOv8s | Object detection | player, goalkeeper, referee, ball |
| `field_best.pt` | YOLOv8s-pose | Keypoint detection | pitch (32 keypoints) |

Both models were trained on datasets from Roboflow Universe:
- Player model: [football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- Field model: [football-field-detection](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)

---

## Installation

```bash
pip install -r requirements.txt
```

Place trained model weights in the `models/` directory:
```
models/player_best.pt
models/field_best.pt
```

---

## Usage

**Analyze a video:**
```bash
python main.py analyze
```

**Train the player detection model:**
```bash
python main.py train_player --data datasets/player/data.yaml --epochs 100
```

**Train the field keypoint model:**
```bash
python main.py train_field --data datasets/field/data.yaml --epochs 100
```

Input/output paths and all other settings are configured in `config.py`.

---

## Running the tests

```bash
pytest tests/
```

The tests cover:
- Bounding box geometry helpers (center, width, foot position, distance)
- Homography mapper (transform, compute, pitch coordinate sanity checks)
- Team assigner (team splitting, ID consistency, no player in both teams)
- Video utilities (read/save)

No GPU or model files are required to run the tests.

---

## Configuration

All settings live in `config.py`:

```python
input_video_path  = "./test.mp4"
output_video_path = "./output_video.avi"
device            = "mps"  # or "cuda" or "cpu"

colors = {
    0: (0, 215, 255),   # yellow  — unknown / referee
    1: (50, 205,  50),  # green   — team 1
    2: (60,  20, 220),  # red     — team 2
}
```

---

## Known limitations

- **Partial pitch coverage** — the field model reliably detects ~24 of 32 keypoints on this broadcast angle. Keypoints outside the camera frame cannot be detected. Players near the far edges of the visible pitch may be mapped to slightly incorrect positions on the minimap.
- **Limited training data** — both models were trained on relatively small datasets. Detection quality (especially for the ball and distant players) will improve with more training data and larger model variants.
- **Static camera assumption** — optical flow compensation handles small camera movements well, but large rapid pans or cuts to a completely different camera angle will cause a temporary loss of accurate homography until the next keyframe recomputation.
- **Team 0 (yellow)** — players whose jersey color doesn't clearly belong to either team cluster (referees misclassified by the detector, coaches near the touchline) are assigned to team 0 and shown in yellow.

---

## Libraries used

| Library | Purpose |
|---------|---------|
| `ultralytics` | YOLO model inference for detection and keypoint detection |
| `supervision` | Detection format conversion, ByteTrack multi-object tracking |
| `opencv-python` | Video I/O, homography, perspective transform, drawing, optical flow |
| `numpy` | Array operations, coordinate handling |
| `scikit-learn` | KMeans clustering for team assignment |
| `torch` | Backend for YOLO model inference |
| `pytest` | Unit testing |

---

## Pipeline diagram

```
Video frame
├── Field model (YOLOv8s-pose)
│     ↓
│   32 field keypoints (x, y, confidence)
│     ↓
│   Confidence filter (≥ 0.5)
│     ↓
│   findHomography + RANSAC
│     ↓
│   Homography matrix H  ←── updated every frame via optical flow
│                         ←── recomputed every 90 frames from keypoints
│                              (smoothly interpolated between recomputations)
│
└── Player model (YOLOv8s)
      ↓
    Raw detections (player · goalkeeper · referee · ball)
      ↓
    ByteTrack → stable track IDs
      ↓
    Team assigner (KMeans on HSV jersey color)
      ↓
    Enriched tracks (bbox · team · team_color)
      ↓
    perspectiveTransform (foot position → pitch canvas)
      ↓
    Annotate frames + bird's-eye minimap overlay
      ↓
    Output video
```
