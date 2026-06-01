import numpy as np
from sklearn.cluster import KMeans


class Assigner:
    """
    Assigns players to one of two teams (and a fallback team 0 for
    referees / unknowns) using KMeans clustering on jersey colour.

    Team IDs
    --------
    0  — unknown / referee / not yet classified
    1  — team 1
    2  — team 2

    Usage
    -----
    assigner = Assigner()
    assigner.init_teams(first_frame, tracks["players"][0])

    for track_id, track in player_track.items():
        pid = assigner.get_or_create_id(frame, track["bounding_box"], track_id)
        team = assigner.get_team(pid)   # 1 or 2  (0 if unknown)
    """

    def __init__(self) -> None:
        self.model = KMeans(
            n_clusters=2,
            n_init=20,
            random_state=42,
        )
        self.fitted: bool = False

        
        self.memory: dict[int, np.ndarray] = {}

        self.team1_ids: set[int] = set()
        self.team2_ids: set[int] = set()



    def init_teams(self, frame: np.ndarray, players: dict) -> None:
        """
        Bootstrap team clusters from the players visible in the first frame.

        Parameters
        ----------
        frame   : BGR image (first video frame).
        players : dict  { track_id → {"bounding_box": [x1,y1,x2,y2], ...} }
                  This must use the *real* ByteTrack IDs, not sequential ints.
        """
        features: list[np.ndarray] = []
        track_ids: list[int] = []

        for track_id, p in players.items():
            f = self._get_feature(frame, p["bounding_box"])
            if self._valid(f):
                features.append(f)
                track_ids.append(track_id)   
                self.memory[track_id] = f

        if len(features) < 2:
            print("[Assigner] init_teams: not enough valid players to cluster.")
            return

        features_arr = np.array(features)
        labels = self.model.fit_predict(features_arr.astype(np.float64))
        self.fitted = True

        for track_id, label in zip(track_ids, labels):
            if label == 0:
                self.team1_ids.add(track_id)
            else:
                self.team2_ids.add(track_id)

        print(
            f"[Assigner] init_teams: team1={len(self.team1_ids)} "
            f"team2={len(self.team2_ids)} players"
        )



    def get_or_create_id(
        self,
        frame: np.ndarray,
        bbox: list | np.ndarray,
        track_id: int,
    ) -> int | None:
        """
        Update the colour memory for *track_id* and assign to a team if
        not yet done.

        Returns the track_id on success, or None if the crop is invalid.
        """
        feature = self._get_feature(frame, bbox)

        if not self._valid(feature):
            return None

        if track_id not in self.memory:
            
            self.memory[track_id] = feature
            self._assign_new_player(track_id)
        else:
            
            self.memory[track_id] = (
                0.9 * self.memory[track_id] + 0.1 * feature
            )

        return track_id


    def get_team(self, player_id: int) -> int:
        """
        Return the team number for *player_id*.

        Returns
        -------
        1  — team 1
        2  — team 2
        0  — unknown (referee, coach, or not yet assigned)
        """
        if player_id in self.team1_ids:
            return 1
        if player_id in self.team2_ids:
            return 2
        return 0   # safe fallback — never raises KeyError in config.colors



    def _assign_new_player(self, pid: int) -> None:
        """Classify a previously unseen player into team 1 or 2."""
        feature = self.memory[pid]

        if not self.fitted:
            
            self.team1_ids.add(pid)
            return

        label = int(self.model.predict([feature.astype(np.float64)])[0])

        if label == 0:
            self.team1_ids.add(pid)
        else:
            self.team2_ids.add(pid)

    def _get_feature(self, frame: np.ndarray, bbox: list | np.ndarray) -> np.ndarray:

        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros(3)

        h, w = crop.shape[:2]

        region = crop[
            int(h * 0.20): int(h * 0.55),
            int(w * 0.25): int(w * 0.75),
        ]

        if region.size == 0:
            return np.zeros(3)

        pixels = region.reshape(-1, 3).astype(np.float32)

        
        mean_brightness = pixels.mean(axis=1)
        pixels = pixels[(mean_brightness > 40) & (mean_brightness < 220)]

        if len(pixels) < 10:
            return np.zeros(3)

        return np.mean(pixels, axis=0)   # shape (3,)  — BGR

    def _valid(self, f: np.ndarray) -> bool:
        """Return True when the feature vector is non-trivial."""
        return bool(np.linalg.norm(f) > 10)

    
    __get_feature = _get_feature
    __valid = _valid