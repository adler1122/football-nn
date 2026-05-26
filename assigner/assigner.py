import numpy as np
from sklearn.cluster import KMeans


class Assigner:

    def __init__(self):

        self.model = KMeans(
            n_clusters=2,
            init="k-means++",
            n_init=20,
            random_state=42
        )

        self.is_fitted = False
        self.teams = {}

    # -----------------------------
    # TRAIN GLOBAL MODEL (FIXED)
    # -----------------------------
    def assign_team(self, frames, tracks):

        samples = []

        # collect from multiple frames (important fix)
        for frame_num in range(min(50, len(frames))):

            frame_players = tracks["players"][frame_num]

            for _, player in frame_players.items():

                color = self.__get_color(frames[frame_num], player["bounding_box"])

                if self.__valid(color):
                    samples.append(color)

        samples = np.array(samples)

        if len(samples) < 2:
            raise ValueError("Not enough valid samples for KMeans")

        self.model.fit(samples)
        self.is_fitted = True

    # -----------------------------
    # PREDICT TEAM
    # -----------------------------
    def get_player_team(self, frame, bbox, player_id):

        if not self.is_fitted:
            return 1

        if player_id in self.teams:
            return self.teams[player_id]

        color = self.__get_color(frame, bbox)

        if not self.__valid(color):
            team = 1
        else:
            team = int(self.model.predict([color])[0]) + 1

        self.teams[player_id] = team
        return team

    # -----------------------------
    # CLEAN JERSEY COLOR EXTRACTION (CRITICAL FIX)
    # -----------------------------
    def __get_color(self, frame, bbox):

        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return np.array([0, 0, 0])

        h, w = crop.shape[:2]

        # strict upper torso region (fixes most of your errors)
        region = crop[
            int(h * 0.15): int(h * 0.55),
            int(w * 0.25): int(w * 0.75)
        ]

        if region.size == 0:
            return np.array([0, 0, 0])

        pixels = region.reshape(-1, 3)

        # remove grass + noise
        mean = pixels.mean(axis=1)
        mask = (mean > 40) & (mean < 220)
        pixels = pixels[mask]

        if len(pixels) == 0:
            return np.array([0, 0, 0])

        return np.median(pixels, axis=0)

    def __valid(self, color):
        return not np.all(color == 0)