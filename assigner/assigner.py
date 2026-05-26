import numpy as np
from sklearn.cluster import KMeans


class Assigner:

    def __init__(self):

        # team clustering model
        self.model = KMeans(
            n_clusters=2,
            n_init=20,
            random_state=42
        )

        self.fitted = False

        # ID system
        self.next_id = 0

        # ID → feature (fixed)
        self.memory = {}

        # 🔥 FINAL FIX: HARD TEAM LISTS
        self.team1_ids = set()
        self.team2_ids = set()

    # -----------------------------
    # INITIAL TEAM CLUSTERING
    # -----------------------------
    def init_teams(self, frame, players):

        features = []
        ids = []

        for pid, p in players.items():

            f = self.__get_feature(frame, p["bounding_box"])

            if self.__valid(f):
                features.append(f)
                ids.append(self.next_id)
                self.memory[self.next_id] = f
                self.next_id += 1

        features = np.array(features)

        labels = self.model.fit_predict(features)
        self.fitted = True

        # 🔥 LOCK TEAM ASSIGNMENT
        for pid, label in zip(range(len(labels)), labels):

            if label == 0:
                self.team1_ids.add(pid)
            else:
                self.team2_ids.add(pid)

    # -----------------------------
    # GET OR CREATE ID
    # -----------------------------
    def get_or_create_id(self, frame, bbox):

        feature = self.__get_feature(frame, bbox)

        if not self.__valid(feature):
            return None

        best_id = None
        best_dist = float("inf")

        for pid, stored in self.memory.items():

            dist = np.linalg.norm(feature - stored)

            if dist < 30 and dist < best_dist:
                best_dist = dist
                best_id = pid

        if best_id is None:

            best_id = self.next_id
            self.next_id += 1

            self.memory[best_id] = feature

            # 🔥 NEW PLAYER → MUST CLASSIFY ONCE
            self.__assign_new_player(best_id)

        return best_id

    # -----------------------------
    # NEW PLAYER CLASSIFICATION (ONLY ONCE)
    # -----------------------------
    def __assign_new_player(self, pid):

        feature = self.memory[pid]

        if not self.fitted:
            self.team1_ids.add(pid)
            return

        label = int(self.model.predict([feature])[0])

        if label == 0:
            self.team1_ids.add(pid)
        else:
            self.team2_ids.add(pid)

    # -----------------------------
    # GET TEAM (PURE LOOKUP NOW)
    # -----------------------------
    def get_team(self, player_id):

        if player_id in self.team1_ids:
            return 1

        if player_id in self.team2_ids:
            return 2

        return 1

    # -----------------------------
    # FEATURE EXTRACTION
    # -----------------------------
    def __get_feature(self, frame, bbox):

        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros(3)

        h, w = crop.shape[:2]

        region = crop[
            int(h * 0.2): int(h * 0.55),
            int(w * 0.25): int(w * 0.75)
        ]

        pixels = region.reshape(-1, 3)

        mean = pixels.mean(axis=1)
        pixels = pixels[(mean > 40) & (mean < 220)]

        if len(pixels) < 10:
            return np.zeros(3)

        return np.mean(pixels, axis=0)

    def __valid(self, f):
        return np.linalg.norm(f) > 10