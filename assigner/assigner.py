import numpy as np
from sklearn.cluster import KMeans


class Assigner:

    def __init__(self):

       
        self.model = KMeans(
            n_clusters=2,
            n_init=20,
            random_state=42
        )

        self.fitted = False

        
        self.next_id = 0

        
        self.memory = {}

        self.team1_ids = set()
        self.team2_ids = set()

    
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

        
        for pid, label in zip(ids, labels):

            if label == 0:
                self.team1_ids.add(pid)
            else:
                self.team2_ids.add(pid)

    
    def get_or_create_id(self, frame, bbox, track_id):

        feature = self.__get_feature(frame, bbox)

        if not self.__valid(feature):
            return None

        if track_id not in self.memory:

            self.memory[track_id] = feature

            self.__assign_new_player(track_id)

        else:

            self.memory[track_id] = (
                0.9 * self.memory[track_id]
                + 0.1 * feature
            )

        return track_id
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

    
    def get_team(self, player_id):

        if player_id in self.team1_ids:
            return 1

        if player_id in self.team2_ids:
            return 2

        
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