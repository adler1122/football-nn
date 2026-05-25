from sklearn.cluster import KMeans
from typing import List, Tuple

class Assigner:
    def __init__(self):
        self.first_team_color = []
        self.second_team_color = []
        self.teams = {}
        self.model = KMeans(n_clusters=2, init="k-means++", n_init=10)

    def assign_team(self, frame, players):
        colors = []

        for _ , player in players.items():
            box = player["bounding_box"]
            actual_color = self.__get_color(frame, box)
            colors.append(actual_color)

        self.model = KMeans(n_clusters=2, init="k-means++", n_init=10)
        self.model.fit(colors)

        self.first_team_color = self.model.cluster_centers_[0]
        self.second_team_color = self.model.cluster_centers_[1]

    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.teams:
            return self.teams[player_id]

        player_color = self.__get_color(frame,player_bbox)

        team_id = self.model.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        if player_id == 101:
            team_id=1

        self.teams[player_id] = team_id

        return team_id

    def __get_color(self, frame, bbox):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        crop = frame[y1:y2, x1:x2]

        h = crop.shape[0]
        mid_third = crop[h // 3 : 2 * h // 3, :]

        image_reshaped = mid_third.reshape(-1, 3)

        model = KMeans(n_clusters = 2, init="k-means++", n_init=1)
        model.fit(image_reshaped)
        labels = model.labels_.reshape(mid_third.shape[0], mid_third.shape[1])

        corner_labels = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        bg_cluster = max(set(corner_labels), key=corner_labels.count)
        shirt_cluster = 1 - bg_cluster

        return model.cluster_centers_[shirt_cluster]
