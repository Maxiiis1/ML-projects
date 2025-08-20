import numpy as np


class MyKMeans:
    def __init__(self, n_clusters=3, max_iter=300):
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.centroids = None

    def fit(self, data):
        centroids_indexes = np.random.choice(data.shape[0], self.n_clusters, replace=False)
        self.centroids = data[centroids_indexes]

        cnt = 0
        while cnt < self.max_iter:
            distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids
            cnt += 1

    def predict(self, data):
        new_distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(new_distances, axis=1)

    def get_params(self, deep=True):
        return {'n_clusters': self.n_clusters, 'max_iter': self.max_iter}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
