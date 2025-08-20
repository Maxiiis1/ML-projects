import matplotlib.pyplot as plt
from myKmeans import MyKMeans


class ClustersVisualizer:
    def __init__(self, model, data):

        self.model = model
        self.data = data
        self.model_type = None

    def plot_clusters(self, feature_x=0, feature_y=1):
        self.model.fit(self.data)
        labels = self.model.predict(self.data)

        if isinstance(self.model, MyKMeans):
            self.model_type = "MyKMeans realization"
        else:
            self.model_type = "KMeans sklearn"


        if hasattr(self.model, 'centroids'):
            centroids = self.model.centroids
        elif hasattr(self.model, 'cluster_centers_'):
            centroids = self.model.cluster_centers_
        else:
            centroids = None

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.data[:, feature_x], self.data[:, feature_y], c=labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Кластеры')
        plt.xlabel(f'Признак {feature_x + 1}')
        plt.ylabel(f'Признак {feature_y + 1}')
        plt.title('Визуализация кластеров по модели: ' + self.model_type)

        if centroids is not None:
            plt.scatter(centroids[:, feature_x], centroids[:, feature_y], marker='X', color='red', s=200,
                        label='Центры кластеров')
            plt.legend()

        plt.show()
