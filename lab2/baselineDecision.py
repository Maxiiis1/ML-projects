from sklearn.cluster import KMeans
from clustersVisualizer import ClustersVisualizer


class BaselineDecision:
    def __init__(self, data):
        self.data = data

    def showBaselineDecision(self):
        baseline = KMeans()
        visualizer = ClustersVisualizer(baseline, self.data)
        visualizer.plot_clusters(feature_x=0, feature_y=1)
