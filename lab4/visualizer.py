import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class EmbeddingsVisualizer:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.config = config

    def generate_embeddings(self):
        self.model.eval()
        embeddings, labels = [], []
        with torch.no_grad():
            for images, label_batch in self.dataloader:
                images = images.to(self.config["DEVICE"])
                features = self.model.get_features(images)
                features = features.view(features.size(0), -1).cpu().numpy()
                embeddings.append(features)
                labels.extend(label_batch.numpy())
        return np.vstack(embeddings), np.array(labels)

    def visualize_embeddings(self):
        embeddings, labels = self.generate_embeddings()
        tsne = TSNE(n_components=2, random_state=42)
        new_embeddings = tsne.fit_transform(embeddings)
        plt.figure(figsize=(10, 8))
        for label in np.unique(labels):
            idx = labels == label
            plt.scatter(new_embeddings[idx, 0], new_embeddings[idx, 1], label=f"Class {label}")
        plt.legend()
        plt.title("t-SNE Visualization of Embeddings")
        plt.show()