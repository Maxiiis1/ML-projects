from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import numpy as np

class ImageDatasetProcessor:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

    def split_dataset(self):
        dataset_size = len(self.dataset)
        indices = np.arange(dataset_size)
        labels = [self.dataset[i][1] for i in indices]

        train_idx, temp_idx, train_labels, temp_labels = train_test_split(
            indices, labels,
            test_size=0.3, stratify=labels, random_state=self.config["SEED"]
        )

        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, temp_labels,
            test_size=0.5, stratify=temp_labels, random_state=self.config["SEED"]
        )

        return train_idx, val_idx, test_idx

    def create_dataloaders(self):
        train_idx, val_idx, test_idx = self.split_dataset()

        train_dataset = Subset(self.dataset, train_idx)
        val_dataset = Subset(self.dataset, val_idx)
        test_dataset = Subset(self.dataset, test_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=False
        )

        return train_loader, val_loader, test_loader
