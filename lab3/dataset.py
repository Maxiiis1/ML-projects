import torch
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, max_len):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        padding = [0] * (self.max_len - len(text))
        text = text + padding if len(text) < self.max_len else text[:self.max_len]
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)
