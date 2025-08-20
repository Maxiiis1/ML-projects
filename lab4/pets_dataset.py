import os
from PIL import Image
from torch.utils.data import Dataset

class PetsDataset(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform
        self.images, self.labels = self.load_images()

    def load_images(self):
        images, labels = [], []
        for label, folder in enumerate(["cats", "dogs"]):
            path = os.path.join(self.data_path, folder)
            for file_name in os.listdir(path):
                file_path = os.path.join(path, file_name)
                if file_name.endswith(".jpg"):
                    images.append(file_path)
                    labels.append(label)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
