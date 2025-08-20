import os
import requests
from torchvision import transforms

class DatasetPreparation:
    def __init__(self, config):
        self.data_path = config["DATA_PATH"]
        self.cat_api = config["CAT_API_URL"]
        self.dog_api = config["DOG_API_URL"]
        self.num_images = config["NUM_IMAGES"]
        self.image_size = config["IMAGE_SIZE"]

    def fetch_images(self, api_url, folder, label):
        os.makedirs(folder, exist_ok=True)
        for i in range(self.num_images):
            response = requests.get(api_url)
            if response.status_code == 200:
                image_url = response.json()[0]["url"]
                image_data = requests.get(image_url).content
                file_path = os.path.join(folder, f"{label}_{i}.jpg")
                with open(file_path, "wb") as file:
                    file.write(image_data)

    def download_dataset(self):
        print("Downloading cat images")
        self.fetch_images(self.cat_api, os.path.join(self.data_path, "cats"), "cat")
        print("Downloading dog images")
        self.fetch_images(self.dog_api, os.path.join(self.data_path, "dogs"), "dog")

    def preprocess_dataset(self):
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform
