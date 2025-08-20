from sklearn.model_selection import train_test_split
from tokenizers.implementations import ByteLevelBPETokenizer
import pandas as pd

class TextProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = ByteLevelBPETokenizer()
        self.tokenizer.train(
            config["DATA_PATH"],
            vocab_size=config["VOCAB_SIZE"],
            min_frequency=2,
        )

    def process_texts(self, texts):
        return [
            self.tokenizer.encode(text).ids[:self.config["MAX_SEQ_LEN"]]
            for text in texts
        ]

    def prepare_datasets(self):
        data = pd.read_csv(self.config["DATA_PATH"])
        texts, labels = data["text"], data["sentiment"]

        label_map = {
            "NEGATIVE": 0,
            "NEUTRAL": 1,
            "POSITIVE": 1
        }
        labels = labels.map(label_map).values

        tokenized_texts = self.process_texts(texts)

        train_x, temp_x, train_y, temp_y = train_test_split(
            tokenized_texts, labels, test_size=0.3, stratify=labels, random_state=42
        )

        val_x, test_x, val_y, test_y = train_test_split(
            temp_x, temp_y, test_size=0.5, stratify=temp_y, random_state=42
        )

        return train_x, train_y, val_x, val_y, test_x, test_y