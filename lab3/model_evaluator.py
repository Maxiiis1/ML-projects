from sklearn.metrics import classification_report
import torch

class ModelEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def evaluate_model(self, data_loader, mode="test"):
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for texts, labels in data_loader:
                texts = texts.to(self.config["DEVICE"])
                outputs = self.model(texts)
                predictions = outputs.argmax(dim=1).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.numpy())

        report = classification_report(all_labels, all_predictions, zero_division=0, output_dict=True)
        return report
