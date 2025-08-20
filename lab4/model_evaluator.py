import torch
import time
from sklearn.metrics import f1_score

from model_trainer import ModelTrainer


class ModelEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate_model(self, model, train_loader, val_loader, test_loader):
        start_time = time.time()
        trainer = ModelTrainer(model, train_loader, val_loader, test_loader, self.config)
        trainer.train()

        training_time = time.time() - start_time
        print(f"Training Time: {training_time:.2f} seconds")

        test_loss, test_accuracy = trainer.run_epoch(test_loader, mode="test")
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.config["DEVICE"]), labels.to(self.config["DEVICE"])
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_predictions, average='weighted')
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, F1 Score: {f1:.4f}")
        return test_loss, test_accuracy, f1, training_time

    def compare_models(self, cnn_model, transfer_model, train_loader, val_loader, test_loader):
        print("Evaluating CNN Model...")
        cnn_test_loss, cnn_test_accuracy, cnn_f1, cnn_training_time = self.evaluate_model(cnn_model, train_loader,
                                                                                          val_loader, test_loader)

        print("\nEvaluating Transfer Learning Model...")
        transfer_test_loss, transfer_test_accuracy, transfer_f1, transfer_training_time = self.evaluate_model(
            transfer_model, train_loader, val_loader, test_loader)

        print("\nComparison of Models:")
        print(
            f"CNN Model Test Accuracy: {cnn_test_accuracy:.2f}% | F1 Score: {cnn_f1:.4f} | Training Time: {cnn_training_time:.2f} seconds")
        print(
            f"Transfer Learning Model Test Accuracy: {transfer_test_accuracy:.2f}% | F1 Score: {transfer_f1:.4f} | Training Time: {transfer_training_time:.2f} seconds")

        return {
            "cnn": {"accuracy": cnn_test_accuracy, "f1_score": cnn_f1, "training_time": cnn_training_time},
            "transfer_learning": {"accuracy": transfer_test_accuracy, "f1_score": transfer_f1,
                                  "training_time": transfer_training_time}
        }
