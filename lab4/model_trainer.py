import torch
from torch import optim, nn

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, config, writer=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["LEARNING_RATE"])
        self.best_model_state = None
        self.writer = writer

    def run_epoch(self, loader, mode="train", epoch=None):
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        epoch_loss = 0.0
        correct, total = 0, 0
        with torch.set_grad_enabled(mode == "train"):
            for images, labels in loader:
                images, labels = images.to(self.config["DEVICE"]), labels.to(self.config["DEVICE"])
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                epoch_loss += loss.item()
                if mode == "train":
                    loss.backward()
                    self.optimizer.step()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        if self.writer and epoch is not None:
            self.writer.add_scalar(f"{mode.capitalize()} Loss", epoch_loss, epoch)
            self.writer.add_scalar(f"{mode.capitalize()} Accuracy", accuracy, epoch)

        return epoch_loss, accuracy

    def train(self):
        best_accuracy = 0.0

        for epoch in range(self.config["EPOCHS"]):
            train_loss, train_accuracy = self.run_epoch(self.train_loader, mode="train", epoch=epoch)
            val_loss, val_accuracy = self.run_epoch(self.val_loader, mode="val", epoch=epoch)

            print(f"Epoch {epoch + 1}/{self.config['EPOCHS']} - "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.best_model_state = self.model.state_dict()

    def test(self):
        test_loss, test_accuracy = self.run_epoch(self.test_loader, mode="test")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")