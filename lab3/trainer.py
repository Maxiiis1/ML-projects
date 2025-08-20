import torch

class ModelTrainer:
    def __init__(self, train_loader, val_loader, model, loss_fn, optimizer, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(config["DEVICE"])
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config
        self.epochs = config["EPOCHS"]
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def run_epoch(self, loader, mode="eval"):
        self.model.train() if mode == "train" else self.model.eval()
        total_loss, correct, total_samples = 0, 0, 0
        with torch.set_grad_enabled(mode == "train"):
            for batch in loader:
                inputs, targets = batch[0].to(self.config["DEVICE"]), batch[1].to(self.config["DEVICE"])
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                if mode == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total_samples
        return avg_loss, accuracy

    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_loss, train_accuracy = self.run_epoch(self.train_loader, mode="train")
            val_loss, val_accuracy = self.run_epoch(self.val_loader, mode="eval")

            print(f"Epoch {epoch}/{self.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)

    def test(self, test_loader):
        test_loss, test_accuracy = self.run_epoch(test_loader, mode="eval")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        return test_loss, test_accuracy
