import matplotlib.pyplot as plt

class ModelVisualizer:
    def __init__(self, train_losses, val_losses, train_accuracies, val_accuracies, epochs, test_loss=None, test_accuracy=None):
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_accuracies = train_accuracies
        self.val_accuracies = val_accuracies
        self.epochs = epochs
        self.test_loss = test_loss
        self.test_accuracy = test_accuracy

    def plot_metrics(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epochs + 1), self.train_losses, label='Train Loss')
        plt.plot(range(1, self.epochs + 1), self.val_losses, label='Validation Loss')
        plt.axhline(y=self.test_loss, color='r', linestyle='--', label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epochs + 1), self.train_accuracies, label='Train Accuracy')
        plt.plot(range(1, self.epochs + 1), self.val_accuracies, label='Validation Accuracy')
        plt.axhline(y=self.test_accuracy, color='r', linestyle='--', label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()
        plt.show()
