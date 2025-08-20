import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class TransferLearningModel(nn.Module):
    def __init__(self, config):
        super(TransferLearningModel, self).__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(config["DROPOUT_PROB"]),
            nn.Linear(128, config["NUM_CLASSES"])
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
