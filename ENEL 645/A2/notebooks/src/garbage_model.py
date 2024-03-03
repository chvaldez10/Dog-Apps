import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import Accuracy
from torchvision import transforms, models

class GarbageModel(pl.LightningModule):
    def __init__(self, input_shape: tuple, num_classes: int, learning_rate: float = 2e-4, transfer: bool = False):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes
        
        # transfer learning if pretrained=True
        self.feature_extractor = models.resnet18(pretrained=transfer)

        if transfer:
            # layers are frozen by using eval()
            self.feature_extractor.eval()
            # freeze params
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

            self.classifier = nn.Linear(self.feature_extractor.fc.in_features, num_classes)

            self.criterion = nn.CrossEntropyLoss()
            self.accuracy = Accuracy()
    
    # will be used during inference
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x