import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import Accuracy
from torchvision import transforms, models
import torch

class GarbageModel(pl.LightningModule):
    def __init__(self, input_shape: tuple, num_classes: int, learning_rate: float = 2e-4, transfer: bool = False):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.input_shape = input_shape

        self.num_classes = num_classes
        
        # transfer learning if pretrained=True
        self.feature_extractor = models.resnet18(pretrained=transfer)

        if transfer:
            # layers are frozen by using eval()
            self.feature_extractor.eval()
            # freeze params
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        n_features = self._get_conv_output(self.input_shape)
        self.classifier = nn.Linear(n_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
    
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self.feature_extractor(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # will be used during inference
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x