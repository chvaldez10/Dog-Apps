import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import pandas as pd
from sklearn.metrics import classification_report
from PIL import Image

class DogBreedClassifier(pl.LightningModule):
    def __init__(self, num_classes: int=143):  # Expected to be 143
        """
        Initialize the DogBreedClassifier.
        
        Parameters:
        - num_classes (int): Set the number of dog breeds where default is 143
        """
        super().__init__()
        # Load ResNet-50 model
        self.base_model = torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT")

        # Freeze all layers in the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace the classifier layer with a new one for 143 dog breeds
        in_features = self.base_model.fc.in_features  # Get the input feature size of the original classifier
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )

        self.base_model.fc = self.classifier

        # Torch metrics accuracy
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        # Confusion Matrix for Test
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.base_model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        # Calculate and log training accuracy
        predictions = torch.argmax(outputs, dim=1)
        self.train_accuracy.update(predictions, labels)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self):
        self.log("train_acc_epoch", self.train_accuracy.compute(), prog_bar=True)
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        predictions = torch.argmax(outputs, dim=1)
        self.val_accuracy.update(predictions, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        predictions = torch.argmax(outputs, dim=1)
        self.test_accuracy.update(predictions, labels)
        self.test_confusion_matrix.update(predictions, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"test_loss": loss, "predictions": predictions, "labels": labels}

    def on_test_epoch_end(self):
        # Log test accuracy
        self.log("test_acc", self.test_accuracy.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Log confusion matrix
        confusion_matrix = self.test_confusion_matrix.compute().cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(confusion_matrix, annot=True, fmt="g", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.close(fig)

        self.test_accuracy.reset()
        self.test_confusion_matrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class DogDataset(Dataset):
    def __init__(self, root_dir: str, dataset_type: str, transforms=None):
        """
        Initialize the DogDataset.
        
        Parameters:
        - root_dir (str): The base directory of the dataset.
        - dataset_type (str): Type of the dataset. Must be "Train", "Test", or "Validation".
        - transforms: Transformations to be applied on the dataset.
        """
        self.root_dir = os.path.join(root_dir, dataset_type)
        self.transforms = transforms
        self.file_paths = []
        self.labels = []
        self.label_to_index = {} 
        self._generate_sample()

    def _generate_sample(self):
        label_index = 0
        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                
                # Assign an index to each label the first time it's encountered
                if label not in self.label_to_index:
                    self.label_to_index[label] = label_index
                    label_index += 1
                
                for file_name in os.listdir(label_dir):
                    if file_name.endswith(".jpg"):
                        self.file_paths.append(os.path.normpath(os.path.join(label_dir, file_name)))
                        self.labels.append(label)
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]  

        # Convert label string to integer index
        label_index = self.label_to_index[label]

        if self.transforms:
            image = self.transforms(image)

        return image, label_index


def generate_classification_report(checkpoint_path, dataset_path, batch_size=32):
    """
    Loads a trained model checkpoint and generates a classification report on the test dataset.
    """
    # Load the model from checkpoint
    model = DogBreedClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()

    # Prepare the test dataset
    test_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = DogDataset(root_dir=dataset_path, dataset_type="Test", transforms=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Predictions and Labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            images, labels = batch
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=list(test_dataset.label_to_index.keys()), output_dict=True)
    print(pd.DataFrame(report).transpose())

if __name__ == "__main__":
    checkpoint_file = "/home/christian.valdez/ENSF-611-ENEL-645/project/best_model/best-model-epoch=06-val_acc=0.67.ckpt"
    dataset_path = "/work/TALC/enel645_2024w/group24/dataset-143-classes/"
    batch_size = 32
    generate_classification_report(checkpoint_file, dataset_path, batch_size)
