"""
This script is designed for training and testing a ResNet 50 model to classify dog breeds.

Command-Line Options:
    --train : Train the model using the training and validation datasets. This will also save the 
            trained model to a specified path.

    --test  : Test the model using the test dataset. This option requires that a trained model is
            available and specified in the script. (not yet implemented)

    --local : Create switch to replace folder paths. This is used as a .env alternative.

Examples:
    To train and then test the model:
    python your_script_name.py --train --test

    To add local switch
    python your_script_name.py --train --local

Note: Ensure that the dataset paths, model save path, and any other configurations are correctly
set within the script before running it.
"""

import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, ConfusionMatrix
import torchvision
from pytorch_lightning import Trainer
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import wandb

# Constants
DATASET_SEGMENTS = ["Train", "Test", "Validation"]
BATCH_SIZE = 32
NUM_WORKERS = 4

CUSTOM_TRANSFORM = {
        "Train": transforms.Compose([
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "Test": transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "Validation": transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

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
        self.base_model.fc = torch.nn.Linear(in_features, num_classes)

        # Torch metrics accuracy
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        # Confusion Matrix for Test
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        # Forward pass through the modified ResNet-50
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

class DogBreedDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size=32):
        """
        Initialize Dog Module to aggregate data.

        Args:
            dataset_path (str): The path to the dataset.
            batch_size (int): Size of batch.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        """
        Initialize datasets for training, validation, and testing
        """
        self.train_dataset = DogDataset(self.dataset_path, "Train", CUSTOM_TRANSFORM["Train"])
        self.val_dataset = DogDataset(self.dataset_path, "Validation", CUSTOM_TRANSFORM["Validation"])
        self.test_dataset = DogDataset(self.dataset_path, "Test", CUSTOM_TRANSFORM["Test"])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

def get_paths(is_local:bool=True) -> list[str, str]:
    """
    Determines the dataset and model save paths based on the execution environment.
    """
    if is_local:
        dataset_path = "D:/chris/Documents/UofC/MEng Soft/winter/ENEL 645/ENEL 645/ENEL 645/project/small_dataset/"
        save_model_path = "D:/chris/Documents/UofC/MEng Soft/winter/ENEL 645/ENEL 645/ENEL 645/project/best_model/"
    else:
        dataset_path = "/work/TALC/enel645_2024w/group24/dataset-143-classes/"
        save_model_path = "/home/christian.valdez/ENSF-611-ENEL-645/ENEL 645/project/best_model/"
    return dataset_path, save_model_path

def train_dog_breed_classifier(dataset_path: str, save_model_path: str, project_name: str, max_epochs: int = 10, batch_size: int = 32, use_gpu: bool = True):
    """
    Trains the dog breed classifier model with Weights & Biases logging and device configuration.

    Args:
        dataset_path (str): The path to the dataset.
        save_model_path (str): The directory path where the model checkpoints will be saved.
        project_name (str): The Weights & Biases project name.
        max_epochs (int, optional): The maximum number of epochs for training. Defaults to 10.
        batch_size (int, optional): The batch size for training and validation. Defaults to 32.
        use_gpu (bool, optional): Whether to use GPU for training if available. Defaults to True.
    """

    # Configure device usage
    if use_gpu and torch.cuda.is_available():
        print(f"✅ Device configured to use CUDA with {device_count} GPU(s).")
    else:
        device_count = 0
        print("✅ Device configured to use CPU.")

    # Initialize Weights & Biases logger
    print(f"📈 Initializing WandB logger.")
    wandb_logger = WandbLogger(project=project_name, log_model="all")

    # Initialize the model and data module
    model = DogBreedClassifier()
    data_module = DogBreedDataModule(dataset_path=dataset_path, batch_size=batch_size)

    # Setup model checkpoints
    model_checkpoint = ModelCheckpoint(
        dirpath=save_model_path,
        filename="best-model-{epoch:02d}-{val_acc:.2f}",
        monitor="val_acc",  # Saving model with highest val accuracy
        save_top_k=1,
        mode="max",
        verbose=True
    )

    # Setup early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min"
    )

    # Initialize the PyTorch Lightning trainer with WandbLogger
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=[model_checkpoint, early_stopping]
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # wandb_logger.experiment.finish()

    return trainer, model

# -------------------------------------------------------------------------------- #
#                                                                                  #
#                               Main Function                                      #
#                                                                                  #
# -------------------------------------------------------------------------------- #

def main(args):
    print("🚀 Starting Main Function...\n")

    # Paths to the dataset (could be in .env file but its okay)
    dataset_path, save_model_path = get_paths(args.local)
    print(f"📂 Dataset and Model paths are set!\nUsing the dataset path from {dataset_path}\nUsing the model path from {save_model_path}")
    print("\n", "=" * 60, "\n")

    if args.train:
        print("🏋️ Starting training process...")
        trainer, model = train_dog_breed_classifier(
            dataset_path=dataset_path,  # Adjust path
            save_model_path=save_model_path,  # Adjust path
            project_name="enel 645 project",  # Set wandb project name
            max_epochs=10,
            batch_size=32,
            use_gpu=True
        )
        
    print("✅ Training completed!")
    print("\n", "=" * 60, "\n")

    if args.test:
        print("🔍 Running tests...")
        wandb.init(project="enel 645 project", job_type="test")
        trainer.test(model, datamodule=DogBreedDataModule(dataset_path=dataset_path, batch_size=32))
        wandb.finish()
    print("✅ Testing completed!")

    print("🎉 Main Function Execution Completed Successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and/or test model.")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--local", action="store_true", help="Check if using local computer")
    args = parser.parse_args()
    main(args)