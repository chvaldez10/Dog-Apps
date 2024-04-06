"""
This script is designed for training and testing a <some> model
for image recognition tasks.

Command-Line Options:
  --train : Train the model using the training and validation datasets. This will also save the 
            trained model to a specified path. If this option is selected, the script will perform
            training operations including model training and validation.

  --test  : Test the model using the test dataset. This option requires that a trained model is
            available and specified in the script. If this option is selected, the script will
            perform testing operations and output the performance metrics of the model on the test dataset.

Both options can be used together to first train the model and then test it without needing to run
the script twice. If no option is specified, the script will not perform any operations.

Examples:
  To train the model:
  python your_script_name.py --train

  To test the model:
  python your_script_name.py --test

  To train and then test the model:
  python your_script_name.py --train --test

Note: Ensure that the dataset paths, model save path, and any other configurations are correctly
set within the script before running it.
"""

import re
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import os
from typing import List, Tuple, Dict
from itertools import product
from PIL import Image
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
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

class DogDataset(Dataset):
    def __init__(self, root_dir: str, dataset_type: str, transforms=None,) -> None:
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
        self._generate_sample()

    def _generate_sample(self):
        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                for file_name in os.listdir(label_dir):
                    if file_name.endswith(".jpg"):
                        self.file_paths.append(os.path.normpath(os.path.join(label_dir, file_name)))
                        self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Convert image to RGB
        label = self.labels[idx]

        if self.transforms:
            image = self.transforms(image)

        return image, label

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience:int=5, verbose:bool=False, delta:float=0, path:str='checkpoint.pt', trace_func=print):
        """
        Initialize EarlyStopping
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def initialize_dataset_and_loader(dataset_path: str, verbose: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Initializes the dataset and data loaders for training, validation, and testing.
    """
    if verbose:
        print("✨ Verbose Mode: ON ✨\n")
        print("🔍 Loading dataset...")
        print("=" * 60)

    # Custom dataset
    train_dataset = DogDataset(dataset_path, "Train", CUSTOM_TRANSFORM["Train"])
    test_dataset = DogDataset(dataset_path, "Test", CUSTOM_TRANSFORM["Test"])
    val_dataset = DogDataset(dataset_path, "Validation", CUSTOM_TRANSFORM["Validation"])

    if verbose:
        print("🐶 Training Dataset Initialized! First Peek:")
        print("   🏷️ Label: ", train_dataset.labels[0])
        print("   📂 File Path: ", train_dataset.file_paths[0])
        print("   " + "-" * 58)

        print("🧪 Testing and Validation Datasets Initialized!")
        print("   🐾 Total Training Samples: ", len(train_dataset))
        print("   🐾 Total Testing Samples: ", len(test_dataset))
        print("   🐾 Total Validation Samples: ", len(val_dataset))
        print("   " + "-" * 58)

    # Dataset loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    if verbose:
        print("🔄 Data Loaders Ready for Action!")
        print("   🚂 Training Loader:    Batches Ready")
        print("   🚂 Testing Loader:     Batches Ready")
        print("   🚂 Validation Loader:  Batches Ready")
        print("=" * 60)
        print("🚀 Dataset and Loaders are fully initialized and ready!")

    return train_loader, val_loader, test_loader

def train_validate(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, learning_rate: float, best_model_path: str, device: torch.device, config: dict, verbose: bool = True) -> None:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    early_stop = EarlyStopping(patience=5, verbose=verbose, path=best_model_path)

    wandb.init(project="your_project_name", config=config)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - Train loss: {train_loss / len(train_loader):.4f}", end=" ")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        if verbose:
            print(f'- Val loss: {val_loss / len(val_loader):.4f}')

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss / len(train_loader), "val_loss": val_loss / len(val_loader)})

        early_stop(val_loss / len(val_loader), model)
        if early_stop.early_stop:
            print("Early stopping")
            break

    print('Finished Training')


# -------------------------------------------------------------------------------- #
#                                                                                  #
#                               Main Function                                      #
#                                                                                  #
# -------------------------------------------------------------------------------- #

def main(args):
    print("🚀 Starting Main Function...")

    # Paths to the dataset
    dataset_path = "D:/chris/Documents/UofC/MEng Soft/winter/ENEL 645/ENEL 645/ENEL 645/project/small_dataset/"
    model_save_path = "D:/chris/Documents/UofC/MEng Soft/winter/ENEL 645/ENEL 645/ENEL 645/project/best_model/model.pth"
    print("📂 Dataset and Model paths are set!")

    # Initialize dataset and loader
    train_loader, val_loader, test_loader = initialize_dataset_and_loader(dataset_path, True)
    print("✅ Dataset and loaders initialized successfully!")

    # Device configuration
    print("💻 Configuring device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device configured to use {'CUDA' if device.type == 'cuda' else 'CPU'}!")

    # Model instantiation
    print("🛠️ Instantiating the model...")
    # model = Model()
    # model.to(device)
    print("🏗️ Model built and moved to device!")

    if args.train:
        print("🏋️ Starting training process...")
        # Training logic here
        # print("✅ Training completed!")

    if args.test:
        print("🔍 Running tests...")
        # Testing logic here
        # print(f"📊 Test Loss: {test_loss}. Test Accuracy: {test_accuracy}")
        # print("✅ Testing completed!")

    print("🎉 Main Function Execution Completed Successfully!")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train and/or test model.")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)