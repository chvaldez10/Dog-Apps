"""
This script is designed for training and testing a 3D Convolutional Neural Network (CNN) model
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
from torchvision import transforms, datasets
import torch.optim as optim
import os
from typing import List, Tuple, Dict
from itertools import product
from PIL import Image
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import wandb

# Constants
DATASET_SEGMENTS = ["Train", "Test", "Validation"]
BATCH_SIZE = 32
NUM_WORKERS = 4
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
                        self.file_paths.append(os.path.join(label_dir, file_name))
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

def get_data_transforms() -> Dict[str, transforms.Compose]:
    """
    Returns the data transformations for each dataset type.
    """
    return {
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

def initialize_dataset_and_loader(dataset_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Initializes the dataset and data loaders for training, validation, and testing.
    """
    data_transforms = get_data_transforms()
    
    # custom dataset
    train_dataset = DogDataset(dataset_path, "Train", data_transforms["Train"])
    test_dataset = DogDataset(dataset_path, "Test", data_transforms["Test"])
    val_dataset = DogDataset(dataset_path, "Validation", data_transforms["Validation"])

    # dataset loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader

# -------------------------------------------------------------------------------- #
#                                                                                  #
#                               Main Function                                      #
#                                                                                  #
# -------------------------------------------------------------------------------- #

def main(args):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((180, 320)),
        transforms.ToTensor(),
    ])

    # Paths to the dataset
    dataset_path = "D:/chris/Documents/UofC/MEng Soft/winter/ENEL 645/ENEL 645/ENEL 645/project/small_dataset/"
    model_save_path = "best_model/model.pth"

    # Initialize dataset and loader
    initialize_dataset_and_loader(dataset_path)
    train_loader, val_loader, test_loader = initialize_dataset_and_loader(dataset_path)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model instantiation
    # model = Model()
    # model.to(device)

    if args.train:
        pass
    
    if args.test:
        pass

        # print(f"test_loss: {test_loss}. test_accuracy: {test_accuracy}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train and/or test model.")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)