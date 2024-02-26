from glob import glob
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
# import torch
# from torchvision import transforms

def list_images(images_path: str) -> np.ndarray:
    """
    List all images in the given path.
    """
    images = glob(images_path, recursive=True)
    return np.array(images)

def extract_labels(images: np.ndarray) -> tuple:
    """
    Extract labels from image paths.
    """
    labels = np.array([f.replace("\\", "/").split("/")[-2] for f in images])
    classes = np.unique(labels)
    return labels, classes

def convert_labels_to_int(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    Convert string labels to integers.
    """
    label_to_int = {label: i for i, label in enumerate(classes)}
    labels_int = np.array([label_to_int[label] for label in labels])
    return labels_int

def list_data_and_prepare_labels(images_path: str) -> tuple:
    """
    List all images, extract labels, and prepare them for training.
    """
    images = list_images(images_path)
    labels, classes = extract_labels(images)
    labels_int = convert_labels_to_int(labels, classes)
    return images, labels_int, classes

def split_data(images: np.ndarray, labels: np.ndarray, val_split: float, test_split: float, random_state: int = 10) -> tuple:
    """
    Split data into train, validation, and test sets and return them as dictionaries.
    """
    # Splitting the data into dev and test sets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=random_state)
    dev_index, test_index = next(sss.split(images, labels))
    dev_images, dev_labels = images[dev_index], labels[dev_index]
    test_images, test_labels = images[test_index], labels[test_index]

    # Splitting the data into train and val sets
    val_size = int(val_split * len(images))
    val_split_adjusted = val_size / len(dev_images)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_split_adjusted, random_state=random_state)
    train_index, val_index = next(sss2.split(dev_images, dev_labels))

    # Creating train, validation, and test dictionaries
    train_images = images[train_index]
    train_labels = labels[train_index]
    val_images = images[val_index]
    val_labels = labels[val_index]

    train_set = {"X": train_images, "Y": train_labels}
    val_set = {"X": val_images, "Y": val_labels}
    test_set = {"X": test_images, "Y": test_labels}

    return {"Train": train_set, "Validation": val_set, "test": test_set}

# def create_data_loaders(train_set, val_set, test_set, batch_size, train_transform, test_transform):
#     """
#     Create data loaders for train, validation, and test sets.
#     """
#     train_dataset = TorchVisionDataset(train_set, transform=train_transform)
#     val_dataset = TorchVisionDataset(val_set, transform=train_transform)
#     test_dataset = TorchVisionDataset(test_set, transform=test_transform)

#     trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#     valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
#     testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

#     return trainloader, valloader, testloader

# def get_data_loaders(images_path, val_split, test_split, batch_size=32, verbose=True):
#     images, labels_int, classes = list_data_and_prepare_labels(images_path)
#     train_images, train_labels, val_images, val_labels, test_images, test_labels = split_data(images, labels_int, val_split, test_split)

#     if verbose:
#         print_dataset_statistics(train_labels, val_labels, test_labels, classes)

#     train_transform, test_transform = apply_transforms()
#     trainloader, valloader, testloader = create_data_loaders({"X": train_images, "Y": train_labels}, {"X": val_images, "Y": val_labels}, {"X": test_images, "Y": test_labels}, batch_size, train_transform, test_transform)

#     return trainloader, valloader, testloader

# def print_dataset_statistics(train_labels, val_labels, test_labels, classes):
#     """
#     Print statistics of the dataset.
#     """
#     print(f"Number of images in the dataset: {len(train_labels) + len(val_labels) + len(test_labels)}")
#     for class_index, class_name in enumerate(classes):
#         print(f"Number of images in class {class_name}: {(train_labels == class_index).sum() + (val_labels == class_index).sum() + (test_labels == class_index).sum()}")