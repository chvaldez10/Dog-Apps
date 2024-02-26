import glob
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
# import torch
# from torchvision import transforms

def list_data_and_prepare_labels(images_path):
    """
    List all images and prepare their labels.
    """
    images = glob.glob(images_path + "*/*.jpg")
    images = np.array(images)
    labels = np.array([f.split("/")[-2] for f in images])

    # Convert string labels to integers
    classes = np.unique(labels)
    label_to_int = {label: i for i, label in enumerate(classes)}
    labels_int = np.array([label_to_int[label] for label in labels])

    return images, labels_int, classes

def split_data(images, labels, val_split, test_split):
    """
    Split data into train, validation, and test sets.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=10)
    dev_index, test_index = next(sss.split(images, labels))
    dev_images, dev_labels = images[dev_index], labels[dev_index]
    test_images, test_labels = images[test_index], labels[test_index]

    val_size = int(val_split * len(images))
    val_split_adjusted = val_size / len(dev_images)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_split_adjusted, random_state=10)
    train_index, val_index = next(sss2.split(dev_images, dev_labels))

    return images[train_index], labels[train_index], images[val_index], labels[val_index], test_images, test_labels

def apply_transforms(mean_train=None, std_train=None):
    """
    Apply transformations to the datasets.
    """
    base_transform = [transforms.Resize((224, 224)), transforms.ToTensor()]
    if mean_train is not None and std_train is not None:
        normalize_transform = transforms.Normalize(mean=mean_train, std=std_train)
        train_transform = transforms.Compose(base_transform + [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), normalize_transform])
        test_transform = transforms.Compose(base_transform + [normalize_transform])
    else:
        train_transform = transforms.Compose(base_transform + [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
        test_transform = transforms.Compose(base_transform)

    return train_transform, test_transform

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