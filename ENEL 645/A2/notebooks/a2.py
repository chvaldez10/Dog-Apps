# Import Libraries
import wandb
import torch
from glob import glob
import matplotlib.pylab as plt
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.models import resnet18, efficientnet_b4
from torchvision import transforms, models
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedShuffleSplit

# Constants
EPOCHS = 12
LEARNING_RATE = 2e-4
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2
BATCH_SIZE = 32
INPUT_SHAPE = (3, 380, 380)  # EfficientNet B4
NUM_CLASSES = 4

# Function definitions and classes
class GarbageModel(pl.LightningModule):
    def __init__(self, input_shape: tuple, num_classes: int, learning_rate: float = 2e-4, transfer: bool = False):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.input_shape = input_shape

        self.num_classes = num_classes
        
        # transfer learning if pretrained=True
        self.feature_extractor = models.efficientnet_b4(pretrained=transfer)

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

class BaseDataset(Dataset):
    def __init__(self, data_dic: dict, transform: transforms.transforms.Compose = None):
        self.file_paths = data_dic["X"]
        self.labels = data_dic["Y"]
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        
        # Read an image with PIL and convert it to RGB
        image = Image.open(file_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Convert label to a Long tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label

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

def train_validate(model: GarbageModel, train_loader: BaseDataset, val_loader: BaseDataset, epochs: int, learning_rate: float, best_model_path: str, device: torch.device, verbose: bool = True) -> None:
    wandb.init(
        project="enel-645-garbage-classifier",
        name="test-run",
        config={"learning_rate": 0.02, "architecture": "efficientNet_b4", "dataset": "CVPR_2024_dataset", "epochs": 12}
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    best_loss = 1e+20

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
        if verbose:
            print(f'Epoch {epoch + 1}, Train loss: {train_loss / len(train_loader):.3f}', end=' ')

        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            if verbose:
                print(f'Val loss: {val_loss / len(val_loader):.3f}')

        # Log training and validation loss to wandb
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss:
            if verbose:
                print("Saving model")
            torch.save(model.state_dict(), best_model_path)
            best_loss = val_loss

    if verbose:
        print('Finished Training')

# Main loop
def main_loop():
    """
    The main loop where the core logic of the script is executed.

    Returns:
    None
    """
    normalized_path = "/work/TALC/enel645_2024w/CVPR_2024_dataset"
    best_model_path = "/home/christian.valdez/ENSF-611-ENEL-645/ENEL 645/A2/best_dataset/garbage_net.pth"

    images_path = normalized_path + "/**/*.png"
    images, labels_int, classes = list_data_and_prepare_labels(images_path)
    all_dataset = split_data(images, labels_int, VAL_SPLIT, TEST_SPLIT)
    train_set = all_dataset["Train"]
    val_set = all_dataset["Validation"]
    test_set = all_dataset["test"]

    torch_vision_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4120, 0.3768, 0.3407],
            std=[0.2944, 0.2759, 0.2598],
        )
    ])

    torch_vision_transform_test = transforms.Compose([
        # transforms.Resize((224, 224)), #resnet18
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4120, 0.3768, 0.3407],
            std=[0.2944, 0.2759, 0.2598],
        )
    ])

    # Get the dataset
    train_dataset = BaseDataset(train_set, transform=torch_vision_transform)
    val_dataset = BaseDataset(val_set, transform=torch_vision_transform)
    test_dataset = BaseDataset(test_set,transform=torch_vision_transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    efficientNet_b4 = GarbageModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, transfer=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    efficientNet_b4.to(device)

    train_validate(efficientNet_b4, train_loader, val_loader, EPOCHS, LEARNING_RATE, best_model_path, device)

    # Load the best model to be used in the test set
    net = GarbageModel((3,224,224), 4, False)
    net.load_state_dict(torch.load(best_model_path))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

# Main entry point
if __name__ == "__main__":
    print("Starting the training")

    # Call the main loop function
    main_loop()

    print("Finish training")