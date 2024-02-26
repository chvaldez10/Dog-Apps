import wandb
import torch
import os
import sys
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms, models, datasets
from PIL import Image
import numpy as np
import glob
from sklearn.model_selection import StratifiedShuffleSplit
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import ssl
import torchvision.models as models
import matplotlib.pyplot as plt
from skimage import transform
from sklearn.metrics import confusion_matrix

wandb.init(project='garbage_classifier_local', entity='yeneirvine')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, data_dic, transform=None):
        self.file_paths = data_dic["X"]
        self.labels = data_dic["Y"]
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]

        image = Image.open(file_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label
class_names = ['Black', 'Green', 'Blue', 'TTR']

data_url = '/work/TALC/enel645_2024w/CVPR_2024_dataset'
model_path = '/home/yene.irvine/software/src/645_assignment_2/output_model_6/model_state_dict.pth'
epochs = 12
test_split = 0.2
val_split = 0.2
batch_size = 64

# STEP 1 - LOAD DATASET
images = glob.glob(data_url + "/**/*.png", recursive=True)
images = np.array(images)
labels = np.asarray([f.split("/")[-2] for f in images])

classes = np.unique(labels).flatten()
labels_int = np.zeros(labels.size, dtype=np.int64)

# Convert string labels to integers
for ii, jj in enumerate(classes):
    labels_int[labels == jj] = ii

# Splitting the data in dev and test sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=10)
sss.get_n_splits(images, labels_int)
dev_index, test_index = next(sss.split(images, labels_int))

dev_images = images[dev_index]
dev_labels = labels_int[dev_index]

test_images = images[test_index]
test_labels = labels_int[test_index]

# Splitting the data in train and val sets
val_size = int(val_split*images.size)
val_split = val_size/dev_images.size
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=10)
sss2.get_n_splits(dev_images, dev_labels)
train_index, val_index = next(sss2.split(dev_images, dev_labels))

train_images = images[train_index]
train_labels = labels_int[train_index]

val_images = images[val_index]
val_labels = labels_int[val_index]

# Representing the sets as dictionaries
train_set = {"X": train_images, "Y": train_labels}
val_set = {"X": val_images, "Y": val_labels}
test_set = {"X": test_images, "Y": test_labels}

# Transforms
torchvision_transform_train = transforms.Compose([transforms.Lambda(lambda x: x.convert('RGB')),transforms.Resize((224, 224)),
                                                    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                                    transforms.ToTensor()])

# Datasets
train_dataset_unorm = CustomDataset(train_set, transform=torchvision_transform_train)
trainloader_unorm = torch.utils.data.DataLoader(train_dataset_unorm, batch_size=batch_size, shuffle=True, num_workers=0)

mean_train = 0
std_train = 0
nb_samples = 0
for data in trainloader_unorm:
    data = data[0]
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean_train += data.mean(2).sum(0)
    std_train += data.std(2).sum(0)
    nb_samples += batch_samples

max_train = data.max()
min_train = data.min()
mean_train /= nb_samples
std_train /= nb_samples

mean_train = mean_train[:3]
std_train = std_train[:3]

torchvision_transform = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                            transforms.ToTensor(), transforms.Normalize(mean=mean_train, std=std_train)])

torchvision_transform_test = transforms.Compose([transforms.Resize((224, 224)),
                                                    transforms.ToTensor(), transforms.Normalize(mean=mean_train, std=std_train)])

# Get the train/val/test loaders
train_dataset = CustomDataset(train_set, transform=torchvision_transform)
val_dataset = CustomDataset(val_set, transform=torchvision_transform)
test_dataset = CustomDataset(test_set, transform=torchvision_transform_test)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

print("Statistics of training set: ")
print("Mean:", mean_train)
print("Std:", std_train)
print("Min:", min_train)
print("Max:", max_train)

print("Number of Images: ", len(images))

#NEED TO ADD SCALING, USE WHATEVER THE ORIGIALLY PRETRAIED MODEL DOES

best_loss = sys.maxsize

#MODEL DEFINITION
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
model.to(device)  # Ensure the model is on the correct device
num_classes = 4  # Number of classes in your dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Criterion, Optimizer, and Learning Rate Scheduler definition
criterion = nn.CrossEntropyLoss()  # Define the loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # Define the optimizer
lr_scheduler = ExponentialLR(optimizer, gamma=0.8)  # Define the LR scheduler

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Log every 20 steps
        if i % 20 == 0:
            wandb.log({"batch_train_loss": loss.item()})
            
    avg_train_loss = train_loss / len(trainloader)
    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss}')

    # Adjust learning rate
    lr_scheduler.step()

    model.eval()  # Switch to evaluation mode for validation
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(valloader)
    print(f'Epoch [{epoch + 1}/{epochs}], Val Loss: {avg_val_loss}')

    # Save model if validation loss has improved
    if avg_val_loss < best_loss:
        print("Saving model with improved validation loss: ", avg_val_loss)
        torch.save(model.state_dict(), model_path)
        best_loss = avg_val_loss

    model.train()  # Switch back to training mode
    wandb.log({"epoch": epoch, "avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss})

    
print('Finished Training')

# Evaluate accuracy
model.eval()  # Ensure the model is in evaluation mode
correct = 0
total = 0

num_classes = len(class_names)  # Assuming class_names list is defined with all your class names
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

test_accuracy = 100 * correct / total
print(f'Accuracy of the network on the test images: {test_accuracy}%')
wandb.log({"test_accuracy": test_accuracy})

for i, class_name in enumerate(class_names):
    if class_total[i] > 0:
        class_accuracy = 100 * class_correct[i] / class_total[i]
        print(f'Accuracy of {class_name} : {class_accuracy:.2f}%')
        wandb.log({f"{class_name}_accuracy": class_accuracy})
    else:
        print(f'Accuracy of {class_name} : N/A')

# Initialize variables for the confusion matrix
all_labels = []
all_predictions = []

# Disable gradient computation for evaluation
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Append true labels and predictions to lists
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())


# Compute confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Print numbers in each cell
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Save the figure
plt.savefig('/home/yene.irvine/software/src/645_assignment_2/outputs/confusion_matrix_9.png') 