import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
import os
import copy
import time
import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import ssl
import pretrainedmodels

# Initialize wandb
wandb.init(project='dog_breed_classification')

# Data augmentation and normalization for training
# Just normalization for validation and test
data_transforms = {
    'Train': transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'Test': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'Validation': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

data_dir = '/home/cmychung/enel645_project/dataset-143-classes'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['Train', 'Test', 'Validation']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=2) for x in ['Train', 'Test', 'Validation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Test', 'Validation']}
class_names = image_datasets['Train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epoch_loss_train = []
    epoch_loss_val = []

    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, -1)

                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'Train':
                scheduler.step()
                epoch_loss_train.append(epoch_loss)
            elif phase == 'Validation':
                epoch_loss_val.append(epoch_loss)
            
            wandb.log({f'{phase}_loss': epoch_loss, f'{phase}_acc': epoch_acc})

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model.pth')

        print('Average Train Loss: {:.4f}'.format(sum(epoch_loss_train) / len(epoch_loss_train)))
        print('Average Validation Loss: {:.4f}'.format(sum(epoch_loss_val) / len(epoch_loss_val)))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

def test_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    for inputs, labels in dataloaders['Test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=-1)
        _, preds = torch.max(probs, -1)

        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / dataset_sizes['Test']
    test_acc = running_corrects.double() / dataset_sizes['Test']

    num_classes = len(class_names)
    subset_size = num_classes // 4
    class_subsets = [class_names[i:i+subset_size] for i in range(0, num_classes, subset_size)]

    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=class_names)

    conf_matrices = []

    for subset_names in class_subsets:
        subset_indices = [class_names.index(name) for name in subset_names]
        subset_labels = [all_labels[i] for i in range(len(all_labels)) if all_labels[i] in subset_indices]
        subset_preds = [all_preds[i] for i in range(len(all_preds)) if all_labels[i] in subset_indices]

        conf_matrix_subset = confusion_matrix(subset_labels, subset_preds, labels=subset_indices)
        conf_matrices.append(conf_matrix_subset)

    wandb.log({'test_loss': test_loss, 'test_acc': test_acc})

    np.set_printoptions(threshold=np.inf)

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))
    print('Confusion Matrix:\n', conf_matrix)
    print('Classification Report:\n', class_report)

    for i, conf_matrix_subset in enumerate(conf_matrices):
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_subset, annot=True, fmt='d', cmap='Blues', xticklabels=class_subsets[i], yticklabels=class_subsets[i])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix Subset {i+1}')
        plt.savefig(f'confusion_matrix_subset_{i+1}.png', bbox_inches='tight')

ssl._create_default_https_context = ssl._create_unverified_context

# Load the pretrained Inception-ResNet-V2 model
model_ft = pretrainedmodels.inceptionresnetv2(pretrained='imagenet')
num_ftrs = model_ft.last_linear.in_features

# Replace the last fully connected layer with a new one that has 142 output features
model_ft.last_linear = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD([
    {'params': model_ft.parameters(), 'lr': 0.001}
], lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# Tran the model
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=15)

# Load the saved weights of the trained model (if no testing is required; otherwise comment it out)
model_ft.load_state_dict(torch.load('best_model.pth'))

print('Testing the model:')
test_model(model_ft, criterion)
