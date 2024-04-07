import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
import time
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Initialize wandb
wandb.init(project='dog_breed_classification')

# Data augmentation and normalization for training
# Just normalization for validation and test
data_transforms = {
    'Train': transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(degrees=15),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Test': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Validation': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/home/cmychung/enel645_project/dataset-143-classes'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['Train', 'Test', 'Validation']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=24, shuffle=True, num_workers=2) for x in ['Train', 'Test', 'Validation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Test', 'Validation']}
class_names = image_datasets['Train'].classes
# print(len(class_names))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Lists to store loss values for each epoch
    epoch_loss_train = []
    epoch_loss_val = []

    # Define the number of layers to unfreeze per epoch
    num_layers_to_unfreeze_per_epoch = 3

    for epoch in range(num_epochs):
        print('-' * 10)
        unfreeze_layers(model_ft, epoch, num_layers_to_unfreeze_per_epoch)

        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # print(phase)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print('Iterating...')

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'Train'):
                    if phase == 'Train':
                        outputs, aux_outputs, *additional_outputs = model(inputs)
                        # print('outputs:', outputs.shape)
                    if phase == 'Validation':
                        outputs = model(inputs)
                        # print('outputs:', outputs.shape)

                    # print("calculating loss")
                    # Calculate loss using raw logits and target labels
                    loss = criterion(outputs, labels)

                    # Apply softmax along the class dimension to get probabilities
                    probs = torch.nn.functional.softmax(outputs, dim=-1)
                    # print('probs:', probs.shape)

                    # Get predictions
                    _, preds = torch.max(probs, -1)
                    # print('preds:', preds.shape)

                    # Backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    # print(labels.data)
                    running_corrects += torch.sum(preds == labels.data)

            # print('Done iterating')

            # Calculate average loss for train and validation phases
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'Train':
                scheduler.step()
                epoch_loss_train.append(epoch_loss)
            elif phase == 'Validation':
                epoch_loss_val.append(epoch_loss)
            
            # epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Log metrics to wandb
            wandb.log({f'{phase}_loss': epoch_loss, f'{phase}_acc': epoch_acc})

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Save the model if it has the best validation accuracy
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model
                torch.save(model.state_dict(), 'best_model.pth')

        # Print average loss for the epoch
        print('Average Train Loss: {:.4f}'.format(sum(epoch_loss_train) / len(epoch_loss_train)))
        print('Average Validation Loss: {:.4f}'.format(sum(epoch_loss_val) / len(epoch_loss_val)))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def unfreeze_layers(model, epoch, num_layers_to_unfreeze):
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'fc' in name:  # Skip fully connected layer
            continue
        if i < (epoch * num_layers_to_unfreeze):
            param.requires_grad = True
            print(f'Unfreezing layer: {name}')
        else:
            param.requires_grad = False

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

    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=class_names)

    # Log test metrics to wandb
    wandb.log({'test_loss': test_loss, 'test_acc': test_acc})

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))
    # print('Confusion Matrix:\n', conf_matrix)
    print('Classification Report:\n', class_report)
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Load the pretrained Inception-v3 model
model_ft = models.inception_v3(pretrained=True)
print('INCEPTION V3 MODEL:', model_ft)
num_ftrs = model_ft.fc.in_features

# Freeze parameters of the pretrained layers
# for param in model_ft.parameters():
#     param.requires_grad = False

# Replace the last fully connected layer with a new one that has 143 output features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Freeze parameters of early layers up to 'Mixed_6'
# for name, param in model_ft.named_parameters():
#     if 'Mixed_5' not in name or 'Mixed_6' not in name or 'Mixed_7' not in name:  # Freeze layers before 'Mixed_6'
#         param.requires_grad = False

# Unfreeze the parameters of some layers closer to the input data
# for name, param in model_ft.named_parameters():
#     if 'Mixed_5' in name or 'Mixed_6' in name or 'Mixed_7' in name:  # Unfreeze layers from 'Mixed_6' onwards
#         param.requires_grad = True

# Specify different learning rates for different parameter groups
optimizer_ft = optim.SGD([
    {'params': model_ft.Conv2d_1a_3x3.parameters(), 'lr': 0.0001},  # Example: Lower LR for early layers
    {'params': model_ft.fc.parameters(), 'lr': 0.001}  # Higher LR for the new fully connected layer
], lr=0.0001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=15)

# Test the model
print('Testing the model:')
test_model(model_ft, criterion)