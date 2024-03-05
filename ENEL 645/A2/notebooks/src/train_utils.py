import torch.nn as nn
import torch
import wandb

# Custom class
from src.base_dataset import BaseDataset
from src.garbage_model import GarbageModel

wandb.init(
    project="enel-645-garbage-classifier",
    name="test-run",
    config={"learning_rate": 0.02, "architecture": "resnet_18", "dataset": "CVPR_2024_dataset", "epochs": 12}
)

def train_validate(model: GarbageModel, train_loader: BaseDataset, val_loader: BaseDataset, epochs: int, learning_rate: float, best_model_path: str, device: torch.device, verbose: bool = True) -> None:
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