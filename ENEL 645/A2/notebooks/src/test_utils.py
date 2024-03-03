import torch

# Custom class
from src.dataset import BaseDataset
from src.garbage_model import GarbageModel

def test(model: GarbageModel, test_loader: BaseDataset, device: torch.device) -> None:
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():  # No gradient calculation for inference
        for images, labels in test_loader:
            outputs = model(images) 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f} %')
