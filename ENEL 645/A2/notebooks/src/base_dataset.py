from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

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