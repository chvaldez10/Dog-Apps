
from torchvision import transforms

torch_vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4120, 0.3768, 0.3407],
        std=[0.2944, 0.2759, 0.2598],
    )
])

torch_vision_transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4120, 0.3768, 0.3407],
        std=[0.2944, 0.2759, 0.2598],
    )
])