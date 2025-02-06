import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import os


class SpecificClassPlaces365(Dataset):
    def __init__(self, original_dataset, specific_class):
        # print(specific_class)
        self.original_dataset = original_dataset
        self.targets = []
        self.indices = []
        selected_classes = []
        public_to_sensitive = {}
        for sensitive_cls in specific_class:
            for public_cls in specific_class[sensitive_cls]:
                selected_classes.append(public_cls)
                public_to_sensitive[int(public_cls)] = int(sensitive_cls)
        # print(selected_classes)
        for i, label in enumerate(original_dataset.targets):
            if label in selected_classes:
                self.targets.append(public_to_sensitive[label])
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.original_dataset[original_idx][0], self.targets[idx]


def SpecificClassPlaces365_ldm(root, split, image_size, c, specific_class=None):
    download = (not os.path.exists(os.path.join(root, "data_256_standard")))
    transform = [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    if c == 1:
        transform = [transforms.Grayscale(num_output_channels=1)] + transform
    transform = transforms.Compose(transform)
    public_train_set_ = torchvision.datasets.Places365(root=root, small=True, download=download, transform=transform)
    if specific_class is None:
        public_train_set = public_train_set_
    else:
        public_train_set = SpecificClassPlaces365(public_train_set_, specific_class)
    return public_train_set