import torch
from torch.utils.data import Dataset, DataLoader


class SpecificClassEMNIST(Dataset):
    def __init__(self, original_dataset, specific_class):
        print(specific_class)
        self.original_dataset = original_dataset
        self.targets = []
        self.indices = []
        selected_classes = []
        public_to_sensitive = {}
        for sensitive_cls in specific_class:
            for public_cls in specific_class[sensitive_cls]:
                selected_classes.append(public_cls)
                public_to_sensitive[int(public_cls)] = int(sensitive_cls)
        print(selected_classes)
        for i, label in enumerate(original_dataset.targets):
            label = int(label)
            if label in selected_classes:
                self.targets.append(public_to_sensitive[label])
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.original_dataset[original_idx][0], self.targets[idx]