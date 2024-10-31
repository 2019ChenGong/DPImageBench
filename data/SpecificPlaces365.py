import torch
from torch.utils.data import Dataset, DataLoader


class SpecificClassPlaces365(Dataset):
    def __init__(self, original_dataset, specific_class):
        self.original_dataset = original_dataset
        self.targets = []
        self.indices = []
        selected_classes = []
        public_to_sensitive = {}
        for sensitive_cls in specific_class:
            selected_classes.append(specific_class[sensitive_cls])
            for public_cls in specific_class[sensitive_cls].numpy():
                public_to_sensitive[int(public_cls)] = int(sensitive_cls)
        for i, label in enumerate(original_dataset.targets):
            if label in selected_classes:
                self.targets.append(public_to_sensitive[label])
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.original_dataset[original_idx][0], self.targets[idx]