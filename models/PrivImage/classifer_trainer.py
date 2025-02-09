
import argparse
import os
import time
import logging

import torch
from torch import nn
from torchvision import models
import torch.optim as optim
from torchvision.datasets import ImageFolder, CIFAR10, MNIST, ImageNet, Places365
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from data.SpecificImagenet import SpecificClassImagenet
from data.SpecificPlaces365 import SpecificClassPlaces365
from models.PrivImage import resnet

criterion = nn.CrossEntropyLoss()


def train(net, loader, optimizer, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if device == 0:
            print(loss.item())
    return correct/total


def test(net, loader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return correct/total


class MyClassifier(nn.Module):
    def __init__(self, model="resnet", num_classes=365):
        super(MyClassifier, self).__init__()
        if model == "resnet50":
            self.model = resnet.ResNet50(num_classes=num_classes)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.model(x)


def train_classifier(model, config):
    # Get the local rank of the current process
    rank = config.setup.local_rank
    
    # Log training start message if this is the main process
    if rank == 0:
        logging.info("Training Semantic Query Function")
    
    # Set up training parameters
    img_size = 32
    max_epoch = 100
    lr = 1e-2
    batch_size = 2048
    val_batch_size = 8192
    num_workers = 8

    # Adjust batch sizes and number of workers based on the global size
    batch_size = batch_size // config.setup.global_size
    val_batch_size = val_batch_size // config.setup.global_size
    num_workers = num_workers // config.setup.global_size

    # Load the appropriate dataset based on the configuration
    if config.public_data.name == "imagenet":
        # Define transformations for ImageNet dataset
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        
        # Load ImageNet dataset
        train_dataset = SpecificClassImagenet(root=config.public_data.train_path, split="train", transform=train_transform)
        val_dataset = SpecificClassImagenet(root=config.public_data.train_path, split="val", transform=val_transform)
    
    elif config.public_data.name == "places365":
        # Check if the dataset needs to be downloaded
        download = not os.path.exists(os.path.join(config.public_data.train_path, "data_256_standard"))
        
        # Load Places365 dataset
        dataset = Places365(root=config.public_data.train_path, small=True, transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ]), download=download)
        
        # Split the dataset into training and validation sets
        train_size = int(len(dataset) * 0.9)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Wrap the model with DistributedDataParallel for multi-GPU training
    model = DDP(model, device_ids=[rank])
    
    # Set up the optimizer and learning rate scheduler
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
    
    # Create data loaders for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, sampler=DistributedSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers, drop_last=False)

    # Initialize the best accuracy
    best_acc = 0
    
    # Training loop
    for epoch in range(max_epoch):
        # Set the epoch for the distributed sampler
        train_loader.sampler.set_epoch(epoch)
        
        # Train the model for one epoch
        train_acc = train(model, train_loader, optimizer, rank)
        
        # Log training accuracy if this is the main process
        if rank == 0 or True:
            logging.info('Epoch: {} Train Acc: {}'.format(epoch, train_acc))
            
            # Evaluate the model on the validation set
            val_acc = test(model, val_loader, rank)
            logging.info('Val Acc: {}'.format(val_acc))
            
            # Save the model if it achieves the best validation accuracy
            if val_acc > best_acc:
                logging.info('Saving..')
                torch.save(model.state_dict(), 'models/pretrained_models/{}_classifier_ckpt.pth'.format(config.public_data.name))
                best_acc = val_acc
        
        # Step the learning rate scheduler
        scheduler.step()
    
    # Synchronize all processes and clear the GPU cache
    dist.barrier()
    torch.cuda.empty_cache()
    
    # Return the path to the saved model checkpoint
    return 'models/pretrained_models/{}_classifier_ckpt.pth'.format(config.public_data.name)
