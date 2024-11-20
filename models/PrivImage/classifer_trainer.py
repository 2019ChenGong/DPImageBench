
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

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from SpecificImagenet import SpecificClassImagenet
from classifier_models import resnet

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


def main(args):
    world_size = torch.cuda.device_count()
    args.ddp = world_size > 1
    if args.ddp:
        dist.init_process_group("nccl", init_method='env://')
        rank = dist.get_rank()
        rank = rank % world_size
        torch.cuda.set_device(rank)
    else:
        rank = 0

    if rank == 0:
        t = time.localtime()
        main_name = 'train_{}_classifier'.format(args.dataset)
        exp_name = '{}_{}_{}_{}_{}_{}_{}'.format(main_name, t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min,
                                                 t.tm_sec)
        if not os.path.exists(exp_name):
            os.mkdir(exp_name)
            os.mkdir('{}/weights'.format(exp_name))
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler("{}/log.txt".format(exp_name), mode='a')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.info(args)

    batch_size = args.batch_size // world_size
    val_batch_size = args.val_batch_size // world_size
    num_workers = args.num_workers // world_size

    model = MyClassifier(model=args.model).to(rank)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    if args.ddp:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    if args.dataset == "imagenet":
        train_dataset = SpecificClassImagenet(root='/p/fzv6enresearch/DPImageBench/dataset/imagenet/imagenet_32', split="train", transform=transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)
         ]))
        val_dataset = SpecificClassImagenet(root='/p/fzv6enresearch/DPImageBench/dataset/imagenet/imagenet_32', split="val", transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)
         ]))
    else:
        dataset = Places365(root="/p/fzv6enresearch/DPImageBench/dataset/places365", small=True, transform=transforms.Compose(
        [
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)
         ]))
        train_size = int(len(dataset) * 0.9)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if rank == 0:
        print(len(train_dataset))
        print(len(val_dataset))

    if args.ddp:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, sampler=DistributedSampler(train_dataset))
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers, drop_last=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                  drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers,
                                drop_last=False)

    best_acc = 0
    for epoch in range(args.epoch):
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)
        train_acc = train(model, train_loader, optimizer, rank)
        if rank == 0:
            print('Epoch: {} Train Acc: {}'.format(epoch, train_acc))
            logger.info('Epoch: {} Train Acc: {}'.format(epoch, train_acc))

            val_acc = test(model, val_loader, rank)
            logger.info('Val Acc: {}'.format(val_acc))
            print('Val Acc: {}'.format(val_acc))
            if val_acc > best_acc:
                logger.info('Saving..')
                print('Saving..')
                torch.save(model.state_dict(), '{}/weights/{:.3f}_ckpt.pth'.format(exp_name, val_acc))
                best_acc = val_acc
        scheduler.step()


def train_classifier(model, config):
    img_size = 32
    epoch = 300
    lr = 1e-2
    batch_size = 2048
    val_batch_size = 8192

    if config.public_data.name == "imagenet":
        train_dataset = SpecificClassImagenet(root=config.public_data.train_path, split="train", transform=transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)
         ]))
        val_dataset = SpecificClassImagenet(root=config.public_data.train_path, split="val", transform=transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)
         ]))
    elif config.public_data.name == "places365":
        download = (not os.path.exists(os.path.join(config.public_data.train_path, "data_256_standard")))
        public_train_set_ = torchvision.datasets.Places365(root=config.public_data.train_path, small=True, download=download, transform=trans)

        dataset = Places365(root=config.public_data.train_path, small=True, transform=transforms.Compose(
        [
        transforms.Resize(32),
        transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)
         ]))
        train_size = int(len(dataset) * 0.9)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    model = DDP(model, device_ids=[rank], output_device=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32, drop_last=True, sampler=DistributedSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=32, drop_last=False)

    best_acc = 0
    for epoch in range(args.epoch):
        train_loader.sampler.set_epoch(epoch)
        train_acc = train(model, train_loader, optimizer, rank)
        if rank == 0:
            print('Epoch: {} Train Acc: {}'.format(epoch, train_acc))
            logger.info('Epoch: {} Train Acc: {}'.format(epoch, train_acc))

            val_acc = test(model, val_loader, rank)
            logger.info('Val Acc: {}'.format(val_acc))
            print('Val Acc: {}'.format(val_acc))
            if val_acc > best_acc:
                logger.info('Saving..')
                print('Saving..')
                torch.save(model.state_dict(), '{}/weights/{:.3f}_ckpt.pth'.format(exp_name, val_acc))
                best_acc = val_acc
        scheduler.step()

    dist.barriar()
    return model