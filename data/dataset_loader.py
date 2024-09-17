import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import logging


from data.stylegan3.dataset import ImageFolderDataset
from data.SpecificImagenet import SpecificClassImagenet
from models.PrivImage import resnet

def load_sensitive_data(config):
    
    sensitive_train_set = ImageFolderDataset(
            config.sensitive_data.train_path, config.sensitive_data.resolution, config.sensitive_data.num_channels, use_labels=True)
    sensitive_test_set = ImageFolderDataset(
        config.sensitive_data.test_path, config.sensitive_data.resolution, config.sensitive_data.num_channels, use_labels=True)

    # if config.sensitive_data.name == "mnist":
    #     sensitive_train_set = torchvision.datasets.MNIST(root=config.sensitive_data.train_path, train=True, download=True, transform=transforms.ToTensor())
    #     sensitive_test_set = torchvision.datasets.MNIST(root=config.sensitive_data.test_path, train=False, download=True, transform=transforms.ToTensor())
    # elif config.sensitive_data.name == "fmnist":
    #     sensitive_train_set = torchvision.datasets.FashionMNIST(root=config.sensitive_data.train_path, train=True, download=True, transform=transforms.ToTensor())
    #     sensitive_test_set = torchvision.datasets.FashionMNIST(root=config.sensitive_data.test_path, train=False, download=True, transform=transforms.ToTensor())
    # elif config.sensitive_data.name == "cifar10":
    #     sensitive_train_set = torchvision.datasets.CIFAR10(root=config.sensitive_data.train_path, train=True, download=True, transform=transforms.ToTensor())
    #     sensitive_test_set = torchvision.datasets.CIFAR10(root=config.sensitive_data.test_path, train=False, download=True, transform=transforms.ToTensor())
    # elif config.sensitive_data.name == "cifar100":
    #     sensitive_train_set = torchvision.datasets.CIFAR100(root=config.sensitive_data.train_path, train=True, download=True, transform=transforms.ToTensor())
    #     sensitive_test_set = torchvision.datasets.CIFAR100(root=config.sensitive_data.test_path, train=False, download=True, transform=transforms.ToTensor())
    # elif config.sensitive_data.name == "celeba":
    #     sensitive_train_set = ImageFolderDataset(
    #         config.sensitive_data.train_path, config.sensitive_data.resolution, attr='Male', split='train', use_labels=True)
    #     sensitive_test_set = ImageFolderDataset(
    #         config.sensitive_data.test_path, config.sensitive_data.resolution, attr='Male', split='test', use_labels=True)
    # elif config.sensitive_data.name == "camelyon":
    #     sensitive_train_set = ImageFolderDataset(
    #         config.sensitive_data.train_path, config.sensitive_data.resolution, split='train', use_labels=True)
    #     sensitive_test_set = ImageFolderDataset(
    #         config.sensitive_data.test_path, config.sensitive_data.resolution, split='test', use_labels=True)
    # else:
    #     raise NotImplementedError('{} is not yet implemented.'.format(config.sensitive_data.name))
    

    sensitive_train_loader = torch.utils.data.DataLoader(dataset=sensitive_train_set, shuffle=True, drop_last=False, batch_size=config.train.batch_size)
    sensitive_test_loader = torch.utils.data.DataLoader(dataset=sensitive_test_set, shuffle=True, drop_last=False, batch_size=config.eval.batch_size)

    return sensitive_train_loader, sensitive_test_loader


def semantic_query(sensitive_train_loader, config):

    sensitive_loader = torch.utils.data.DataLoader(dataset=sensitive_train_loader.dataset, shuffle=True, drop_last=False, batch_size=config.public_data.selective.batch_size)

    def load_weight(net, weight_path):
        weight = torch.load(weight_path, map_location= 'cuda:%d' % config.setup.local_rank)
        weight = {k.replace('module.', ''): v for k, v in weight.items()}
        net.load_state_dict(weight)

    class MyClassifier(torch.nn.Module):
        def __init__(self):
            super(MyClassifier, self).__init__()
            self.model = resnet.ResNet50(num_classes=1000)

        def forward(self, x):
            return self.model(x)
        
    model = MyClassifier()

    load_weight(model, config.public_data.selective.model_path)
    model = model.to(config.setup.local_rank)
    model.eval()

    semantics_hist = torch.zeros((config.sensitive_data.n_classes, config.public_data.n_classes)).cuda()

    for (x, y) in sensitive_loader:
        if len(y.shape) == 2:
            x = x.to(torch.float32) / 255.
            y = torch.argmax(y, dim=1)
        if x.shape[-1] != 32:
            x = F.interpolate(x, size=[32, 32])
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = x.to(config.setup.local_rank) * 2. - 1.
        y = y.to(config.setup.local_rank).long()
        out = model(x)
        words_idx = torch.topk(out, k=config.public_data.selective.num_words, dim=1)[1]
        for i in range(x.shape[0]):
            cls = y[i]
            words = words_idx[i]
            semantics_hist[cls, words] += 1

    sensitivity = np.sqrt(config.public_data.selective.num_words)
    semantics_hist = semantics_hist + torch.rand_like(semantics_hist) * sensitivity * config.public_data.selective.sigma

    semantics_description = torch.topk(semantics_hist, k=config.public_data.selective.num_words, dim=1)

    cls_dict = {cls: list(semantics_description[1][cls].detach().cpu().numpy()) for cls in range(config.sensitive_data.n_classes)}
    logging.info(cls_dict)
    return cls_dict


def load_data(config):
    sensitive_train_loader, sensitive_test_loader = load_sensitive_data(config)

    if config.public_data.name is None:
        return sensitive_train_loader, sensitive_test_loader, None
    elif config.public_data.name == "imagenet":
        trans = [
                transforms.Resize(config.public_data.resolution),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        if config.public_data.num_channels == 1:
            trans = [transforms.Grayscale(num_output_channels=1)] + trans
        trans = transforms.Compose(trans)
        if config.public_data.selective.ratio == 1.0:
            specific_class = None
        else:
            with torch.no_grad():
                specific_class = semantic_query(sensitive_train_loader, config)
        public_train_set = SpecificClassImagenet(root=config.public_data.train_path, specific_class=specific_class, transform=trans, split="train")
    else:
        raise NotImplementedError('public data {} is not yet implemented.'.format(config.public_data.name))
    
    public_train_loader = torch.utils.data.DataLoader(dataset=public_train_set, shuffle=True, drop_last=False, batch_size=config.pretrain.batch_size)

    return sensitive_train_loader, sensitive_test_loader, public_train_loader