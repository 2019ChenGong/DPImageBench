import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import random_split
from torchvision import transforms
import numpy as np
import logging


from data.stylegan3.dataset import ImageFolderDataset
from data.SpecificImagenet import SpecificClassImagenet
from models.PrivImage import resnet

def load_sensitive_data(config):    
    # sensitive_train_set = ImageFolderDataset(
    #         config.sensitive_data.train_path, config.sensitive_data.resolution, config.sensitive_data.num_channels, use_labels=True)
    
    sensitive_train_set = ImageFolderDataset(
            config.sensitive_data.train_path, config.sensitive_data.resolution, config.sensitive_data.num_channels, use_labels=True)
    sensitive_test_set = ImageFolderDataset(
            config.sensitive_data.test_path, config.sensitive_data.resolution, config.sensitive_data.num_channels, use_labels=True)
    
    if config.sensitive_data.train_num != "all":
        if "mnist" in config.sensitive_data.name:
            train_size = 55000
        elif "cifar" in config.sensitive_data.name:
            train_size = 45000
        elif "eurosat" in config.sensitive_data.name:
            train_size = 21000
        elif "celeba" in config.sensitive_data.name:
            train_size = 162770
        elif "camelyon" in config.sensitive_data.name:
            train_size = 302436
        else:
            raise NotImplementedError

        val_size = len(sensitive_train_set) - train_size
        torch.manual_seed(0)
        sensitive_train_set, sensitive_val_set = random_split(sensitive_train_set, [train_size, val_size])
        sensitive_val_loader = torch.utils.data.DataLoader(dataset=sensitive_val_set, shuffle=False, drop_last=False, batch_size=config.eval.batch_size)
    else:
        sensitive_val_set = None
        sensitive_val_loader = None
    

    sensitive_train_loader = torch.utils.data.DataLoader(dataset=sensitive_train_set, shuffle=True, drop_last=False, batch_size=config.train.batch_size)
    sensitive_test_loader = torch.utils.data.DataLoader(dataset=sensitive_test_set, shuffle=False, drop_last=False, batch_size=config.eval.batch_size)

    return sensitive_train_loader, sensitive_val_loader, sensitive_test_loader


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

    semantics_hist = torch.zeros((config.sensitive_data.n_classes, config.public_data.n_classes)).to(config.setup.local_rank)

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
    torch.manual_seed(0)
    semantics_hist = semantics_hist + torch.rand_like(semantics_hist) * sensitivity * config.public_data.selective.sigma
    
    cls_dict = {}
    for i in range(config.sensitive_data.n_classes):
        semantics_hist_i = semantics_hist[i]
        if i != 0:
            semantics_hist_i[topk_mask] = -999
        semantics_description_i = torch.topk(semantics_hist_i, k=config.public_data.selective.num_words)[1]
        if i == 0:
            topk_mask = semantics_description_i
        else:
            topk_mask = torch.cat([topk_mask, semantics_description_i])
        cls_dict[i] = list(semantics_description_i.detach().cpu().numpy())
            
    if config.setup.global_rank == 0:
        logging.info(cls_dict)
    del model
    torch.cuda.empty_cache()
    return cls_dict


def load_data(config):
    sensitive_train_loader, sensitive_val_loader, sensitive_test_loader = load_sensitive_data(config)
    N = len(sensitive_train_loader.dataset)
    config.train.dp.delta = float(1.0 / (N * np.log(N)))

    if config.setup.global_rank == 0:
        logging.info("delta is reset as {}".format(config.train.dp.delta))

    if config.public_data.name is None:
        public_train_loader = None
    else:
        trans = [
                transforms.Resize(config.public_data.resolution),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        if config.public_data.num_channels == 1:
            trans = [transforms.Grayscale(num_output_channels=1)] + trans
        trans = transforms.Compose(trans)
        if config.public_data.name == "imagenet":
            if config.public_data.selective.ratio == 1.0:
                specific_class = None
            else:
                with torch.no_grad():
                    specific_class = semantic_query(sensitive_train_loader, config)
            public_train_set = SpecificClassImagenet(root=config.public_data.train_path, specific_class=specific_class, transform=trans, split="train")
        elif config.public_data.name == "places365":
            # TODO: we only have a classifier on imagenet. Besides, we cannot train the classifer on LAION.
            public_train_set = torchvision.datasets.Places365(root=config.public_data.train_path, small=True, download=True, transform=trans)
        else:
            raise NotImplementedError('public data {} is not yet implemented.'.format(config.public_data.name))
    
        public_train_loader = torch.utils.data.DataLoader(dataset=public_train_set, shuffle=True, drop_last=True, batch_size=config.pretrain.batch_size)

    if config.sensitive_data.name is None:
        sensitive_train_loader = None
        sensitive_val_loader = None
        sensitive_test_loader = None

    return sensitive_train_loader, sensitive_val_loader, sensitive_test_loader, public_train_loader, config