import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import random_split, Dataset, DataLoader
from opacus.data_loader import DPDataLoader, switch_generator
from torchvision import transforms
import numpy as np
import logging
from PIL import Image
import cv2


from data.stylegan3.dataset import ImageFolderDataset
from data.SpecificImagenet import SpecificClassImagenet
from data.SpecificPlaces365 import SpecificClassPlaces365
from data.SpecificEMNIST import SpecificClassEMNIST
from models.PrivImage import resnet
from models.PrivImage.classifer_trainer import train_classifier

import random
class random_aug(object):
    def __init__(self, magnitude, num_ops):
        self.mag = magnitude
        self.no = num_ops
    def __call__(self, img):
        mag = random.choice([i for i in range(1, self.mag+1)])
        return transforms.RandAugment(num_ops=self.no, magnitude=mag)(img)

    def __repr__(self):
        return self.__class__.__name__
                
def load_sensitive_data(config):    
    sensitive_train_set = ImageFolderDataset(
            config.sensitive_data.train_path, config.sensitive_data.resolution, config.sensitive_data.num_channels, use_labels=True)
    sensitive_test_set = ImageFolderDataset(
            config.sensitive_data.test_path, config.sensitive_data.resolution, config.sensitive_data.num_channels, use_labels=True)
    
    if config.eval.mode == "val":
        # split the sensitive dataset into training set and validation set
        if "mnist" in config.sensitive_data.name:
            train_size = 55000
        elif "cifar" in config.sensitive_data.name:
            train_size = 45000
        elif "eurosat" in config.sensitive_data.name:
            train_size = 21000
        elif "celeba" in config.sensitive_data.name:
            train_size = 145064
        elif "camelyon" in config.sensitive_data.name:
            train_size = 269538
        else:
            raise NotImplementedError

        val_size = len(sensitive_train_set) - train_size
        torch.manual_seed(0)
        sensitive_train_set, sensitive_val_set = random_split(sensitive_train_set, [train_size, val_size])
        sensitive_val_loader = torch.utils.data.DataLoader(dataset=sensitive_val_set, shuffle=False, drop_last=False, batch_size=config.eval.batch_size)
        logging.info("train size: {} val size: {}".format(len(sensitive_train_set), len(sensitive_val_set)))
    else:
        sensitive_val_set = None
        sensitive_val_loader = None
    

    sensitive_train_loader = torch.utils.data.DataLoader(dataset=sensitive_train_set, shuffle=True, drop_last=False, batch_size=config.train.batch_size)
    sensitive_test_loader = torch.utils.data.DataLoader(dataset=sensitive_test_set, shuffle=False, drop_last=False, batch_size=config.eval.batch_size)

    return sensitive_train_loader, sensitive_val_loader, sensitive_test_loader


def semantic_query(sensitive_train_loader, config):

    try:
        batch_size = config.public_data.selective.batch_size
        sigma = config.public_data.selective.sigma
    except:
        batch_size = 1000
        sigma = 50

    sensitive_loader = torch.utils.data.DataLoader(dataset=sensitive_train_loader.dataset, shuffle=True, drop_last=False, batch_size=batch_size)

    def load_weight(net, weight_path):
        weight = torch.load(weight_path, map_location= 'cuda:%d' % config.setup.local_rank)
        weight = {k.replace('module.', ''): v for k, v in weight.items()}
        net.load_state_dict(weight)

    class MyClassifier(torch.nn.Module):
        def __init__(self):
            super(MyClassifier, self).__init__()
            self.model = resnet.ResNet50(num_classes=config.public_data.n_classes)

        def forward(self, x):
            return self.model(x)
    
    # load semantic query function
    model = MyClassifier()
    model = model.to(config.setup.local_rank)

    if os.path.exists('models/pretrained_models/{}_classifier_ckpt.pth'.format(config.public_data.name)):
        model_path = 'models/pretrained_models/{}_classifier_ckpt.pth'.format(config.public_data.name)
    else:
        model_path = train_classifier(model, config)
    load_weight(model, model_path)
    model.eval()

    # query semantic distribution
    semantics_hist = torch.zeros((config.sensitive_data.n_classes, config.public_data.n_classes)).to(config.setup.local_rank)

    num_words = int(config.public_data.n_classes * config.public_data.selective.ratio / config.sensitive_data.n_classes)
    config.train.dp['privacy_history'] = [[config.public_data.selective.sigma, 1.0, 1]]
    if config.setup.global_rank == 0:
        logging.info("Additional privacy cost: {}".format(str([config.public_data.selective.sigma, 1.0, 1])))

    with torch.no_grad():
        for (x, y) in sensitive_loader:
            if x.shape[-1] != 32:
                x = F.interpolate(x, size=[32, 32])
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            x = x.to(config.setup.local_rank) * 2. - 1.
            y = y.to(config.setup.local_rank).long()
            out = model(x)
            words_idx = torch.topk(out, k=num_words, dim=1)[1]
            for i in range(x.shape[0]):
                cls = y[i]
                words = words_idx[i]
                semantics_hist[cls, words] += 1

    sensitivity = np.sqrt(num_words)
    torch.manual_seed(0)
    semantics_hist = semantics_hist + torch.randn_like(semantics_hist) * sensitivity * sigma
    
    cls_dict = {}
    for i in range(config.sensitive_data.n_classes):
        semantics_hist_i = semantics_hist[i]
        if i != 0:
            semantics_hist_i[topk_mask] = -999
        semantics_description_i = torch.topk(semantics_hist_i, k=num_words)[1]
        if i == 0:
            topk_mask = semantics_description_i
        else:
            topk_mask = torch.cat([topk_mask, semantics_description_i])
        cls_dict[i] = list(semantics_description_i.detach().cpu().numpy())
            
    del model
    torch.cuda.empty_cache()
    return cls_dict, config

class CentralDataset(Dataset):
    def __init__(self, sensitive_dataset, sample_num=50, sigma=5, batch_size=6000, num_classes=10, c_type='mean'):
        super().__init__()

        self.trans = random_aug(magnitude=9, num_ops=2)
        self.privacy_history = [sigma, batch_size/len(sensitive_dataset), sample_num]

        if c_type == 'mean':
            self.central_x, self.central_y = self.query_mean_image(sensitive_dataset, sample_num, sigma, batch_size, num_classes)
        else:
            self.central_x, self.central_y = self.query_mode_image(sensitive_dataset, sample_num, sigma, batch_size, num_classes)
    
    def query_mean_image(self, sensitive_dataset, sample_num, sigma, batch_size, num_classes):
        c = 0
        central_x, central_y = [], []
        ds = 0.5
        dataloader = DataLoader(sensitive_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        dataloader = DPDataLoader.from_data_loader(dataloader)
        for _ in range(10000):
            for x, y in dataloader:
                for cls in range(num_classes):
                    
                    x_cls = x if num_classes==1 else x[y==cls]
                    # logging.info('{} {} {}'.format(c, cls, x_cls.shape[0]))

                    sensitivity_m = np.sqrt((ds**2) * np.prod(x_cls.shape[1:])) / (batch_size//num_classes)

                    xm = torch.mean(x_cls, dim=0, keepdim=True)
                    xm = F.interpolate(xm, scale_factor=ds)
                    xm = xm + torch.randn_like(xm) * sensitivity_m * sigma
                    xm = xm.clamp(0., 1.)
                    xm = F.interpolate(xm, size=x.shape[-2:], mode="bilinear")

                    xm = (xm.cpu() * 255.).to(torch.uint8)

                    xm = self.post_process(xm)

                    central_x.append(xm)
                    central_y.append(cls)
                c += 1
                if c == sample_num:
                    break
            if c == sample_num:
                break
        
        return torch.cat(central_x), central_y

    def query_mode_image(self, sensitive_dataset, sample_num, sigma, batch_size, num_classes):
        c = 0
        central_x, central_y = [], []
        ds = 0.5
        dataloader = DataLoader(sensitive_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
        dataloader = DPDataLoader.from_data_loader(dataloader, generator=self.secure_rng, distributed=True)
        for _ in range(10000):
            for x, y in dataloader:
                if x.shape[-1] == 28:
                    bins = 2
                else:
                    bins = 16
                for cls in range(num_classes):
                    
                    x_cls = x if num_classes==1 else x[y==cls]

                    sensitivity = np.sqrt((ds**2) * np.prod(x_cls.shape[1:]))

                    x_cls = F.interpolate(x_cls, scale_factor=ds)
                    x_cls_shape = x_cls.shape[-3:]
                    x_cls = x_cls.view(x_cls.shape[0], -1)
                    x_mode_noisy = []
                    for i in range(x_cls.shape[1]):
                        hist = torch.histc(x_cls[:, i], bins=bins, min=0, max=1)
                        hist = hist + torch.randn_like(hist) * sensitivity * sigma
                        x_mode_noisy.append(torch.argmax(hist))
                    x_mode_noisy = torch.tensor(x_mode_noisy).view(1, *x_cls_shape).float() / (bins-1)

                    x_mode_noisy = F.interpolate(x_mode_noisy.float(), size=x.shape[-2:], mode="bilinear")

                    x_mode_noisy = (x_mode_noisy.cpu()*255).to(torch.uint8)

                    x_mode_noisy = self.post_process(x_mode_noisy)

                    central_x.append(x_mode_noisy)
                    central_y.append(cls)

                c += 1
                if c == sample_num:
                    break
            if c == sample_num:
                break
        return torch.cat(central_x), central_y

    def post_process(self, x):
        is_gray = x.shape[1] == 1
        if is_gray:
            x = x[0][0].numpy()
        else:
            x = x[0].permute(1, 2, 0).numpy()
        if not is_gray:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.medianBlur(x, 3)
        if is_gray:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            x = cv2.filter2D(x, -1, kernel)
        if not is_gray:
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        x = torch.tensor(x)
        if is_gray:
            x = x.unsqueeze(-1)
        x = x.permute(2, 0, 1).unsqueeze(0)
        return x


    def __len__(self,):
        return len(self.central_x)

    def __getitem__(self, index):
        x, y = self.central_x[index:index+1], self.central_y[index]

        return self.trans(x)[0].float() / 255., y



def load_data(config):
    # load sensitive dataset
    sensitive_train_loader, sensitive_val_loader, sensitive_test_loader = load_sensitive_data(config)
    N = len(sensitive_train_loader.dataset)
    config.train.dp.delta = float(1.0 / (N * np.log(N)))

    if config.setup.global_rank == 0:
        logging.info("delta is reset as {}".format(config.train.dp.delta))

    # load public dataset
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

        # whether to select the public data
        if config.public_data.selective.ratio == 1.0:
            specific_class = None
        else:
            try:
                specific_class = torch.load(config.public_data.selective.semantic_path)
                logging.info(specific_class)
            except:
                specific_class, config = semantic_query(sensitive_train_loader, config)
        if config.public_data.name == "imagenet":
            public_train_set = SpecificClassImagenet(root=config.public_data.train_path, specific_class=specific_class, transform=trans, split="train")
        elif config.public_data.name == "places365":
            download = (not os.path.exists(os.path.join(config.public_data.train_path, "data_256_standard")))
            public_train_set_ = torchvision.datasets.Places365(root=config.public_data.train_path, small=True, download=download, transform=trans)
            if specific_class is None:
                public_train_set = public_train_set_
            else:
                public_train_set = SpecificClassPlaces365(public_train_set_, specific_class)
        elif config.public_data.name == "emnist":
            public_train_set_ = torchvision.datasets.EMNIST(root=config.public_data.train_path, split="letters", train=True)
            if specific_class is None:
                public_train_set = public_train_set_
            else:
                public_train_set = SpecificClassEMNIST(public_train_set_, specific_class)
        elif "central" in config.public_data.name:
            public_train_set = CentralDataset(sensitive_train_loader.dataset, num_classes=config.sensitive_data.n_classes, c_type=config.public_data.name.split('_')[-1], **config.public_data.central)
            config.train.dp['privacy_history'] = [public_train_set.privacy_history]
            if config.setup.global_rank == 0:
                logging.info("Additional privacy cost: {}".format(str(public_train_set.privacy_history)))
        else:
            raise NotImplementedError('public data {} is not yet implemented.'.format(config.public_data.name))
    
        public_train_loader = torch.utils.data.DataLoader(dataset=public_train_set, shuffle=True, drop_last=True, batch_size=config.pretrain.batch_size, num_workers=16)

    if config.sensitive_data.name is None:
        sensitive_train_loader = None
        sensitive_val_loader = None
        sensitive_test_loader = None

    return sensitive_train_loader, sensitive_val_loader, sensitive_test_loader, public_train_loader, config