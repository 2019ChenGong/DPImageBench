import torch
import pickle
import numpy as np
from scipy import linalg
import logging


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import logging
import os

from sklearn import linear_model, neural_network
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from evaluation.ema import ExponentialMovingAverage
from evaluation.classifier.wrn import WideResNet


from models.DP_Diffusion.dnnlib.util import open_url


class Evaluator(object):
    def __init__(self, config):

        self.device = config.setup.local_rank

        self.sensitive_stats_path = config.sensitive_data.fid_stats
        self.config = config
    
    def eval(self, synthetic_images, synthetic_labels, sensitive_test_loader):
        if self.device != 0:
            return
        
        fid = self.cal_fid(synthetic_images)

        acc = self.cal_acc(synthetic_images, synthetic_labels, sensitive_test_loader)
        logging.info("The FID of synthetic images is {}".format(fid))
        logging.info("The best acc of synthetic images is {}".format(acc))
        print("The FID of synthetic images is {}".format(fid))

    def cal_fid(self, synthetic_images, batch_size=500):
        with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
            inception_model = pickle.load(f).to(self.device)

        act = []
        chunks = torch.chunk(torch.from_numpy(synthetic_images), len(synthetic_images) // batch_size)
        print('Starting to sample.')
        for batch in chunks:
            batch = (batch * 255.).to(torch.uint8)

            batch = batch.to(self.device)
            if batch.shape[1] == 1:  # if image is gray scale
                batch = batch.repeat(1, 3, 1, 1)
            elif len(batch.shape) == 3:  # if image is gray scale
                batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)

            with torch.no_grad():
                pred = inception_model(batch.to(self.device), return_features=True).unsqueeze(-1).unsqueeze(-1)

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            act.append(pred)

        act = np.concatenate(act, axis=0)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)

        stats = np.load(self.sensitive_stats_path)
        data_pools_mean = stats['mu']
        data_pools_sigma = stats['sigma']

        m = np.square(mu - data_pools_mean).sum()
        s, _ = linalg.sqrtm(np.dot(sigma, data_pools_sigma), disp=False)
        fd = np.real(m + np.trace(sigma + data_pools_sigma - s * 2))

        return fd
    
    def cal_acc(self, synthetic_images, synthetic_labels, sensitive_test_loader):
        batch_size = 256
        lr = 1e-4
        max_epoch = 50
        num_classes = len(set(synthetic_labels))
        criterion = nn.CrossEntropyLoss()
        model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], num_classes=num_classes).to(self.device)
        ema = ExponentialMovingAverage(model.parameters(), 0.9999)

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

        train_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images).float(), torch.from_numpy(synthetic_labels).long()), shuffle=True, batch_size=batch_size, num_workers=2)

        best_acc = 0.

        for epoch in range(max_epoch):
            model.train()
            total = 0
            correct = 0
            for _, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                ema.update(model.parameters())
            train_acc = correct / total * 100
            #scheduler.step()
            model.eval()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            total = 0
            correct = 0
            with torch.no_grad():
                for _, (inputs, targets) in enumerate(sensitive_test_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            test_acc = correct / total * 100

            if test_acc >= best_acc:
                best_acc = test_acc

            logging.info("Epoch: {} Train acc: {} Test acc: {}".format(epoch, train_acc, test_acc))
            ema.restore(model.parameters())
        return best_acc
    
    