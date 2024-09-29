import torch
import pickle
import numpy as np
from scipy import linalg
import logging
import zipfile
import os


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchvision import datasets, transforms

import logging

from evaluation.ema import ExponentialMovingAverage
from evaluation.classifier.wrn import WideResNet
from evaluation.classifier.resnet import ResNet
from evaluation.classifier.resnext import ResNeXt
from evaluation.classifier.densenet import DenseNet


from models.DP_Diffusion.dnnlib.util import open_url


class Evaluator(object):
    def __init__(self, config):

        self.device = config.setup.local_rank

        self.sensitive_stats_path = config.sensitive_data.fid_stats
        self.acc_models = ["resnet", "wrn", "resnext"]
        self.config = config
        torch.cuda.empty_cache()
    
    def eval(self, synthetic_images, synthetic_labels, sensitive_test_loader):
        if self.device != 0:
            return
        
        # fid = self.cal_fid(synthetic_images)
        fid = 0
        logging.info("The FID of synthetic images is {}".format(fid))

        acc_list = []

        for model_name in self.acc_models:
            acc = self.cal_acc(model_name, synthetic_images, synthetic_labels, sensitive_test_loader)
            logging.info("The best acc of synthetic images from {} is {}".format(model_name, acc))

            acc_list.append(acc)
        
        acc_mean = np.array(acc_list).mean()
        acc_std = np.array(acc_list).std()

        logging.info(f"The best acc of accuracy of synthetic images from resnet, wrn, and resnext are {acc_list}.")

        logging.info(f"The average and std of accuracy of synthetic images are {acc_mean:.2f} and {acc_std:.2f}")


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
    
    def cal_acc(self, model_name, synthetic_images, synthetic_labels, sensitive_test_loader):
        batch_size = 512
        lr = 1e-4
        max_epoch = 50
        num_classes = len(set(synthetic_labels))
        criterion = nn.CrossEntropyLoss()
        if model_name == "wrn":
            model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], num_classes=num_classes, dropRate=0.3)
        elif model_name == "resnet":
            model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2])
        elif model_name == "resnext":
            model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], dropRate=0.3)
        # elif model_name == "densenet":
        #     model = DenseNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], dropRate=0.3)

        model = torch.nn.DataParallel(model).to(self.device)

        ema = ExponentialMovingAverage(model.parameters(), 0.9999)

        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

        train_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images).float(), torch.from_numpy(synthetic_labels).long()), shuffle=True, batch_size=batch_size, num_workers=2)

        best_acc = 0.

        for epoch in range(max_epoch):
            model.train()
            train_loss = 0
            test_loss = 0
            total = 0
            correct = 0
            for _, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                ema.update(model.parameters())
            train_acc = correct / total * 100
            train_loss = train_loss / total
            #scheduler.step()
            model.eval()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            total = 0
            correct = 0
            with torch.no_grad():
                for _, (inputs, targets) in enumerate(sensitive_test_loader):
                    if len(targets.shape) == 2:
                        inputs = inputs.to(torch.float32) / 255.
                        targets = torch.argmax(targets, dim=1)

                    inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            test_acc = correct / total * 100
            test_loss = test_loss / total

            if test_acc >= best_acc:
                best_acc = test_acc

            logging.info("Epoch: {} Train acc: {} Test acc: {} Train loss: {} Test loss: {}".format(epoch, train_acc, test_acc, train_loss, test_loss))
            ema.restore(model.parameters())

        return best_acc
    
    def unzip_file(self, zip_path, extract_to):
        if not os.path.exists(extract_to):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
    
    def cal_acc_no_dp(self):

        # Unzip train and test datasets if not already extracted
        train_zip_path = self.config.sensitive_data.train_path  # This is the zip file path
        test_zip_path = self.config.sensitive_data.test_path  # This is the zip file path
        
        # Set extraction paths
        train_extract_path = train_zip_path.replace(".zip", "")  # Extract to the folder without .zip
        test_extract_path = test_zip_path.replace(".zip", "")
        
        # Unzip the datasets if they are not already extracted
        self.unzip_file(train_zip_path, train_extract_path)
        self.unzip_file(test_zip_path, test_extract_path)

        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Standardization
        ])

        # Load the dataset from the extracted directories
        train_dataset = datasets.ImageFolder(root=train_extract_path, transform=transform)
        test_dataset = datasets.ImageFolder(root=test_extract_path, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Extract training images and labels
        train_images = []
        train_labels = []
        for images, labels in train_loader:
            train_images.append(images.numpy())
            train_labels.append(labels.numpy())
        
        train_images = np.concatenate(train_images, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        acc_list = []

        # Loop through different models and calculate accuracy
        for model_name in self.acc_models:
            acc = self.cal_acc(model_name, train_images, train_labels, test_loader)
            logging.info(f"The best accuracy of model {model_name} on sensitive data is {acc:.2f}%")
            acc_list.append(acc)

        # Compute mean and standard deviation
        acc_mean = np.mean(acc_list)
        acc_std = np.std(acc_list)

        logging.info(f"The best accuracies for resnet, wrn, and resnext are: {acc_list}")
        logging.info(f"The average accuracy is {acc_mean:.2f}% and the standard deviation is {acc_std:.2f}%")

    def train_classifier_no_dp(self, model_name, synthetic_images, synthetic_labels, sensitive_test_loader):
        batch_size = 512
        lr = 1e-4
        max_epoch = 50
        num_classes = len(set(synthetic_labels))
        criterion = nn.CrossEntropyLoss()

        if model_name == "wrn":
            model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], num_classes=num_classes, dropRate=0.3)
        elif model_name == "resnet":
            model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], num_classes=num_classes)
        elif model_name == "resnext":
            model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], num_classes=num_classes, dropRate=0.3)

        model = torch.nn.DataParallel(model).to(self.device)

        ema = ExponentialMovingAverage(model.parameters(), 0.9999)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images).float(), torch.from_numpy(synthetic_labels).long()), shuffle=True, batch_size=batch_size, num_workers=2)

        best_acc = 0.

        for epoch in range(max_epoch):
            model.train()
            total = 0
            correct = 0
            train_loss = 0.0
            for _, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                ema.update(model.parameters())

            train_acc = correct / total * 100
            logging.info(f"Epoch {epoch+1}/{max_epoch}, Train Accuracy: {train_acc:.2f}%")

            model.eval()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            total = 0
            correct = 0
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in sensitive_test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            test_acc = correct / total * 100
            logging.info(f"Test Accuracy: {test_acc:.2f}%")

            if test_acc > best_acc:
                best_acc = test_acc

            ema.restore(model.parameters())

        logging.info(f"The best accuracy for {model_name} is {best_acc:.2f}%")
        return best_acc

    
    