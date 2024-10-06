import torch
import pickle
import copy
import numpy as np
from scipy import linalg
from sklearn import linear_model, ensemble, naive_bayes, svm, tree, discriminant_analysis, neural_network
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import logging
import zipfile
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchvision import datasets, transforms

import logging

from evaluation.ema import ExponentialMovingAverage
from evaluation.classifier.wrn import WideResNet
from evaluation.classifier.resnet import ResNet
from evaluation.classifier.resnext import ResNeXt
from evaluation.classifier.densenet import DenseNet
from data.preprocess_dataset import target_trans


from models.DP_Diffusion.dnnlib.util import open_url


class Evaluator(object):
    def __init__(self, config):

        self.device = config.setup.local_rank

        self.sensitive_stats_path = config.sensitive_data.fid_stats
        self.acc_models = ["resnet", "wrn", "resnext"]
        self.config = config
        torch.cuda.empty_cache()
    
    def eval(self, synthetic_images, synthetic_labels, sensitive_test_loader):
        if self.device != 0 or sensitive_test_loader is None:
            return
        
        # fid = self.cal_fid(synthetic_images)
        fid = 0
        logging.info("The FID of synthetic images is {}".format(fid))

        acc_list = []

        for model_name in self.acc_models:
            acc, test_acc = self.cal_acc_2(model_name, synthetic_images, synthetic_labels, sensitive_test_loader)
            logging.info("The best acc of synthetic images on val and test dataset from {} is {} and {}".format(model_name, acc, test_acc))

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
    
        num_classes = len(set(synthetic_labels))
        criterion = nn.CrossEntropyLoss()
        lr = 1e-4

        if self.config['sensitive_data']['name'] == 'cifar10_32' or self.config['sensitive_data']['name'] == 'cifar100_32':
            batch_size = 128
            max_epoch = 200
            if model_name == "wrn":
                model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], num_classes=num_classes, depth=28, widen_factor=10, dropRate=0.3)
            elif model_name == "resnet":
                model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], num_classes=num_classes, depth=164, block_name='BasicBlock')
            elif model_name == "resnext":
                model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], cardinality=8, depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)

            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
        else:
            batch_size = 256
            max_epoch = 50
            if model_name == "wrn":
                model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], num_classes=num_classes, dropRate=0.3)
            elif model_name == "resnet":
                model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], num_classes=num_classes)
            elif model_name == "resnext":
                model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], num_classes=num_classes, dropRate=0.3)

            # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) 
            optimizer = optim.Adam(model.parameters(), lr=0.01)       
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=99999, gamma=0.2)

        model = torch.nn.DataParallel(model).to(self.device)

        ema = ExponentialMovingAverage(model.parameters(), 0.9999)

        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
        # optimizer = optim.Adam(model.parameters(), lr=0.01)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

        # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)

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

            scheduler.step()

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
    
    def cal_acc_2(self, model_name, synthetic_images, synthetic_labels, sensitive_test_loader):

        synthetic_images_train, synthetic_images_val = synthetic_images[:55000], synthetic_images[55000:]
        synthetic_labels_train, synthetic_labels_val = synthetic_labels[:55000], synthetic_labels[55000:]
    
        num_classes = len(set(synthetic_labels))
        criterion = nn.CrossEntropyLoss()
        lr = 1e-4

        if self.config['sensitive_data']['name'] == 'cifar10' or self.config['sensitive_data']['name'] == 'cifar100':
            batch_size = 126
            max_epoch = 200
            if model_name == "wrn":
                model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], num_classes=num_classes, depth=28, widen_factor=10, dropRate=0.3)
            elif model_name == "resnet":
                model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], num_classes=num_classes, depth=164, block_name='BasicBlock')
            elif model_name == "resnext":
                model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], cardinality=8, depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)

            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
        else:
            batch_size = 256
            max_epoch = 50
            if model_name == "wrn":
                model = WideResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], num_classes=num_classes, dropRate=0.3)
            elif model_name == "resnet":
                model = ResNet(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], num_classes=num_classes)
            elif model_name == "resnext":
                model = ResNeXt(in_c=synthetic_images.shape[1], img_size=synthetic_images.shape[2], num_classes=num_classes, dropRate=0.3)

            # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4) 
            optimizer = optim.Adam(model.parameters(), lr=0.01)       
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=99999, gamma=0.2)

        model = torch.nn.DataParallel(model).to(self.device)

        ema = ExponentialMovingAverage(model.parameters(), 0.9999)

        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
        # optimizer = optim.Adam(model.parameters(), lr=0.01)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

        # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)

        train_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images_train).float(), torch.from_numpy(synthetic_labels_train).long()), shuffle=True, batch_size=batch_size, num_workers=2)
        val_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images_val).float(), torch.from_numpy(synthetic_labels_val).long()), shuffle=True, batch_size=batch_size, num_workers=2)

        best_acc = 0.
        best_test_acc = 0.

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

            scheduler.step()

            train_acc = correct / total * 100
            train_loss = train_loss / total
            
            model.eval()
            ema.store(model.parameters())
            ema.copy_to(model.parameters())

            val_total = 0
            val_correct = 0
            
            with torch.no_grad():
                for _, (inputs, targets) in enumerate(val_loader):

                    inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            val_acc = val_correct / val_total * 100
            val_loss = test_loss / val_total


            test_total = 0
            test_correct = 0
        
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
                    test_correct += predicted.eq(targets).sum().item()
            
            test_acc = test_correct / test_total * 100

            logging.info("Epoch: {} Train acc: {} Val acc: {} Test acc{}; Train loss: {} Val loss: {}".format(epoch, train_acc, val_acc, test_acc, train_loss, val_loss))

            if val_acc >= best_acc:
                best_acc = val_acc
                best_test_acc = test_acc
                best_model = copy.deepcopy(model)

            # logging.info("Epoch: {} Train acc: {} Val acc: {} Train loss: {} Val loss: {}".format(epoch, train_acc, val_acc, train_loss, val_loss))
            ema.restore(model.parameters())
        

        return best_acc, best_test_acc
    
    def cal_acc_no_dp(self, sensitive_train_loader, sensitive_test_loader):
        if self.device != 0 or sensitive_test_loader is None or sensitive_train_loader is None:
            return
        
        batch_size = 128
        lr = 1e-4
        max_epoch = 200


        train_loader = DataLoader(sensitive_train_loader.dataset, batch_size=batch_size, shuffle=True)
        test_loader = sensitive_test_loader

        num_classes = self.config.sensitive_data.n_classes
        c = self.config.sensitive_data.num_channels
        img_size = self.config.sensitive_data.resolution

        acc_list = []

        for model_name in self.acc_models:

            logging.info(f'model type:{model_name}')

            if self.config['sensitive_data']['name'] == 'cifar10' or self.config['sensitive_data']['name'] == 'cifar100':

                if model_name == "wrn":
                    model = WideResNet(in_c=c, img_size=img_size, num_classes=num_classes, depth=28, widen_factor=10, dropRate=0.3)
                elif model_name == "resnet":
                    model = ResNet(in_c=c, img_size=img_size, num_classes=num_classes, depth=164, block_name='BasicBlock')
                elif model_name == "resnext":
                    model = ResNeXt(in_c=c, img_size=img_size, cardinality=8, depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)

                optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
            else:
                if model_name == "wrn":
                    model = WideResNet(in_c=c, img_size=img_size, num_classes=num_classes,  dropRate=0.3)
                elif model_name == "resnet":
                    model = ResNet(in_c=c, img_size=img_size, num_classes=num_classes,  block_name='BasicBlock')
                elif model_name == "resnext":
                    model = ResNeXt(in_c=c, img_size=img_size, num_classes=num_classes, dropRate=0.3)

                
                optimizer = optim.Adam(model.parameters(), lr=0.01)
                # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)        
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)


            # Move model to device (GPU/CPU)
            model = model.to(self.device)

            # Define the loss function and optimizer
            criterion = nn.CrossEntropyLoss()

            # optimizer = optim.Adam(model.parameters(), lr=0.01)

            # Training the model
            best_acc = 0.0
            for epoch in range(max_epoch):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    inputs = inputs.float() / 255. * 2. - 1.
                    labels = torch.argmax(labels, dim=1)
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    # Calculate statistics
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                scheduler.step()

                train_acc = 100. * correct / total
                logging.info(f"Epoch [{epoch+1}/{max_epoch}], Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%")

                # Testing the model
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        inputs = inputs.float() / 255. * 2. - 1.
                        labels = torch.argmax(labels, dim=1)
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

                test_acc = 100. * correct / total
                logging.info(f"Test Accuracy: {test_acc:.2f}%")

                # Return the best accuracy
                if test_acc > best_acc:
                    best_acc = test_acc

            logging.info("The best acc of original images from {} is {}".format(model_name, best_acc))

            acc_list.append(best_acc)

        acc_mean = np.array(acc_list).mean()
        acc_std = np.array(acc_list).std()

        logging.info(f"The best acc of accuracy of synthetic images from resnet, wrn, and resnext are {acc_list}.")

        logging.info(f"The average and std of accuracy of synthetic images are {acc_mean:.2f} and {acc_std:.2f}")

