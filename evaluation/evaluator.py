import torch
import pickle
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

        if self.config.setup.method in ["DP-MERF", "DP-NTK"]:
            return mlp_acc(synthetic_images, synthetic_labels, sensitive_test_loader)
        batch_size = 256
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
    
    
    def cal_acc_no_dp(self, sensitive_train_loader, sensitive_test_loader):
        if self.device != 0 or sensitive_test_loader is None or sensitive_train_loader is None:
            return
        batch_size = 256
        lr = 1e-4
        max_epoch = 50
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(sensitive_train_loader.dataset, batch_size=batch_size, shuffle=True)
        test_loader = sensitive_test_loader

        num_classes = self.config.sensitive_data.n_classes
        c = self.config.sensitive_data.num_channels
        img_size = self.config.sensitive_data.resolution

        acc_list = []

        for model_name in self.acc_models:

            if model_name == "wrn":
                model = WideResNet(in_c=c, img_size=img_size, num_classes=num_classes, dropRate=0.3)
            elif model_name == "resnet":
                model = ResNet(in_c=c, img_size=img_size, num_classes=num_classes)
            elif model_name == "resnext":
                model = ResNeXt(in_c=c, img_size=img_size, num_classes=num_classes, dropRate=0.3)

            criterion = nn.CrossEntropyLoss()

            # Move model to device (GPU/CPU)
            model = model.to(self.device)

            # Define the loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

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


def mlp_acc(synthetic_images, synthetic_labels, sensitive_test_loader):
    x_tr = synthetic_images.reshape(synthetic_images.shape[0], -1)
    y_tr = synthetic_labels.reshape(-1)
    x_ts = []
    y_ts = []
    for x, y in sensitive_test_loader:
        x = x.float() / 255.
        y = torch.argmax(y, dim=1)
        x_ts.append(x)
        y_ts.append(y)
    x_ts = torch.cat(x_ts).cpu().numpy()
    x_ts = x_ts.reshape(x_ts.shape[0], -1)
    y_ts = torch.cat(y_ts).cpu().numpy().reshape(-1)

    model = neural_network.MLPClassifier()
    model.fit(x_tr, y_tr)
    y_pred = model.predict(x_ts)
    acc = accuracy_score(y_pred, y_ts)

    return acc


def dpkernel_acc(synthetic_images, synthetic_labels, sensitive_test_loader):
    def NNfit(model, train_loader, lr=0.01, num_epochs=5, data_mode='real'):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        loss_func = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(num_epochs):
            correct = 0
            total = 0
            for batch_idx, data_true in enumerate(train_loader):
                X, y_true = data_true
                X = X.view(X.size(0), -1)
                optimizer.zero_grad()
                y_pred = model(X)
                loss = loss_func(y_pred, y_true.to(torch.long))
                loss.backward()
                optimizer.step()
                # Total correct predictions
                predicted = torch.max(y_pred.data, 1)[1]
                correct += (predicted == y_true.to(torch.long)).sum()
                total += X.size(0)
                # print(correct)
            print('Epoch : {} \tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                epoch, loss.item(), float(correct * 100) / total)
            )
        return model

    def evalNN(model, test_loader, data_mode='real', dataset='mnist'):
        model.eval()
        correct = 0
        total = 0

        for batch_idx, data in enumerate(test_loader):
            X, y_true = data
            X = X.view(X.size(0), -1)
            X = X.float() / 255.
            y_true = torch.argmax(y_true, dim=1)

            y_pred = model(X)
            # Total correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            correct += (predicted == y_true.to(torch.long)).sum()
            total += X.size(0)

        return float(correct * 100) / total

    class NNClassifier(nn.Module):
        def __init__(self, input_shape, type='MLP', data_mode='real', n_class=10):
            super(NNClassifier, self).__init__()

            self.input_shape = input_shape
            self.input_channel, h, w = input_shape
            self.input_dim = self.input_channel * h * w
            self.type = type
            self.data_mode = data_mode
            self.output_dim = n_class

            self.MLP = nn.Sequential(
                nn.Linear(self.input_dim, 100),
                nn.ReLU(),
                nn.Linear(100, self.output_dim),
                nn.Softmax()
            )
            self.flatten_dim = 64 * int(h / 4) * int(w / 4)
            self.conv = nn.Sequential(
                nn.Conv2d(self.input_channel, 32, kernel_size=3, stride = 2, padding=1),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride = 2, padding=1),
                nn.Dropout(p=0.5),
                nn.ReLU(),
            )

            self.linear = nn.Sequential(
                nn.Linear(self.flatten_dim, self.output_dim),
                nn.Softmax()
            )

        def forward(self, x):
            if self.type == 'MLP':
                if self.data_mode == 'real':
                    x = x.view(-1, self.input_dim)
                x = self.MLP(x)
            elif self.type == 'CNN':
                if self.data_mode == 'syn':
                    x = x.view(-1, *self.input_shape)
                x = self.conv(x)
                x = x.view(-1, self.flatten_dim)
                x = self.linear(x)
            return x
    
    num_classes = len(set(synthetic_labels))
    train_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images).float(), torch.from_numpy(synthetic_labels).long()), shuffle=True, batch_size=200, num_workers=2)
    model = NNClassifier((synthetic_images.shape[1], synthetic_images.shape[2], synthetic_images.shape[3]), type='CNN', data_mode='syn', n_class=num_classes)
    model = NNfit(model, train_loader, lr=0.001, num_epochs=5, data_mode='syn')
    return evalNN(model, sensitive_test_loader, data_mode='syn')