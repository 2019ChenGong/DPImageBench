import torch
from PIL import Image
import copy
import numpy as np
from scipy import linalg
from scipy.stats import entropy
import logging

from pytorch_fid.inception import fid_inception_v3
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.metrics.FLD import FLD
from fld.metrics.PrecisionRecall import PrecisionRecall
from fld.metrics.FID import FID
import ImageReward as RM


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import logging

from evaluation.ema import ExponentialMovingAverage
from evaluation.classifier.wrn import WideResNet
from evaluation.classifier.resnet import ResNet
from evaluation.classifier.resnext import ResNeXt

class Evaluator(object):
    def __init__(self, config):

        self.device = config.setup.local_rank

        self.sensitive_stats_path = config.sensitive_data.fid_stats
        self.acc_models = ["resnet", "wrn", "resnext"]
        self.config = config
        torch.cuda.empty_cache()
    
    def eval(self, synthetic_images, synthetic_labels, sensitive_train_loader, sensitive_val_loader, sensitive_test_loader):
        if self.device != 0 or sensitive_test_loader is None:
            return
        if synthetic_images.shape[-1] != self.config.sensitive_data.resolution:
            synthetic_images = F.interpolate(torch.from_numpy(synthetic_images), size=[self.config.sensitive_data.resolution, self.config.sensitive_data.resolution]).numpy()
        if synthetic_images.shape[1] == 3 and self.config.sensitive_data.num_channels == 1:
            synthetic_images = 0.299 * synthetic_images[:, 2:, ...] + 0.587 * synthetic_images[:, 1:2, ...] + 0.114 * synthetic_images[:, :1, ...]
        # fid, is_mean = self.visual_metric(synthetic_images)
        # fid, is_mean, fld, p, r, ir = self.visual_metric(synthetic_images, synthetic_labels, sensitive_train_loader, sensitive_test_loader)
        # # fid = is_mean = 0
        # logging.info("The FID of synthetic images is {}".format(fid))
        # logging.info("The Inception Score of synthetic images is {}".format(is_mean))
        # logging.info("The Precision and Recall of synthetic images is {} and {}".format(p, r))
        # logging.info("The FLD of synthetic images is {}".format(fld))
        # logging.info("The ImageReward of synthetic images is {}".format(ir))

        acc_list = []

        for model_name in self.acc_models:
            acc, test_acc_on_val, test_acc_on_test, best_noisy_acc, best_test_acc_on_noisy_val = self.cal_acc_2(model_name, synthetic_images, synthetic_labels, sensitive_val_loader, sensitive_test_loader)
            if sensitive_val_loader is not None:
                logging.info("The best acc of synthetic images on sensitive val and the corresponding acc on test dataset from {} is {} and {}".format(model_name, acc, test_acc_on_val))
                logging.info("The best acc of synthetic images on noisy sensitive val and the corresponding acc on test dataset from {} is {} and {}".format(model_name, acc, test_acc_on_val))
            else:
                logging.info("The best acc of synthetic images on val and the corresponding acc on test dataset from {} is {} and {}".format(model_name, acc, test_acc_on_val))
            logging.info("The best acc test dataset from {} is {}".format(model_name, test_acc_on_test))
            acc_list.append(test_acc_on_val)
        
        acc_mean = np.array(acc_list).mean()
        acc_std = np.array(acc_list).std()

        logging.info(f"The best acc of accuracy (using synthetic images as the validation set) of synthetic images from resnet, wrn, and resnext are {acc_list}.")

        logging.info(f"The average and std of accuracy of synthetic images are {acc_mean:.2f} and {acc_std:.2f}")
    
    def visual_metric(self, synthetic_images, synthetic_labels, sensitive_train_loader, sensitive_test_loader):
        feature_extractor = InceptionFeatureExtractor(save_path="dataset/{}_{}/".format(self.config.sensitive_data.name, self.config.sensitive_data.resolution))
        fc_layer = fid_inception_v3().fc
        fc_layer.eval()

        gen_images = torch.from_numpy(synthetic_images)
        if gen_images.shape[1] == 1:
            gen_images = gen_images.repeat(1, 3, 1, 1)
        gen_feat = feature_extractor.get_tensor_features(gen_images)
        with torch.no_grad():
            gen_logit = fc_layer(gen_feat).detach().cpu().numpy()
            gen_logit = np.exp(gen_logit) / np.sum(np.exp(gen_logit), 1, keepdims=True)

        try:
            train_feat = feature_extractor.get_tensor_features(torch.tensor([0]), name="train")
            test_feat = feature_extractor.get_tensor_features(torch.tensor([0]), name="test")
        except:
            train_images = []
            test_images = []
            for x, _ in sensitive_train_loader:
                train_images.append(x.float()/255.)
            for x, _ in sensitive_test_loader:
                test_images.append(x.float()/255.)
            train_images = torch.cat(train_images)
            test_images = torch.cat(test_images)
            if train_images.shape[1] == 1:
                train_images = train_images.repeat(1, 3, 1, 1)
                test_images = test_images.repeat(1, 3, 1, 1)
            
            train_feat = feature_extractor.get_tensor_features(train_images, name="train")
            test_feat = feature_extractor.get_tensor_features(test_images, name="test")
        # print(train_feat.shape, test_feat.shape, gen_feat.shape)

        num_classes = len(set(synthetic_labels))
        rm_model = RM.load("ImageReward-v1.0")
        ir = 0
        prompt = get_prompt(self.config.sensitive_data.name)
        with torch.no_grad():
            for cls in range(num_classes):
                imgs = (synthetic_images[synthetic_labels==cls] * 255.).astype('uint8')
                imgs = np.transpose(imgs, (0, 2, 3, 1))
                if imgs.shape[-1] == 1:
                    imgs = np.repeat(imgs, 3, axis=-1)
                imgs = [Image.fromarray(img) for img in imgs]
                score = rm_model.score(prompt[cls], imgs)
                ir += np.sum(score)
        ir /= len(synthetic_images)

        is_mean, _ = compute_inception_score_from_logits(gen_logit)
        fid = FID().compute_metric(train_feat, None, gen_feat)
        fld = FLD(eval_feat="train").compute_metric(train_feat, test_feat, gen_feat)
        p = PrecisionRecall(mode="Precision").compute_metric(train_feat, None, gen_feat) # Default precision
        r = PrecisionRecall(mode="Recall", num_neighbors=5).compute_metric(train_feat, None, gen_feat)

        return fid, is_mean, fld, p, r, ir
    
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
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

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
    
    def cal_acc_2(self, model_name, synthetic_images, synthetic_labels, sensitive_val_loader, sensitive_test_loader):

        synthetic_images_train, synthetic_images_val = synthetic_images[:55000], synthetic_images[55000:]
        synthetic_labels_train, synthetic_labels_val = synthetic_labels[:55000], synthetic_labels[55000:]
    
        num_classes = len(set(synthetic_labels))
        criterion = nn.CrossEntropyLoss()
        lr = 1e-4

        if 'cifar' in self.config.sensitive_data.name:
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
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

        model = torch.nn.DataParallel(model).to(self.device)

        ema = ExponentialMovingAverage(model.parameters(), 0.9999)

        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
        # optimizer = optim.Adam(model.parameters(), lr=0.01)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

        # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)

        train_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images_train).float(), torch.from_numpy(synthetic_labels_train).long()), shuffle=True, batch_size=batch_size)

        if sensitive_val_loader is None:
            val_loader = DataLoader(TensorDataset(torch.from_numpy(synthetic_images_val).float(), torch.from_numpy(synthetic_labels_val).long()), shuffle=True, batch_size=batch_size)
            sensitive_val = False
        else:
            val_loader = sensitive_val_loader
            sensitive_val = True
            val_acc_list = []

        best_acc = 0.
        best_noisy_correct = 0.
        best_test_acc_on_val = 0.
        best_test_acc_on_test = 0.

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
                    if len(targets.shape) == 2:
                        inputs = inputs.to(torch.float32) / 255.
                        targets = torch.argmax(targets, dim=1)
                    inputs, targets = inputs.to(self.device) * 2. - 1., targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
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
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
            
            test_acc = test_correct / test_total * 100

            logging.info("Epoch: {} Train acc: {} Val acc: {} Test acc{}; Train loss: {} Val loss: {}".format(epoch, train_acc, val_acc, test_acc, train_loss, val_loss))

            if sensitive_val:
                noisy_val_correct = val_correct + np.random.laplace(loc=0, scale=1/self.config.train.dp.epsilon)
                if noisy_val_correct >= best_noisy_correct:
                    best_noisy_correct = noisy_val_correct
                    best_noisy_acc = noisy_val_correct / val_total * 100
                    best_test_acc_on_noisy_val = test_acc
            else:
                best_noisy_acc = 0
                best_test_acc_on_noisy_val = 0

            if val_acc >= best_acc:
                best_acc = val_acc
                best_test_acc_on_val = test_acc

            if test_acc >= best_test_acc_on_test:
                best_test_acc_on_test = test_acc
            
            ema.restore(model.parameters())
        

        return best_acc, best_test_acc_on_val, best_test_acc_on_test, best_noisy_acc, best_test_acc_on_noisy_val
    
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

            if 'cifar' in self.config.sensitive_data.name:

                batch_size = 126
                max_epoch = 200

                if model_name == "wrn":
                    model = WideResNet(in_c=c, img_size=img_size, num_classes=num_classes, depth=28, widen_factor=10, dropRate=0.3)
                elif model_name == "resnet":
                    model = ResNet(in_c=c, img_size=img_size, num_classes=num_classes, depth=164, block_name='BasicBlock')
                elif model_name == "resnext":
                    model = ResNeXt(in_c=c, img_size=img_size, cardinality=8, depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)

                optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
            else:
                
                batch_size = 126
                max_epoch = 50

                if model_name == "wrn":
                    model = WideResNet(in_c=c, img_size=img_size, num_classes=num_classes,  dropRate=0.3)
                elif model_name == "resnet":
                    model = ResNet(in_c=c, img_size=img_size, num_classes=num_classes,  block_name='BasicBlock')
                elif model_name == "resnext":
                    model = ResNeXt(in_c=c, img_size=img_size, num_classes=num_classes, dropRate=0.3)

                
                optimizer = optim.Adam(model.parameters(), lr=0.01)
                # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)        
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)


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


def compute_inception_score_from_logits(logits, splits=1):
    scores = []
    for i in range(splits):
        part = logits[(i * logits.shape[0] // splits):((i + 1) * logits.shape[0] // splits), :]
        py = np.mean(part, axis=0)
        kl = np.mean([entropy(part[i, :], py) for i in range(part.shape[0])])
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def get_prompt(data_name: str):
    if data_name.startswith("mnist"):
        return ["A grayscale image of a handwritten digit 0", "A grayscale image of a handwritten digit 1", "an image of hand-written 2", "an image of hand-written 3", "an image of hand-written 4", "an image of hand-written 5", "an image of hand-written 6", "an image of hand-written 7", "an image of hand-written 8", "an image of hand-written 9"]
    elif data_name.startswith("fmnist"):
        return ["A grayscale image of a T-shirt", "A grayscale image of a handwritten digit Trouser", "an image of hand-written Pullover", "an image of hand-written Dress", "an image of hand-written Coat", "an image of hand-written Sandal", "an image of hand-written Shirt", "an image of hand-written Sneaker", "an image of hand-written Bag", "an image of hand-written Ankle boot"]
    elif data_name.startswith("cifar100"):
        cifar100_y = '''Superclass	Classes
        aquatic mammals	beaver, dolphin, otter, seal, whale
        fish	aquarium fish, flatfish, ray, shark, trout
        flowers	orchids, poppies, roses, sunflowers, tulips
        food containers	bottles, bowls, cans, cups, plates
        fruit and vegetables	apples, mushrooms, oranges, pears, sweet peppers
        household electrical devices	clock, computer keyboard, lamp, telephone, television
        household furniture	bed, chair, couch, table, wardrobe
        insects	bee, beetle, butterfly, caterpillar, cockroach
        large carnivores	bear, leopard, lion, tiger, wolf
        large man-made outdoor things	bridge, castle, house, road, skyscraper
        large natural outdoor scenes	cloud, forest, mountain, plain, sea
        large omnivores and herbivores	camel, cattle, chimpanzee, elephant, kangaroo
        medium-sized mammals	fox, porcupine, possum, raccoon, skunk
        non-insect invertebrates	crab, lobster, snail, spider, worm
        people	baby, boy, girl, man, woman
        reptiles	crocodile, dinosaur, lizard, snake, turtle
        small mammals	hamster, mouse, rabbit, shrew, squirrel
        trees	maple, oak, palm, pine, willow
        vehicles 1	bicycle, bus, motorcycle, pickup truck, train
        vehicles 2	lawn-mower, rocket, streetcar, tank, tractor'''
        class_lins = cifar100_y.split("\n")
        prompt = []
        for line in class_lins[1:]:
            super_class, child_classes = line.split("\t")
            for child_class in child_classes.split(", "):
                prompt.append("An image of {}".format(child_class))
        return prompt
    elif data_name.startswith("cifar10"):
        return ["An image of an airplane", "An image of an automobile", "An image of a bird", "An image of a cat", "An image of a deer", "An image of a dog", "An image of a frog", "An image of a horse", "An image of a ship", "An image of a truck"]
    elif data_name.startwith("eurosat"):
        return ["A remote sensing image of an industrial area", "A remote sensing image of a residential area", "A remote sensing image of an annual crop area", "A remote sensing image of a permanent crop area", "A remote sensing image of a river area", "A remote sensing image of a sea or lake area", "A remote sensing image of a herbaceous veg. area", "A remote sensing image of a highway area", "A remote sensing image of a pasture area", "A remote sensing image of a forest area"]
    elif data_name.startswith("celeba_male"):
        return ["An image of a female face", "An image of a male face"]
    elif data_name.startswith("camelyon"):
        return ["A normal lymph node image", "A lymph node histopathology image"]
    else:
        raise NotImplementedError
