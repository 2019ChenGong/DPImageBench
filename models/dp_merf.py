import os

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torchvision
import numpy as np
import logging

import importlib
opacus = importlib.import_module('opacus')
from opacus.accountants.utils import get_noise_multiplier


from models.synthesizer import DPSynther
from models.DP_MERF.rff_mmd_approx import get_rff_losses
from models.DP_GAN.generator import Generator

class DP_MERF(DPSynther):
    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.z_dim = config.Generator.z_dim
        self.private_num_classes = config.private_num_classes
        self.public_num_classes = config.public_num_classes
        label_dim = max(self.private_num_classes, self.public_num_classes)
        self.img_size = config.img_size
        self.device = device

        self.n_feat = config.n_feat
        self.d_rff = config.d_rff
        self.rff_sigma = config.rff_sigma
        self.mmd_type = config.mmd_type

        self.gen = Generator(img_size=self.img_size, num_classes=label_dim, **config.Generator).to(device)

        model_parameters = filter(lambda p: p.requires_grad, self.gen.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)
    def pretrain(self, public_dataloader, config):
        if public_dataloader is None:
            return
        os.mkdir(config.log_dir)
        # define loss function
        n_data = len(public_dataloader.dataset)
        sr_loss, mb_loss, _ = get_rff_losses(public_dataloader, self.n_feat, self.d_rff, self.rff_sigma, self.device, self.public_num_classes, 0., self.mmd_type, cond=config.cond)

        # rff_mmd_loss = get_rff_mmd_loss(n_feat, ar.d_rff, ar.rff_sigma, device, ar.n_labels, ar.noise_factor, ar.batch_size)

        # init optimizer
        optimizer = torch.optim.Adam(list(self.gen.parameters()), lr=config.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=config.lr_decay)

        # training loop
        for epoch in range(1, config.epochs + 1):
            train_single_release(self.gen, self.device, optimizer, epoch, sr_loss, config.log_interval, config.batch_size, n_data, self.public_num_classes, cond=config.cond)
            scheduler.step()

        # save trained model and data
        torch.save(self.gen.state_dict(), os.path.join(config.log_dir, 'gen.pt'))

    def train(self, sensitive_dataloader, config):
        if sensitive_dataloader is None:
            return
        if config.ckpt is not None:
            self.gen.load_state_dict(torch.load(config.ckpt))
        os.mkdir(config.log_dir)
        # define loss function
        self.noise_factor = get_noise_multiplier(target_epsilon=config.dp.epsilon, target_delta=config.dp.delta, sample_rate=1., epochs=1)

        logging.info("The noise factor is {}".format(self.noise_factor))
        
        n_data = len(sensitive_dataloader.dataset)
        sr_loss, mb_loss, _ = get_rff_losses(sensitive_dataloader, self.n_feat, self.d_rff, self.rff_sigma, self.device, self.private_num_classes, self.noise_factor, self.mmd_type)

        # rff_mmd_loss = get_rff_mmd_loss(n_feat, ar.d_rff, ar.rff_sigma, device, ar.n_labels, ar.noise_factor, ar.batch_size)

        # init optimizer
        optimizer = torch.optim.Adam(list(self.gen.parameters()), lr=config.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=config.lr_decay)

        # training loop
        for epoch in range(1, config.epochs + 1):
            train_single_release(self.gen, self.device, optimizer, epoch, sr_loss, config.log_interval, config.batch_size, n_data, self.private_num_classes)
            scheduler.step()

        # save trained model and data
        torch.save(self.gen.state_dict(), os.path.join(config.log_dir, 'gen.pt'))

    def generate(self, config):
        os.mkdir(config.log_dir)
        syn_data, syn_labels = synthesize_mnist_with_uniform_labels(self.gen, self.device, gen_batch_size=config.batch_size, n_data=config.data_num, n_labels=self.private_num_classes)
        syn_data = syn_data.reshape(syn_data.shape[0], config.num_channels, config.resolution, config.resolution)
        syn_labels = syn_labels.reshape(-1)
        np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

        show_images = []
        for cls in range(self.private_num_classes):
            show_images.append(syn_data[syn_labels==cls][:8])
        show_images = np.concatenate(show_images)
        torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
        return syn_data, syn_labels


def synthesize_mnist_with_uniform_labels(gen, device, gen_batch_size=1000, n_data=60000, n_labels=10):
    gen.eval()
    assert n_data % gen_batch_size == 0
    assert gen_batch_size % n_labels == 0
    n_iterations = n_data // gen_batch_size

    data_list = []
    ordered_labels = torch.repeat_interleave(torch.arange(n_labels), gen_batch_size // n_labels)[:, None].to(device)
    labels_list = [ordered_labels] * n_iterations

    with torch.no_grad():
        for idx in range(n_iterations):
            y = ordered_labels.view(-1)
            z = torch.randn(gen_batch_size, gen.z_dim).to(device)
            gen_samples = gen(z, y).reshape(gen_batch_size, -1) / 2 + 0.5
            data_list.append(gen_samples)
    return torch.cat(data_list, dim=0).cpu().numpy(), torch.cat(labels_list, dim=0).cpu().numpy()


def train_single_release(gen, device, optimizer, epoch, rff_mmd_loss, log_interval, batch_size, n_data, num_classes, cond=True):
    n_iter = n_data // batch_size
    for batch_idx in range(n_iter):
        if cond:
            y = torch.randint(num_classes, (batch_size,)).to(device)
        else:
            y = torch.zeros((batch_size,)).long().to(device)
        z = torch.randn(batch_size, gen.z_dim).to(device)
        gen_one_hots = F.one_hot(y, num_classes=num_classes)
        gen_samples = gen(z, y).reshape(batch_size, -1) / 2 + 0.5
        loss = rff_mmd_loss(gen_samples, gen_one_hots)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            logging.info('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * batch_size, n_data, loss.item()))