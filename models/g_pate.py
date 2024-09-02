import os

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torchvision
import numpy as np
import logging

import importlib
opacus = importlib.import_module('opacus')
from opacus.accountants.utils import get_noise_multiplier


from models.synthesizer import DPSynther
from models.DP_MERF.rff_mmd_approx import get_rff_losses

class DP_MERF(DPSynther):
    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.z_dim = config.z_dim
        self.h_dim = config.h_dim
        self.num_class = config.num_class
        self.n_channels = config.n_channels
        self.kernel_sizes = config.kernel_sizes
        self.device = device
        # self.gen = CondDecoder(isize=32, nc=3, k=100).to(device)

        self.n_feat = config.n_feat
        self.d_rff = config.d_rff
        self.rff_sigma = config.rff_sigma
        self.mmd_type = config.mmd_type

        self.gen = ConvCondGen(self.z_dim, self.h_dim, self.num_class, self.n_channels, self.kernel_sizes, self.n_feat).to(device)

    def train(self, sensitive_dataloader, config):
        os.mkdir(config.log_dir)
        # define loss function
        self.noise_factor = get_noise_multiplier(target_epsilon=config.dp.epsilon, target_delta=config.dp.delta, sample_rate=1., epochs=1)

        logging.info("The noise factor is {}".format(self.noise_factor))

        sr_loss, mb_loss, _ = get_rff_losses(sensitive_dataloader, self.n_feat, self.d_rff, self.rff_sigma, self.device, 
                                             self.num_class, self.noise_factor, self.mmd_type)

        # rff_mmd_loss = get_rff_mmd_loss(n_feat, ar.d_rff, ar.rff_sigma, device, ar.n_labels, ar.noise_factor, ar.batch_size)

        # init optimizer
        optimizer = torch.optim.Adam(list(self.gen.parameters()), lr=config.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=config.lr_decay)

        # training loop
        for epoch in range(1, config.epochs + 1):
            train_single_release(self.gen, self.device, optimizer, epoch, sr_loss, config.log_interval, config.batch_size, config.n_data)
            scheduler.step()

        # save trained model and data
        torch.save(self.gen.state_dict(), os.path.join(config.log_dir, 'gen.pt'))

    def generate(self, config):
        os.mkdir(config.log_dir)
        syn_data, syn_labels = synthesize_mnist_with_uniform_labels(self.gen, self.device, gen_batch_size=config.batch_size, n_data=config.data_num, n_labels=self.num_class)
        syn_data = syn_data.reshape(syn_data.shape[0], config.num_channels, config.resolution, config.resolution)
        syn_labels = syn_labels.reshape(-1)
        np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

        show_images = []
        for cls in range(self.num_class):
            show_images.append(syn_data[syn_labels==cls][:8])
        show_images = np.concatenate(show_images)
        torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
        return syn_data, syn_labels


class ConvCondGen(nn.Module):
    def __init__(self, d_code, d_hid, n_labels, nc_str, ks_str, n_feats, use_sigmoid=True, batch_norm=True):
        super(ConvCondGen, self).__init__()
        self.nc = [int(k) for k in nc_str.split(',')]
        self.ks = [int(k) for k in ks_str.split(',')]  # kernel sizes
        d_hid = [int(k) for k in d_hid.split(',')]
        assert len(self.nc) == 3 and len(self.ks) == 2
        self.hw = int(np.sqrt(n_feats // self.nc[-1])) // 4
        self.reshape_size = self.nc[0]*self.hw**2
        self.fc1 = nn.Linear(d_code + n_labels, d_hid[0])
        self.fc2 = nn.Linear(d_hid[0], self.reshape_size)
        self.bn1 = nn.BatchNorm1d(d_hid[0]) if batch_norm else None
        self.bn2 = nn.BatchNorm1d(self.reshape_size) if batch_norm else None
        self.conv1 = nn.Conv2d(self.nc[0], self.nc[1], kernel_size=self.ks[0], stride=1, padding=(self.ks[0]-1)//2)
        self.conv2 = nn.Conv2d(self.nc[1], self.nc[2], kernel_size=self.ks[1], stride=1, padding=(self.ks[1]-1)//2)
        self.upsamp = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid
        self.d_code = d_code
        self.n_labels = n_labels

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x) if self.bn1 is not None else x
        x = self.fc2(self.relu(x))
        x = self.bn2(x) if self.bn2 is not None else x
        # print(x.shape)
        x = x.reshape(x.shape[0], self.nc[0], self.hw, self.hw)
        x = self.upsamp(x)
        x = self.relu(self.conv1(x))
        x = self.upsamp(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x
    def get_code(self, batch_size, device, return_labels=True, labels=None):
        if labels is None:  # sample labels
            labels = torch.randint(self.n_labels, (batch_size, 1), device=device)
        code = torch.randn(batch_size, self.d_code, device=device)
        gen_one_hots = torch.zeros(batch_size, self.n_labels, device=device)
        gen_one_hots.scatter_(1, labels, 1)
        code = torch.cat([code, gen_one_hots.to(torch.float32)], dim=1)
        # print(code.shape)
        if return_labels:
            return code, gen_one_hots
        else:
            return code


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
            gen_code, gen_labels = gen.get_code(gen_batch_size, device, labels=ordered_labels)
            gen_samples = gen(gen_code)
            data_list.append(gen_samples)
    return torch.cat(data_list, dim=0).cpu().numpy(), torch.cat(labels_list, dim=0).cpu().numpy()


def train_single_release(gen, device, optimizer, epoch, rff_mmd_loss, log_interval, batch_size, n_data):
    n_iter = n_data // batch_size
    for batch_idx in range(n_iter):
        gen_code, gen_labels = gen.get_code(batch_size, device)
        loss = rff_mmd_loss(gen(gen_code), gen_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            logging.info('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * batch_size, n_data, loss.item()))