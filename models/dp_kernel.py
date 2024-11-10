import os

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math
from torch.distributions.multivariate_normal import MultivariateNormal as mvn
import numpy as np
import logging

import importlib
opacus = importlib.import_module('opacus')
from opacus.accountants.utils import get_noise_multiplier

from models.DP_GAN.generator import Generator
from models.synthesizer import DPSynther

class DP_Kernel(DPSynther):
    def __init__(self, config, device):
        super().__init__()

        self.image_size = config.image_size
        self.nz = config.Generator.z_dim
        self.n_class = config.n_class
        self.sigma_list = config.sigma_list
        self.config = config
        self.device = device

        G_decoder =  Generator(img_size=self.image_size, num_classes=self.n_class, **config.Generator).to(device)
        self.gen = CNetG(G_decoder).to(device)

        model_parameters = filter(lambda p: p.requires_grad, self.gen.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)
    
    def pretrain(self, public_dataloader, config):
        if public_dataloader is None:
            return
        os.mkdir(config.log_dir)

        fixed_noise = torch.randn(8 * self.n_class, self.nz).to(self.device)
        fixed_label = torch.arange(self.n_class).repeat(8).to(self.device)

        optimizer = torch.optim.RMSprop(self.gen.parameters(), lr=config.lr)
        for epoch in range(config.n_epochs):
            for x, label in public_dataloader:
                iter_loss = 0

                if len(label.shape) == 2:
                    x = x.to(torch.float32) / 255.
                    label = torch.argmax(label, dim=1)
                if config.cond:
                    label = label % self.n_class
                else:
                    label = torch.zeros_like(label)
                x = x.to(self.device) * 2 - 1
                label = label.to(self.device)
                batch_size = x.size(0)

                gen_labels = get_gen_labels(label, self.n_class)

                optimizer.zero_grad()

                noise = torch.randn(batch_size, self.nz).to(self.device)
                y = self.gen(noise, label=gen_labels)

                label = F.one_hot(label, self.n_class).float()
                gen_labels = F.one_hot(gen_labels, self.n_class).float()

                #### compute mmd loss using my implementation ####
                DP_mmd_loss = rbf_kernel_DP_loss_with_labels(x.view(batch_size, -1), y.view(batch_size, -1), label, gen_labels, self.sigma_list, 0.)

                errG = torch.pow(DP_mmd_loss, 2)
                errG.backward()
                optimizer.step()
                iter_loss += errG.item()

            logging.info('Training loss: {}\tLoss:{:.6f}\t'.format(epoch, iter_loss))
            y_fixed = self.gen(fixed_noise, label=fixed_label)
            y_fixed.data = y_fixed.data.mul(0.5).add(0.5)
            grid = torchvision.utils.make_grid(y_fixed.data, nrow=10)
            torchvision.utils.save_image(grid, os.path.join(config.log_dir, f'netG_epoch{epoch}_lr{config.lr}_bs{config.batch_size}.png'))
            torch.save(self.gen.state_dict(), os.path.join(config.log_dir, f'netG_epoch{epoch}_lr{config.lr}_bs{config.batch_size}.pkl'))


    def train(self, sensitive_dataloader, config):
        if sensitive_dataloader is None:
            return
        if config.ckpt is not None:
            self.gen.load_state_dict(torch.load(config.ckpt))
        os.mkdir(config.log_dir)

        self.noise_factor = get_noise_multiplier(target_epsilon=config.dp.epsilon, target_delta=config.dp.delta, sample_rate=1/len(sensitive_dataloader), steps=config.max_iter)

        logging.info("The noise factor is {}".format(self.noise_factor))

        fixed_noise = torch.randn(8 * self.n_class, self.nz).to(self.device)
        fixed_label = torch.arange(self.n_class).repeat(8).to(self.device)

        optimizer = torch.optim.RMSprop(self.gen.parameters(), lr=config.lr)
        noise_multiplier = self.noise_factor
        iter = 0
        while True:
            for x, label in sensitive_dataloader:
                iter_loss = 0

                if len(label.shape) == 2:
                    x = x.to(torch.float32) / 255.
                    label = torch.argmax(label, dim=1)

                x = x.to(self.device) * 2 - 1
                label = label.to(self.device)
                batch_size = x.size(0)

                gen_labels = get_gen_labels(label, self.n_class)

                optimizer.zero_grad()

                noise = torch.randn(batch_size, self.nz).to(self.device)
                y = self.gen(noise, label=gen_labels)
                label = F.one_hot(label, self.n_class).float()
                gen_labels = F.one_hot(gen_labels, self.n_class).float()

                #### compute mmd loss using my implementation ####
                DP_mmd_loss = rbf_kernel_DP_loss_with_labels(x.view(batch_size, -1), y.view(batch_size, -1), label, gen_labels, self.sigma_list, noise_multiplier)

                errG = torch.pow(DP_mmd_loss, 2)
                errG.backward()
                optimizer.step()
                iter_loss += errG.item()

                if iter % config.vis_step == 0:
                    # training loss
                    logging.info('Current iter: {}'.format(iter) + 'Total training iters: {}'.format(config.max_iter))
                    logging.info('Training loss: {}\tLoss:{:.6f}\t'.format(iter, iter_loss))
                    y_fixed = self.gen(fixed_noise, label=fixed_label)
                    y_fixed.data = y_fixed.data.mul(0.5).add(0.5)
                    grid = torchvision.utils.make_grid(y_fixed.data, nrow=10)
                    torchvision.utils.save_image(grid, os.path.join(config.log_dir, f'netG_iter{iter}_noise{self.noise_factor}_lr{config.lr}_bs{config.batch_size}.png'))
                    torch.save(self.gen.state_dict(), os.path.join(config.log_dir, f'netG_iter{iter}_noise{self.noise_factor}_lr{config.lr}_bs{config.batch_size}.pkl'))

                iter += 1
                if iter >= config.max_iter:
                    break
            if iter >= config.max_iter:
                break

    def generate(self, config):
        os.mkdir(config.log_dir)
        syn_data = []
        syn_labels = []

        for _ in range(int(config.data_num / config.batch_size)):
            y = torch.randint(self.n_class, (config.batch_size,)).to(self.device)
            z = torch.randn(config.batch_size, self.nz).to(self.device)
            images = self.gen(z, y)
            if images.shape[1] == 1:
                images = F.interpolate(images, size=[28, 28])

            syn_data.append(images.detach().cpu().numpy())
            syn_labels.append(y.detach().cpu().numpy())
        
        syn_data = np.concatenate(syn_data) / 2 + 0.5
        syn_labels = np.concatenate(syn_labels)

        np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

        show_images = []
        for cls in range(self.n_class):
            show_images.append(syn_data[syn_labels==cls][:8])
        show_images = np.concatenate(show_images)
        torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
        return syn_data, syn_labels

class CNetG(nn.Module):
    def __init__(self, decoder):
        super(CNetG, self).__init__()
        self.decoder = decoder

    def forward(self, input, label):
        output = self.decoder(input, label)
        return output


def rbf_kernel_DP_loss_with_labels(X, Y, x_label, y_label, sigma_list, noise_multiplier):
    '''
    Compute Gaussian kernel between dataset X and Y, with labels
    :param X: N*d
    :param Y: M*d
    :return:
    '''
    N = X.size(0)
    M = Y.size(0)

    Z = torch.cat((X, Y), 0)
    L = torch.cat((x_label, y_label), 0)
    ZZT = torch.mm(Z, Z.t())
    LLT = torch.mm(L, L.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t() # (N+M)*(N+M)

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K += torch.exp(-gamma * exponent)

    K = K * LLT # new kernel account for labels

    K_XX = K[:N, :N]
    K_XY = K[:N, N:]
    K_YY = K[N:, N:]
    f_Dx = torch.mean(K_XX, dim=0) # (N,)
    f_Dy = torch.mean(K_XY, dim=0) # (M,)
    f_Dxy = torch.cat([f_Dx, f_Dy]) # size [N+M]

    # batch method
    if noise_multiplier == 0.:
        f_Dxy_tilde = f_Dxy
    else:
        coeff =  math.sqrt(2 * len(sigma_list)) / N * noise_multiplier
        try:
            mvn_Dxy = mvn(torch.zeros_like(f_Dxy), K * coeff)
            f_Dxy_tilde = f_Dxy + mvn_Dxy.sample()
            del mvn_Dxy
        except:
            f_Dxy_tilde = f_Dxy
    f_Dx_tilde = f_Dxy_tilde[:N] # [N]
    f_Dy_tilde = f_Dxy_tilde[N:] # [M]
    mmd_XX = torch.mean(f_Dx_tilde)
    mmd_XY = torch.mean(f_Dy_tilde)
    mmd_YY = torch.mean(K_YY)

    return mmd_XX - 2 * mmd_XY + mmd_YY


def get_gen_labels(labels, num_classes):
    return labels
    def has_isolated_integer(int_list):
        count_dict = {}
        for num in int_list:
            if num in count_dict:
                count_dict[num] += 1
            else:
                count_dict[num] = 1

        for count in count_dict.values():
            if count == 1:
                return True

        return False
    
    gen_labels = torch.randint(num_classes, (labels.shape[0],)).to(labels.device)
    while has_isolated_integer(torch.cat([labels, gen_labels])):
        gen_labels = torch.randint(num_classes, (labels.shape[0],)).to(labels.device)
    
    return gen_labels
    