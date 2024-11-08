import os
import logging

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
import torchvision

import importlib
opacus = importlib.import_module('opacus')
from opacus.accountants.utils import get_noise_multiplier


from models.synthesizer import DPSynther
from models.DP_NTK.dp_ntk_mean_emb1 import calc_mean_emb1
from models.DP_NTK.ntk import *
from models.DP_GAN.generator import Generator

class DP_NTK(DPSynther):
    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.device = device
        self.img_size = config.img_size
        self.c = config.c
        self.ntk_width = config.ntk_width
        self.input_dim = self.img_size * self.img_size * self.c
        self.n_classes = config.n_classes

        if config.model_ntk == 'fc_1l':
            self.model_ntk = NTK(input_size=self.input_dim, hidden_size_1=self.ntk_width, output_size=self.n_classes)
        elif config.model_ntk == 'fc_2l':
            self.model_ntk = NTK_TL(input_size=self.input_dim, hidden_size_1=self.ntk_width, hidden_size_2=config.ntk_width2, output_size=self.n_classes)
        elif config.model_ntk == 'lenet5':
            self.model_ntk = LeNet5()
        else:
            raise NotImplementedError('{} is not yet implemented.'.format(config.model_ntk))

        self.model_ntk.to(device)
        self.model_ntk.eval()

        self.model_gen = Generator(img_size=self.img_size, num_classes=self.n_classes, **config.Generator).to(device)
        self.model_gen.train()

        model_parameters = filter(lambda p: p.requires_grad, self.model_gen.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)
    
    def pretrain(self, public_dataloader, config):
        if public_dataloader is None:
            return
        os.mkdir(config.log_dir)
        # define loss function

        self.noisy_mean_emb = calc_mean_emb1(self.model_ntk, public_dataloader, self.n_classes, 0., self.device, label_random=config.label_random)

        torch.save(self.noisy_mean_emb, os.path.join(config.log_dir, 'noisy_mean_emb.pt'))

        optimizer = torch.optim.Adam(self.model_gen.parameters(), lr=config.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=config.lr_decay)

        """ initialize the variables """
        mean_v_samp = torch.Tensor([]).to(self.device)
        for p in self.model_ntk.parameters():
            mean_v_samp = torch.cat((mean_v_samp, p.flatten()))
        d = len(mean_v_samp)
        logging.info('Feature Length: {}'.format(d))

        """ training a Generator via minimizing MMD """
        for epoch in range(config.n_iter):  # loop over the dataset multiple times
            running_loss = 0.0
            optimizer.zero_grad()  # zero the parameter gradients

            """ synthetic data """
            for accu_step in range(config.n_splits):
                batch_size = config.batch_size // config.n_splits
                gen_labels_numerical = torch.randint(self.n_classes, (batch_size,)).to(self.device)
                z = torch.randn(batch_size, self.model_gen.z_dim).to(self.device)
                gen_samples = self.model_gen(z, gen_labels_numerical).reshape(batch_size, -1) / 2 + 0.5

                """ synthetic data mean_emb init """
                mean_emb2 = torch.zeros((d, self.n_classes), device=self.device)
                for idx in range(gen_samples.shape[0]):
                    """ manually set the weight if needed """
                    # model_ntk.fc1.weight = torch.nn.Parameter(output_weights[gen_labels_numerical[idx], :][None, :])
                    mean_v_samp = torch.Tensor([]).to(self.device)  # sample mean vector init
                    f_x = self.model_ntk(gen_samples[idx][None, :])

                    """ get NTK features """
                    f_idx_grad = torch.autograd.grad(f_x, self.model_ntk.parameters(),
                                                    grad_outputs=torch.ones_like(f_x), create_graph=True)
                    # f_idx_grad = torch.autograd.grad(f_x.sum(), self.model_ntk.parameters(), create_graph=True)
                    for g in f_idx_grad:
                        mean_v_samp = torch.cat((mean_v_samp, g.flatten()))
                    # mean_v_samp = mean_v_samp[:-1]

                    """ normalize the sample mean vector """
                    mean_emb2[:, gen_labels_numerical[idx]] += mean_v_samp / torch.linalg.vector_norm(mean_v_samp)

                """ average by batch size """
                mean_emb2 = mean_emb2 / batch_size

                """ calculate loss """
                # loss = (self.noisy_mean_emb - mean_emb2).sum()
                loss = torch.norm(self.noisy_mean_emb - mean_emb2, p=2) ** 2 / config.n_splits
                loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (epoch + 1) % config.log_interval == 0:
                logging.info('iter {} and running loss are {}'.format(epoch, running_loss))
            if epoch % config.scheduler_interval == 0:
                scheduler.step()
        
        torch.save(self.model_gen.state_dict(), os.path.join(config.log_dir, 'gen.pt'))

    def train(self, sensitive_dataloader, config):
        if sensitive_dataloader is None:
            return
        if config.ckpt is not None:
            self.model_gen.load_state_dict(torch.load(config.ckpt))
        os.mkdir(config.log_dir)
        # define loss function
        self.noise_factor = get_noise_multiplier(target_epsilon=config.dp.epsilon, target_delta=config.dp.delta, sample_rate=1., epochs=1)

        logging.info("The noise factor is {}".format(self.noise_factor))

        self.noisy_mean_emb = calc_mean_emb1(self.model_ntk, sensitive_dataloader, self.n_classes, self.noise_factor, self.device)
        print(self.noisy_mean_emb.shape)

        torch.save(self.noisy_mean_emb, os.path.join(config.log_dir, 'noisy_mean_emb.pt'))


        optimizer = torch.optim.Adam(self.model_gen.parameters(), lr=config.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=config.lr_decay)

        """ initialize the variables """
        mean_v_samp = torch.Tensor([]).to(self.device)
        for p in self.model_ntk.parameters():
            mean_v_samp = torch.cat((mean_v_samp, p.flatten()))
        d = len(mean_v_samp)
        logging.info('Feature Length: {}'.format(d))

        """ training a Generator via minimizing MMD """
        for epoch in range(config.n_iter):  # loop over the dataset multiple times
            running_loss = 0.0
            optimizer.zero_grad()  # zero the parameter gradients

            """ synthetic data """
            for accu_step in range(config.n_splits):
                batch_size = config.batch_size // config.n_splits
                gen_labels_numerical = torch.randint(self.n_classes, (batch_size,)).to(self.device)
                z = torch.randn(batch_size, self.model_gen.z_dim).to(self.device)
                gen_samples = self.model_gen(z, gen_labels_numerical).reshape(batch_size, -1) / 2 + 0.5

                """ synthetic data mean_emb init """
                mean_emb2 = torch.zeros((d, self.n_classes), device=self.device)
                for idx in range(gen_samples.shape[0]):
                    """ manually set the weight if needed """
                    # model_ntk.fc1.weight = torch.nn.Parameter(output_weights[gen_labels_numerical[idx], :][None, :])
                    mean_v_samp = torch.Tensor([]).to(self.device)  # sample mean vector init
                    f_x = self.model_ntk(gen_samples[idx][None, :])

                    """ get NTK features """
                    f_idx_grad = torch.autograd.grad(f_x, self.model_ntk.parameters(),
                                                    grad_outputs=torch.ones_like(f_x), create_graph=True)
                    # f_idx_grad = torch.autograd.grad(f_x.sum(), self.model_ntk.parameters(), create_graph=True)
                    for g in f_idx_grad:
                        mean_v_samp = torch.cat((mean_v_samp, g.flatten()))
                    # mean_v_samp = mean_v_samp[:-1]

                    """ normalize the sample mean vector """
                    mean_emb2[:, gen_labels_numerical[idx]] += mean_v_samp / torch.linalg.vector_norm(mean_v_samp)

                """ average by batch size """
                mean_emb2 = mean_emb2 / batch_size

                """ calculate loss """
                # loss = (self.noisy_mean_emb - mean_emb2).sum()
                loss = torch.norm(self.noisy_mean_emb - mean_emb2, p=2) ** 2 / config.n_splits
                loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (epoch + 1) % config.log_interval == 0:
                logging.info('iter {} and running loss are {}'.format(epoch, running_loss))
            if epoch % config.scheduler_interval == 0:
                scheduler.step()
        
        torch.save(self.model_gen.state_dict(), os.path.join(config.log_dir, 'gen.pt'))

    def generate(self, config):
        os.mkdir(config.log_dir)

        """evaluate the model"""
        syn_data, syn_labels = synthesize_mnist_with_uniform_labels(self.model_gen, self.device, gen_batch_size=config.batch_size, n_data=config.data_num, n_labels=self.n_classes)
        syn_data = syn_data.reshape(syn_data.shape[0], self.c, self.img_size, self.img_size)
        syn_labels = syn_labels.reshape(-1)
        np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

        show_images = []
        for cls in range(self.n_classes):
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
