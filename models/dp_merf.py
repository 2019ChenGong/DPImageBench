import os

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torchvision
import numpy as np
import logging
from scipy.optimize import root_scalar
import scipy

import importlib
opacus = importlib.import_module('opacus')
# from opacus.accountants.utils import get_noise_multiplier


from models.synthesizer import DPSynther
from models.DP_MERF.rff_mmd_approx import get_rff_losses
from models.DP_GAN.generator import Generator

import torch
import torch.nn as nn

class ParameterizedImageGenerator(nn.Module):
    def __init__(self, num_images, img_channels, img_height, img_width, initial_images=None):
        super(ParameterizedImageGenerator, self).__init__()
        
        self.num_images = num_images
        self.img_shape = (img_channels, img_height, img_width)
        self.z_dim = 5

        if initial_images is not None:
            if initial_images.shape != (num_images, img_channels, img_height, img_width):
                raise ValueError(f"initial_images shape {initial_images.shape} does not match "
                                 f"expected shape ({num_images}, {img_channels}, {img_height}, {img_width})")
            # 将提供的初始图片作为模型参数
            self.images = nn.Parameter(initial_images.clone())
        else:
            # 随机初始化图片参数
            self.images = nn.Parameter(torch.randn(num_images, img_channels, img_height, img_width))
        
        self.labels = None

    def forward(self, x=None, y=None):
        return self.images / 255 * 2 - 1
    
def get_noise_multiplier(epsilon, num_steps, delta, min_noise_multiplier=1e-1, max_noise_multiplier=500, max_epsilon=1e7):

    def delta_Gaussian(eps, mu):
        # Compute delta of Gaussian mechanism with shift mu or equivalently noise scale 1/mu
        if mu == 0:
            return 0
        if np.isinf(np.exp(eps)):
            return 0
        return scipy.stats.norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * scipy.stats.norm.cdf(-eps / mu - mu / 2)

    def eps_Gaussian(delta, mu):
        # Compute eps of Gaussian mechanism with shift mu or equivalently noise scale 1/mu
        def f(x):
            return delta_Gaussian(x, mu) - delta
        return root_scalar(f, bracket=[0, max_epsilon], method='brentq').root

    def compute_epsilon(noise_multiplier, num_steps, delta):
        return eps_Gaussian(delta, np.sqrt(num_steps) / noise_multiplier)

    def objective(x):
        return compute_epsilon(noise_multiplier=x, num_steps=num_steps, delta=delta) - epsilon

    output = root_scalar(objective, bracket=[min_noise_multiplier, max_noise_multiplier], method='brentq')

    if not output.converged:
        raise ValueError("Failed to converge")

    return output.root


class DP_MERF(DPSynther):
    def __init__(self, config, device, sensitivity_ratio=1.0):
        super().__init__()
        # Initialize class variables based on the provided configuration
        self.config = config
        self.z_dim = config.Generator.z_dim  # Dimension of the latent space
        self.private_num_classes = config.private_num_classes  # Number of private classes
        self.public_num_classes = config.public_num_classes  # Number of public classes
        label_dim = max(self.private_num_classes, self.public_num_classes)  # Determine the maximum number of classes
        self.img_size = config.img_size  # Image size
        self.device = device  # Device (CPU or GPU)
        self.n_feat = config.n_feat  # Number of features
        self.d_rff = config.d_rff  # Dimension of random Fourier features
        self.rff_sigma = config.rff_sigma  # Sigma for random Fourier features
        self.mmd_type = config.mmd_type  # Type of Maximum Mean Discrepancy (MMD)
        self.sensitivity_ratio = sensitivity_ratio

        # Initialize the generator network
        if 'start_time' in config and config['start_time']:
            self.gen = ParameterizedImageGenerator(num_images=config['num_images'], img_channels=1 if self.img_size == 28 else 3, img_height=self.img_size, img_width=self.img_size).to(device)
        else:
            self.gen = Generator(img_size=self.img_size, num_classes=label_dim, **config.Generator).to(device)

        if config.ckpt is not None:
            self.gen.load_state_dict(torch.load(config.ckpt))  # Load checkpoint if provided

        # Count and log the number of trainable parameters in the model
        model_parameters = filter(lambda p: p.requires_grad, self.gen.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)
    
    # Method for pretraining using public data
    def pretrain(self, public_dataloader, config):
        if public_dataloader is None:
            return
        os.mkdir(config.log_dir)  # Create a directory for logs
        os.mkdir(os.path.join(config.log_dir, "samples"))
        os.mkdir(os.path.join(config.log_dir, "checkpoints"))

        # Define loss functions
        n_data = len(public_dataloader.dataset)  # Number of data points in the public dataset
        sr_loss, mb_loss, _ = get_rff_losses(public_dataloader, self.n_feat, self.d_rff, self.rff_sigma, self.device, self.public_num_classes, 0., self.mmd_type, cond=config.cond)

        # Initialize optimizer
        optimizer = torch.optim.Adam(list(self.gen.parameters()), lr=config.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=config.lr_decay)

        # Training loop
        for epoch in range(1, config.epochs + 1):
            train_single_release(self.gen, self.device, optimizer, epoch, sr_loss, config.log_interval, config.batch_size, n_data, self.public_num_classes, cond=config.cond)
            scheduler.step()  # Update learning rate

        # Save the pretrained model
        torch.save(self.gen.state_dict(), os.path.join(config.log_dir, 'checkpoints', 'final_checkpoint.pth'))

    # Method for training using sensitive data with differential privacy
    def train(self, sensitive_dataloader, config, start_images=None, start_labels=None):
        if sensitive_dataloader is None:
            return

        if start_images is not None:
            self.gen.images.data.copy_(start_images.to(self.device))
            self.gen.labels = start_labels.to(self.device)

        os.mkdir(config.log_dir)  # Create a directory for logs
        os.mkdir(os.path.join(config.log_dir, "samples"))
        os.mkdir(os.path.join(config.log_dir, "checkpoints"))

        # Define loss functions and compute noise factor
        # self.noise_factor = get_noise_multiplier(target_epsilon=config.dp.epsilon, target_delta=config.dp.delta, sample_rate=1., epochs=1)
        if 'sigma' in config.dp:
            self.noise_factor = config.dp.sigma
        elif config.dp.epsilon > 99999:
            self.noise_factor = 0.
        else:
            self.noise_factor = get_noise_multiplier(
                epsilon=config.dp.epsilon, 
                delta=config.dp.delta, 
                num_steps=1
            )
        logging.info("The noise factor is {}".format(self.noise_factor))

        n_data = len(sensitive_dataloader.dataset)  # Number of data points in the sensitive dataset
        sr_loss, mb_loss, noisy_emb = get_rff_losses(sensitive_dataloader, self.n_feat, self.d_rff, self.rff_sigma, self.device, self.private_num_classes, self.noise_factor * self.sensitivity_ratio, self.mmd_type, image_size=self.img_size)

        torch.save(noisy_emb.detach().cpu(), os.path.join(config.log_dir, "checkpoints", 'noisy_emb.pt'))

        # Initialize optimizer
        optimizer = torch.optim.Adam(list(self.gen.parameters()), lr=config.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=config.lr_decay)

        # Training loop
        for epoch in range(1, config.epochs + 1):
            train_single_release(self.gen, self.device, optimizer, epoch, sr_loss, config.log_interval, config.batch_size, n_data, self.private_num_classes)
            scheduler.step()  # Update learning rate

        # Save the trained model
        torch.save(self.gen.state_dict(), os.path.join(config.log_dir, 'checkpoints', 'final_checkpoint.pth'))

    # Method to generate synthetic data
    def generate(self, config):
        os.mkdir(config.log_dir)  # Create a directory for logs

        # Generate synthetic data and labels
        syn_data, syn_labels = synthesize_with_uniform_labels(self.gen, self.device, gen_batch_size=config.batch_size, n_data=config.data_num, n_labels=self.private_num_classes)
        syn_data = F.interpolate(syn_data, size=[config.resolution, config.resolution]).numpy()
        syn_labels = syn_labels.reshape(-1)

        # Save the generated data and labels
        np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

        # Prepare images to display
        show_images = []
        for cls in range(self.private_num_classes):
            show_images.append(syn_data[syn_labels == cls][:8])
        show_images = np.concatenate(show_images)
        torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)

        return syn_data, syn_labels



# Function to generate synthetic data with uniform labels
def synthesize_with_uniform_labels(gen, device, gen_batch_size=1000, n_data=60000, n_labels=10):
    gen.eval()
    assert n_data % gen_batch_size == 0
    assert gen_batch_size % n_labels == 0
    n_iterations = n_data // gen_batch_size

    data_list = []
    if hasattr(gen, 'images'):
        ordered_labels = gen.labels
        n_iterations = n_data // gen.labels.shape[0]
    else:
        ordered_labels = torch.repeat_interleave(torch.arange(n_labels), gen_batch_size // n_labels)[:, None].to(device)
    labels_list = [ordered_labels] * n_iterations

    with torch.no_grad():
        for idx in range(n_iterations):
            y = ordered_labels.view(-1)
            z = torch.randn(gen_batch_size, gen.z_dim).to(device)
            gen_samples = gen(z, y) / 2 + 0.5
            data_list.append(gen_samples)
    return torch.cat(data_list, dim=0).cpu(), torch.cat(labels_list, dim=0).cpu().numpy()

# function to train the generator for a single release
def train_single_release(gen, device, optimizer, epoch, rff_mmd_loss, log_interval, batch_size, n_data, num_classes, cond=True):
    n_iter = n_data // batch_size
    for batch_idx in range(n_iter):
        if cond:
            y = torch.randint(num_classes, (batch_size,)).to(device)
        else:
            y = torch.zeros((batch_size,)).long().to(device)
        if hasattr(gen, 'labels'):
            y = gen.labels
        z = torch.randn(batch_size, gen.z_dim).to(device)
        gen_one_hots = F.one_hot(y, num_classes=num_classes)
        gen_samples = gen(z, y).reshape(y.shape[0], -1) / 2 + 0.5
        # print(gen_samples.min(), gen_samples.max())
        loss = rff_mmd_loss(gen_samples, gen_one_hots)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            # logging.info('Train Epoch: {:.6f} {:.6f}'.format(z.sum().item(), gen_samples.sum().item()))
            logging.info('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * batch_size, n_data, loss.item()))
        # break