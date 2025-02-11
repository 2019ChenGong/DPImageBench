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
        # Initialize attributes from the configuration
        self.image_size = config.image_size
        self.nz = config.Generator.z_dim
        self.private_num_classes = config.private_num_classes
        self.public_num_classes = config.public_num_classes
        label_dim = max(self.private_num_classes, self.public_num_classes)
        self.sigma_list = config.sigma_list
        self.config = config
        self.device = device

        # Instantiate the generator
        G_decoder =  Generator(img_size=self.image_size, num_classes=label_dim, **config.Generator).to(device)
        self.gen = CNetG(G_decoder).to(device)

        # Load pre-trained model if a checkpoint is provided
        if config.ckpt is not None:
            self.gen.load_state_dict(torch.load(config.ckpt))

        # Log the number of trainable parameters
        model_parameters = filter(lambda p: p.requires_grad, self.gen.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)
    
    # Pretraining function using public data
    def pretrain(self, public_dataloader, config):
        # If dataloader is not provided, exit the function early
        if public_dataloader is None:
            return
        
        # Create directory for logs
        os.mkdir(config.log_dir)
        os.mkdir(os.path.join(config.log_dir, "samples"))
        os.mkdir(os.path.join(config.log_dir, "checkpoints"))

        # Prepare fixed noise and labels for visualization
        fixed_noise = torch.randn(8 * 10, self.nz).to(self.device)
        fixed_label = torch.arange(10).repeat(8).to(self.device)

        # Initialize optimizer for the generator network
        optimizer = torch.optim.RMSprop(self.gen.parameters(), lr=config.lr)
        
        # Loop over number of epochs specified in config
        for epoch in range(config.n_epochs):
            for x, label in public_dataloader:
                iter_loss = 0
                
                # Convert labels to float and normalize images if labels are one-hot encoded
                if len(label.shape) == 2:
                    x = x.to(torch.float32) / 255.
                    label = torch.argmax(label, dim=1)
                
                # Set labels to zero if conditional training is disabled
                if not config.cond:
                    label = torch.zeros_like(label)
                
                # Normalize images to [-1, 1] and move data to device
                x = x.to(self.device) * 2 - 1
                label = label.to(self.device)
                batch_size = x.size(0)

                # Get generated labels for current batch
                gen_labels = get_gen_labels(label, self.public_num_classes)

                # Zero the gradients before running the backward pass
                optimizer.zero_grad()

                # Generate noise and labels for the current batch
                noise = torch.randn(batch_size, self.nz).to(self.device)
                y = self.gen(noise, label=gen_labels)

                # Convert labels to one-hot encoded format
                label = F.one_hot(label, self.public_num_classes).float()
                gen_labels = F.one_hot(gen_labels, self.public_num_classes).float()

                # Calculate Maximum Mean Discrepancy (MMD) loss between real and generated data
                DP_mmd_loss = rbf_kernel_DP_loss_with_labels(x.view(batch_size, -1), 
                                                            y.view(batch_size, -1), 
                                                            label, 
                                                            gen_labels, 
                                                            self.sigma_list, 
                                                            0.)
                
                # Compute squared MMD loss
                errG = torch.pow(DP_mmd_loss, 2)
                # Backpropagate the loss
                errG.backward()
                # Update the parameters
                optimizer.step()
                # Accumulate the loss
                iter_loss += errG.item()

            # Log training loss after each epoch
            logging.info('Training loss: {}\tLoss:{:.6f}\t'.format(epoch, iter_loss))
            
            # Generate images using fixed noise and labels for visualization
            y_fixed = self.gen(fixed_noise, label=fixed_label)
            # Denormalize the images
            y_fixed.data = y_fixed.data.mul(0.5).add(0.5)
            # Make a grid of images for visualization
            grid = torchvision.utils.make_grid(y_fixed.data, nrow=10)

            # Save the generated images
            torchvision.utils.save_image(grid, os.path.join(config.log_dir, 'samples', f'epoch{epoch}.png'))
            
            # Save the state dictionary of the generator network
            torch.save(self.gen.state_dict(), os.path.join(config.log_dir, 'checkpoints', 'snapshot_checkpoint.pth'))
        torch.save(self.gen.state_dict(), os.path.join(config.log_dir, 'checkpoints', 'final_checkpoint.pth'))


    # Training function using sensitive data with differential privacy
    def train(self, sensitive_dataloader, config):
        # Check if the dataloader is provided
        if sensitive_dataloader is None:
            return
        
        # Create a directory for logging
        os.mkdir(config.log_dir)
        os.mkdir(os.path.join(config.log_dir, "samples"))
        os.mkdir(os.path.join(config.log_dir, "checkpoints"))

        # Calculate the noise multiplier for differential privacy
        self.noise_factor = get_noise_multiplier(
            target_epsilon=config.dp.epsilon, 
            target_delta=config.dp.delta, 
            sample_rate=1/len(sensitive_dataloader), 
            steps=config.max_iter
        )

        # Log the noise factor
        logging.info("The noise factor is {}".format(self.noise_factor))

        # Generate fixed noise and labels for visualization
        fixed_noise = torch.randn(8 * self.private_num_classes, self.nz).to(self.device)
        fixed_label = torch.arange(self.private_num_classes).repeat(8).to(self.device)

        # Initialize the optimizer
        optimizer = torch.optim.RMSprop(self.gen.parameters(), lr=config.lr)
        noise_multiplier = self.noise_factor
        iter = 0

        # Training loop
        while True:
            for x, label in sensitive_dataloader:
                iter_loss = 0

                # Preprocess the input data
                if len(label.shape) == 2:
                    x = x.to(torch.float32) / 255.
                    label = torch.argmax(label, dim=1)

                # Normalize the input data
                x = x.to(self.device) * 2 - 1
                label = label.to(self.device)
                batch_size = x.size(0)

                # Generate labels for the generator
                gen_labels = get_gen_labels(label, self.private_num_classes)

                # Zero the gradients
                optimizer.zero_grad()

                # Generate noise and pass it through the generator
                noise = torch.randn(batch_size, self.nz).to(self.device)
                y = self.gen(noise, label=gen_labels)
                label = F.one_hot(label, self.private_num_classes).float()
                gen_labels = F.one_hot(gen_labels, self.private_num_classes).float()

                # Compute the MMD loss with differential privacy
                DP_mmd_loss = rbf_kernel_DP_loss_with_labels(
                    x.view(batch_size, -1), 
                    y.view(batch_size, -1), 
                    label, 
                    gen_labels, 
                    self.sigma_list, 
                    noise_multiplier
                )

                # Compute the error and backpropagate
                errG = torch.pow(DP_mmd_loss, 2)
                errG.backward()
                optimizer.step()
                iter_loss += errG.item()

                # Log and visualize results at specified intervals
                if iter % config.vis_step == 0:
                    # Log the current iteration and total iterations
                    logging.info('Current iter: {}'.format(iter) + ' Total training iters: {}'.format(config.max_iter))
                    logging.info('Training loss: {}\tLoss:{:.6f}\t'.format(iter, iter_loss))

                    # Generate images for visualization
                    y_fixed = self.gen(fixed_noise, label=fixed_label)
                    y_fixed.data = y_fixed.data.mul(0.5).add(0.5)
                    grid = torchvision.utils.make_grid(y_fixed.data, nrow=10)
                    torchvision.utils.save_image(grid, os.path.join(config.log_dir, 'samples', f'iter{iter}.png'))

                    # Save the model state
                    torch.save(self.gen.state_dict(), os.path.join(config.log_dir, 'checkpoints', 'snapshot_checkpoint.pth'))

                # Increment the iteration counter
                iter += 1

                # Break the loop if the maximum number of iterations is reached
                if iter >= config.max_iter:
                    break
            
            # Break the outer loop if the maximum number of iterations is reached
            if iter >= config.max_iter:
                break
        torch.save(self.gen.state_dict(), os.path.join(config.log_dir, 'checkpoints', 'final_checkpoint.pth'))


    # Function to generate synthetic data
    def generate(self, config):
        # Create the directory to store logs and generated data
        os.mkdir(config.log_dir)
        
        # Initialize empty lists to store synthetic data and labels
        syn_data = []
        syn_labels = []

        # Generate synthetic data in batches
        for _ in range(int(config.data_num / config.batch_size)):
            # Generate random labels for the batch
            y = torch.randint(self.private_num_classes, (config.batch_size,)).to(self.device)
            # Generate random noise vectors for the batch
            z = torch.randn(config.batch_size, self.nz).to(self.device)
            
            # Generate images using the generator model
            images = self.gen(z, y)
            
            # If the generated images are grayscale, resize them to 28x28
            if images.shape[1] == 1:
                images = F.interpolate(images, size=[28, 28])

            # Append the generated images and labels to the lists
            syn_data.append(images.detach().cpu().numpy())
            syn_labels.append(y.detach().cpu().numpy())
        
        # Concatenate all generated data and normalize it
        syn_data = np.concatenate(syn_data) / 2 + 0.5
        # Concatenate all generated labels
        syn_labels = np.concatenate(syn_labels)

        # Save the generated data and labels as an .npz file
        np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

        # Prepare a set of images to display
        show_images = []
        for cls in range(self.private_num_classes):
            # Select up to 8 images for each class
            show_images.append(syn_data[syn_labels == cls][:8])
        # Concatenate the selected images
        show_images = np.concatenate(show_images)
        
        # Save the concatenated images as a grid image
        torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
        
        # Return the generated data and labels
        return syn_data, syn_labels


# Wrapper class for the generator
class CNetG(nn.Module):
    def __init__(self, decoder):
        super(CNetG, self).__init__()
        self.decoder = decoder

    def forward(self, input, label):
        output = self.decoder(input, label)
        return output

# Function to compute the RBF kernel loss with labels
def rbf_kernel_DP_loss_with_labels(X, Y, x_label, y_label, sigma_list, noise_multiplier):
    N = X.size(0)
    M = Y.size(0)

    Z = torch.cat((X, Y), 0)
    L = torch.cat((x_label, y_label), 0)
    ZZT = torch.mm(Z, Z.t())
    LLT = torch.mm(L, L.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma ** 2)
        K += torch.exp(-gamma * exponent)

    K = K * LLT

    K_XX = K[:N, :N]
    K_XY = K[:N, N:]
    K_YY = K[N:, N:]
    f_Dx = torch.mean(K_XX, dim=0)
    f_Dy = torch.mean(K_XY, dim=0)
    f_Dxy = torch.cat([f_Dx, f_Dy])

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
    f_Dx_tilde = f_Dxy_tilde[:N]
    f_Dy_tilde = f_Dxy_tilde[N:]
    mmd_XX = torch.mean(f_Dx_tilde)
    mmd_XY = torch.mean(f_Dy_tilde)
    mmd_YY = torch.mean(K_YY)

    return mmd_XX - 2 * mmd_XY + mmd_YY

# Function to get generator labels
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
    