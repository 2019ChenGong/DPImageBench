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
        """
        Initializes the class with the given configuration and device.

        Args:
            config (object): Configuration object containing various settings and parameters.
            device (str): Device on which to run the models (e.g., 'cuda' or 'cpu').
        """
        super().__init__()

        self.config = config  # Store the configuration object
        self.device = device  # Store the device information
        self.img_size = config.img_size  # Image size from the configuration
        self.c = config.c  # Number of channels in the image
        self.ntk_width = config.ntk_width  # Width of the NTK layer
        self.input_dim = self.img_size * self.img_size * self.c  # Input dimension based on image size and channels
        self.private_num_classes = config.private_num_classes  # Number of private classes
        self.public_num_classes = config.public_num_classes  # Number of public classes
        label_dim = max(self.private_num_classes, self.public_num_classes)  # Determine the maximum number of classes

        # Initialize the NTK model based on the configuration
        if config.model_ntk == 'fc_1l':
            self.model_ntk = NTK(input_size=self.input_dim, hidden_size_1=self.ntk_width, output_size=label_dim)
        elif config.model_ntk == 'fc_2l':
            self.model_ntk = NTK_TL(input_size=self.input_dim, hidden_size_1=self.ntk_width, hidden_size_2=config.ntk_width2, output_size=label_dim)
        elif config.model_ntk == 'lenet5':
            self.model_ntk = LeNet5()
        else:
            raise NotImplementedError('{} is not yet implemented.'.format(config.model_ntk))

        # Move the NTK model to the specified device and set it to evaluation mode
        self.model_ntk.to(device)
        self.model_ntk.eval()

        # Initialize the generator model with the specified image size, number of classes, and additional configuration parameters
        self.model_gen = Generator(img_size=self.img_size, num_classes=label_dim, **config.Generator).to(device)
        if config.ckpt is not None:
            self.model_gen.load(torch.load(config.ckpt))
        self.model_gen.train()  # Set the generator model to training mode

        # Calculate the number of trainable parameters in the generator model
        model_parameters = filter(lambda p: p.requires_grad, self.model_gen.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)  # Log the number of trainable parameters

    def pretrain(self, public_dataloader, config):
        if public_dataloader is None:
            return
        os.mkdir(config.log_dir)
        os.mkdir(os.path.join(config.log_dir, "samples"))
        os.mkdir(os.path.join(config.log_dir, "checkpoints"))

        # Calculate the mean embedding of the public dataset
        self.noisy_mean_emb = calc_mean_emb1(self.model_ntk, public_dataloader, self.public_num_classes, 0., self.device, cond=config.cond)

        # Save the noisy mean embedding to disk
        torch.save(self.noisy_mean_emb, os.path.join(config.log_dir, "checkpoints", 'noisy_mean_emb.pt'))

        # Initialize the optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(self.model_gen.parameters(), lr=config.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=config.lr_decay)

        """ Initialize the variables """
        mean_v_samp = torch.Tensor([]).to(self.device)  # Initialize an empty tensor on the device
        for p in self.model_ntk.parameters():
            mean_v_samp = torch.cat((mean_v_samp, p.flatten()))  # Flatten and concatenate all parameters
        d = len(mean_v_samp)  # Determine the feature length
        logging.info('Feature Length: {}'.format(d))  # Log the feature length

        """ Training a Generator via minimizing MMD (Maximum Mean Discrepancy) """
        for epoch in range(config.n_iter):  # Loop over the dataset multiple times
            running_loss = 0.0
            optimizer.zero_grad()  # Zero the parameter gradients

            """ Generate synthetic data """
            for accu_step in range(config.n_splits):
                batch_size = config.batch_size // config.n_splits  # Determine the batch size for this split
                if config.cond:
                    gen_labels_numerical = torch.randint(self.public_num_classes, (batch_size,)).to(self.device)  # Randomly generate labels
                else:
                    gen_labels_numerical = torch.zeros((batch_size,)).long().to(self.device)  # Use a fixed label (0) if not conditional
                z = torch.randn(batch_size, self.model_gen.z_dim).to(self.device)  # Generate random noise
                gen_samples = self.model_gen(z, gen_labels_numerical).reshape(batch_size, -1) / 2 + 0.5  # Generate samples and normalize

                """ Initialize mean embedding for synthetic data """
                mean_emb2 = torch.zeros((d, self.public_num_classes), device=self.device)  # Initialize mean embedding matrix
                for idx in range(gen_samples.shape[0]):
                    """ Manually set the weight if needed (commented out) """
                    # model_ntk.fc1.weight = torch.nn.Parameter(output_weights[gen_labels_numerical[idx], :][None, :])
                    
                    mean_v_samp = torch.Tensor([]).to(self.device)  # Initialize the sample mean vector
                    f_x = self.model_ntk(gen_samples[idx][None, :])  # Pass the sample through the NTK model

                    """ Get NTK features """
                    f_idx_grad = torch.autograd.grad(f_x, self.model_ntk.parameters(),
                                                    grad_outputs=torch.ones_like(f_x), create_graph=True)
                    for g in f_idx_grad:
                        mean_v_samp = torch.cat((mean_v_samp, g.flatten()))  # Concatenate gradients

                    """ Normalize the sample mean vector """
                    mean_emb2[:, gen_labels_numerical[idx]] += mean_v_samp / torch.linalg.vector_norm(mean_v_samp)

                """ Average by batch size """
                mean_emb2 = mean_emb2 / batch_size

                """ Calculate the loss """
                loss = torch.norm(self.noisy_mean_emb - mean_emb2, p=2) ** 2 / config.n_splits
                loss.backward()  # Backpropagate the loss

            optimizer.step()  # Update the generator parameters

            running_loss += loss.item()
            if (epoch + 1) % config.log_interval == 0:
                logging.info('iter {} and running loss are {}'.format(epoch, running_loss))  # Log the running loss
            if epoch % config.scheduler_interval == 0:
                scheduler.step()  # Step the learning rate scheduler

        # Save the trained generator model
        torch.save(self.model_gen.state_dict(), os.path.join(config.log_dir, 'checkpoints', 'final_checkpoint.pth'))


    def train(self, sensitive_dataloader, config):
        # Check if the dataloader is provided
        if sensitive_dataloader is None:
            return
        
        # Create log directory
        os.mkdir(config.log_dir)
        os.mkdir(os.path.join(config.log_dir, "samples"))
        os.mkdir(os.path.join(config.log_dir, "checkpoints"))
        
        # Calculate noise multiplier for differential privacy
        self.noise_factor = get_noise_multiplier(
            target_epsilon=config.dp.epsilon, 
            target_delta=config.dp.delta, 
            sample_rate=1., 
            epochs=1
        )
        
        # Log the noise factor
        logging.info("The noise factor is {}".format(self.noise_factor))
        
        # Calculate noisy mean embedding
        self.noisy_mean_emb = calc_mean_emb1(
            self.model_ntk, 
            sensitive_dataloader, 
            self.private_num_classes, 
            self.noise_factor, 
            self.device
        )
        
        # Save the noisy mean embedding
        torch.save(self.noisy_mean_emb, os.path.join(config.log_dir, "checkpoints", 'noisy_mean_emb.pt'))
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.Adam(self.model_gen.parameters(), lr=config.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=config.lr_decay)
        
        # Calculate feature length
        mean_v_samp = torch.Tensor([]).to(self.device)
        for p in self.model_ntk.parameters():
            mean_v_samp = torch.cat((mean_v_samp, p.flatten()))
        d = len(mean_v_samp)
        logging.info('Feature Length: {}'.format(d))
        
        # Training loop
        for epoch in range(config.n_iter):
            running_loss = 0.0
            optimizer.zero_grad()  # Reset gradients
            
            # Accumulate gradients over multiple mini-batches
            for accu_step in range(config.n_splits):
                batch_size = config.batch_size // config.n_splits
                gen_labels_numerical = torch.randint(self.private_num_classes, (batch_size,)).to(self.device)
                z = torch.randn(batch_size, self.model_gen.z_dim).to(self.device)
                gen_samples = self.model_gen(z, gen_labels_numerical).reshape(batch_size, -1) / 2 + 0.5
                
                mean_emb2 = torch.zeros((d, self.private_num_classes), device=self.device)
                
                # Compute mean embeddings for generated samples
                for idx in range(gen_samples.shape[0]):
                    mean_v_samp = torch.Tensor([]).to(self.device)
                    f_x = self.model_ntk(gen_samples[idx][None, :])
                    
                    # Compute gradients of the NTK model output
                    f_idx_grad = torch.autograd.grad(f_x, self.model_ntk.parameters(),
                                                    grad_outputs=torch.ones_like(f_x), create_graph=True)
                    for g in f_idx_grad:
                        mean_v_samp = torch.cat((mean_v_samp, g.flatten()))
                    
                    # Normalize and accumulate gradients
                    mean_emb2[:, gen_labels_numerical[idx]] += mean_v_samp / torch.linalg.vector_norm(mean_v_samp)
                
                mean_emb2 = mean_emb2 / batch_size
                
                # Compute loss
                loss = torch.norm(self.noisy_mean_emb - mean_emb2, p=2) ** 2 / config.n_splits
                loss.backward()  # Backpropagate loss
                
            optimizer.step()  # Update model parameters
            
            running_loss += loss.item()
            
            # Log training progress
            if (epoch + 1) % config.log_interval == 0:
                logging.info('iter {} and running loss are {}'.format(epoch, running_loss))
            
            # Adjust learning rate
            if epoch % config.scheduler_interval == 0:
                scheduler.step()
        
        # Save the trained generator model
        torch.save(self.model_gen.state_dict(), os.path.join(config.log_dir, 'checkpoints', 'final_checkpoint.pth'))


    def generate(self, config):
        # Create the directory to store logs and generated data
        os.mkdir(config.log_dir)

        # Synthesize data with uniformly distributed labels using the generator model
        syn_data, syn_labels = synthesize_with_uniform_labels(
            self.model_gen, 
            self.device, 
            gen_batch_size=config.batch_size, 
            n_data=config.data_num, 
            n_labels=self.private_num_classes
        )
        
        # Reshape synthesized data to match the expected input dimensions of the model
        syn_data = syn_data.reshape(syn_data.shape[0], self.c, self.img_size, self.img_size)
        
        # Flatten the labels array
        syn_labels = syn_labels.reshape(-1)
        
        # Save the synthesized data and labels as an .npz file
        np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

        # Prepare a set of images to display, 8 images per class
        show_images = []
        for cls in range(self.private_num_classes):
            # Select the first 8 images for each class
            show_images.append(syn_data[syn_labels == cls][:8])
        
        # Concatenate all selected images into a single array
        show_images = np.concatenate(show_images)
        
        # Save the concatenated images as a grid in a PNG file
        torchvision.utils.save_image(
            torch.from_numpy(show_images), 
            os.path.join(config.log_dir, 'sample.png'), 
            padding=1, 
            nrow=8
        )
        
        # Return the synthesized data and labels
        return syn_data, syn_labels



def synthesize_with_uniform_labels(gen, device, gen_batch_size=1000, n_data=60000, n_labels=10):
    # Set the generator to evaluation mode
    gen.eval()
    
    # Ensure that the total number of data points is divisible by the batch size
    assert n_data % gen_batch_size == 0
    
    # Ensure that the batch size is divisible by the number of labels
    assert gen_batch_size % n_labels == 0
    
    # Calculate the number of iterations needed to generate the required amount of data
    n_iterations = n_data // gen_batch_size

    # Initialize an empty list to store generated data
    data_list = []
    
    # Create a tensor of ordered labels that will be repeated for each iteration
    ordered_labels = torch.repeat_interleave(torch.arange(n_labels), gen_batch_size // n_labels)[:, None].to(device)
    
    # Create a list of label tensors for all iterations
    labels_list = [ordered_labels] * n_iterations

    # Disable gradient calculation to speed up the generation process
    with torch.no_grad():
        for idx in range(n_iterations):
            # Flatten the ordered labels tensor
            y = ordered_labels.view(-1)
            
            # Generate random noise vectors
            z = torch.randn(gen_batch_size, gen.z_dim).to(device)
            
            # Generate samples using the generator model
            gen_samples = gen(z, y).reshape(gen_batch_size, -1) / 2 + 0.5
            
            # Append the generated samples to the data list
            data_list.append(gen_samples)
    
    # Concatenate all generated data and labels into single tensors
    # Convert the tensors to numpy arrays and move them to CPU memory
    return torch.cat(data_list, dim=0).cpu().numpy(), torch.cat(labels_list, dim=0).cpu().numpy()
