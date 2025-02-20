import os
import logging
import torch
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import pickle
import torchvision

from models.DP_Diffusion.model.ema import ExponentialMovingAverage
from models.DP_GAN.utils.util import set_seeds, make_dir, save_checkpoint, sample_random_image_batch, compute_fid, generate_batch, save_img
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor

from models.DP_GAN.generator import Generator
from models.DP_GAN.discriminator import Discriminator

import importlib
opacus = importlib.import_module('opacus')

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

from models.synthesizer import DPSynther

class DPGAN(DPSynther):
    def __init__(self, config, device):
        """
        Initializes the model with given configuration and device settings.

        Parameters:
        - config (object): Configuration object containing various settings and parameters.
        - device (str): Device on which the model will be run ('cpu' or 'cuda').
        """
        super().__init__()
        self.local_rank = config.local_rank  # Local rank of the process
        self.global_rank = config.global_rank  # Global rank of the process
        self.global_size = config.global_size  # Total number of processes

        self.fid_stats = config.fid_stats  # FID statistics for evaluation
        self.ema_rate = config.ema_rate  # Rate for exponential moving average

        self.z_dim = config.Generator.z_dim  # Dimension of the latent space for the generator
        self.img_size = config.img_size  # Size of the images
        self.private_num_classes = config.private_num_classes  # Number of private classes
        self.public_num_classes = config.public_num_classes  # Number of public classes
        label_dim = max(self.private_num_classes, self.public_num_classes)  # Maximum number of classes for labels

        self.config = config  # Store the configuration object
        self.device = 'cuda:%d' % self.local_rank  # Set the device based on local rank

        # Initialize the generator and discriminators
        self.G = Generator(img_size=self.img_size, num_classes=label_dim, **config.Generator).to(self.device)
        self.D = Discriminator(img_size=self.img_size, num_classes=label_dim, **config.Discriminator).to(self.device)
        self.D_copy = Discriminator(img_size=self.img_size, num_classes=label_dim, **config.Discriminator).to(self.device)

        # Load checkpoint if provided
        if config.ckpt is not None:
            state = torch.load(config.ckpt, map_location=self.device)  # Load the checkpoint
            new_state_dict = {}
            for k, v in state['G'].items():
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix from keys
            logging.info(self.G.load_state_dict(new_state_dict, strict=True))  # Load state dict into the generator

            new_state_dict = {}
            for k, v in state['D'].items():
                new_state_dict[k[7:]] = v  # Remove 'module.' prefix from keys
            logging.info(self.D.load_state_dict(new_state_dict, strict=True))  # Load state dict into the discriminator
            logging.info(self.D_copy.load_state_dict(new_state_dict, strict=True))  # Load state dict into the copy of the discriminator

            del state, new_state_dict  # Free up memory

        self.D_copy.eval()  # Set the copy of the discriminator to evaluation mode
        self.ema_G = ExponentialMovingAverage(self.G.parameters(), decay=self.ema_rate)  # Initialize EMA for the generator

        # Log model information if this is the main process
        if self.global_rank == 0:
            model_parameters = filter(lambda p: p.requires_grad, self.G.parameters())  # Filter trainable parameters of the generator
            n_params = sum([np.prod(p.size()) for p in model_parameters])  # Calculate the number of trainable parameters
            logging.info('Number of trainable parameters in G: %d' % n_params)  # Log the number of trainable parameters in the generator

            model_parameters = filter(lambda p: p.requires_grad, self.D.parameters())  # Filter trainable parameters of the discriminator
            n_params = sum([np.prod(p.size()) for p in model_parameters])  # Calculate the number of trainable parameters
            logging.info('Number of trainable parameters in D: %d' % n_params)  # Log the number of trainable parameters in the discriminator
    
    def pretrain(self, public_dataloader, config):
        """
        Pre-train the model using the provided public dataloader and configuration.

        :param public_dataloader: DataLoader for the public dataset.
        :param config: Configuration object containing training parameters.
        """
        if public_dataloader is None:
            return

        # Set random seeds for reproducibility
        set_seeds(self.global_rank, config.seed)

        if self.global_rank == 0:
            logging.info('Number of total epochs: %d' % config.n_epochs)  # Log the total number of epochs

        # Set the CUDA device based on the local rank
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Define directories for saving samples and checkpoints
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        # Create necessary directories if this is the main process
        if self.global_rank == 0:
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)
        dist.barrier()  # Synchronize processes

        # Wrap models with DistributedDataParallel for multi-GPU training
        D = DDP(self.D)
        G = DDP(self.G)
        G.eval()

        # Initialize Exponential Moving Average for the generator
        ema = ExponentialMovingAverage(G.parameters(), decay=self.ema_rate)

        # Set up optimizers based on the configuration
        if config.optim.optimizer == 'Adam':
            optimizerD = torch.optim.Adam(D.parameters(), lr=config.optim.params.d_lr, betas=(config.optim.params.beta1, 0.999))
            optimizerG = torch.optim.Adam(G.parameters(), lr=config.optim.params.g_lr, betas=(config.optim.params.beta1, 0.999))
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize training state
        state = dict(G=G, D=D, emaG=ema, step=0)

        # Create a distributed data loader
        dataset_loader = torch.utils.data.DataLoader(
            dataset=public_dataloader.dataset,
            batch_size=config.batch_size // self.global_size,
            sampler=DistributedSampler(public_dataloader.dataset),
            pin_memory=True,
            drop_last=True,
            num_workers=16
        )

        # Initialize Inception model for FID score calculation
        inception_model = InceptionFeatureExtractor()
        inception_model.model = inception_model.model.to(self.device)

        # Define shapes for snapshot and FID sampling
        snapshot_sampling_shape = (config.snapshot_batch_size, self.z_dim)
        fid_sampling_shape = (config.fid_batch_size, self.z_dim)

        D_steps = 0  # Counter for discriminator steps

        # Training loop
        for epoch in range(config.n_epochs):
            for _, (train_x, train_y) in enumerate(dataset_loader):
                # Preprocess input data
                if len(train_y.shape) == 2:
                    train_x = train_x.to(torch.float32) / 255.
                    train_y = torch.argmax(train_y, dim=1)
                if not config.cond:
                    train_y = torch.zeros_like(train_y)

                real_images = train_x.to(self.device) * 2. - 1.
                real_labels = train_y.to(self.device).long()
                batch_size = real_images.size(0)

                # Generate fake labels and noise
                fake_labels = torch.randint(0, self.public_num_classes, (batch_size,), device=self.device)
                noise = torch.randn((batch_size, self.z_dim), device=self.device)
                fake_images = G(noise, fake_labels)

                # Compute outputs of the discriminator
                real_out = D(real_images, real_labels)
                fake_out = D(fake_images.detach(), fake_labels)

                # Compute and backpropagate discriminator loss
                loss_D = self.d_hinge(real_out, fake_out)
                loss_D.backward()
                optimizerD.step()
                optimizerD.zero_grad(set_to_none=True)

                D_steps += 1

                # Update generator every `d_updates` steps
                if D_steps % config.d_updates == 0:
                    self.d_copy(self.D.parameters(), self.D_copy.parameters())
                    G.train()
                    self.D_copy.zero_grad()
                    optimizerG.zero_grad()

                    batch_size = config.batch_size // self.global_size
                    optimizerD.zero_grad(set_to_none=True)

                    # Generate new fake images
                    fake_labels = torch.randint(0, self.public_num_classes, (batch_size,), device=self.device)
                    noise = torch.randn((batch_size, self.z_dim), device=self.device)
                    fake_images = G(noise, fake_labels)

                    # Compute generator loss
                    output_g = self.D_copy(fake_images, fake_labels)
                    loss_G = self.g_hinge(output_g)
                    loss_G.backward()
                    optimizerG.step()
                    G.eval()

                    state['step'] += 1
                    state['emaG'].update(G.parameters())

                    # Save snapshots and compute FID scores periodically
                    if state['step'] % config.snapshot_freq == 0 and state['step'] >= config.snapshot_threshold:
                        logging.info('Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])
                        with torch.no_grad():
                            ema.store(G.parameters())
                            ema.copy_to(G.parameters())
                            samples = sample_random_image_batch(G, snapshot_sampling_shape, self.device, self.public_num_classes)
                            ema.restore(G.parameters())

                        if self.global_rank == 0:
                            save_img(samples, os.path.join(sample_dir, 'sample_%d.png' % state['step']))

                    if state['step'] % config.fid_freq == 0 and state['step'] >= config.fid_threshold:
                        with torch.no_grad():
                            ema.store(G.parameters())
                            ema.copy_to(G.parameters())
                            fid = compute_fid(config.fid_samples, self.global_size, fid_sampling_shape, G, inception_model, self.fid_stats, self.device, self.public_num_classes)
                            ema.restore(G.parameters())

                        if self.global_rank == 0:
                            logging.info('FID at iteration %d: %.6f' % (state['step'], fid))

                    # Save checkpoints periodically
                    if state['step'] % config.save_freq == 0 and state['step'] >= config.save_threshold and self.global_rank == 0:
                        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                        save_checkpoint(checkpoint_file, state)
                        logging.info('Saving checkpoint at iteration %d' % state['step'])

                    # Log losses periodically
                    if state['step'] % config.log_freq == 0 and self.global_rank == 0:
                        logging.info('Loss D: %.4f, Loss G: %.4f, step: %d' % (loss_D, loss_G, state['step'] + 1))

            if self.global_rank == 0:
                logging.info('%d epochs: is finished' % (epoch + 1))

        # Save final checkpoint
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')

        dist.barrier()

        # Apply EMA to the final generator parameters
        ema.copy_to(self.G.parameters())
        self.ema = ema


    def train(self, sensitive_dataloader, config):
        # Check if the dataloader is None, and if so, return early
        if sensitive_dataloader is None:
            return
        
        # Set random seeds for reproducibility
        set_seeds(self.global_rank, config.seed)

        if self.global_rank == 0:
            logging.info('Number of total epochs: %d' % config.n_epochs)  # Log the total number of epochs
        
        # Set the CUDA device based on the local rank
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Define directories for saving samples and checkpoints
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        # Create necessary directories if this is the main process
        if self.global_rank == 0:
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)
        dist.barrier()  # Synchronize all processes

        # Initialize the discriminator and generator models with DDP (Distributed Data Parallel)
        D = DPDDP(self.D)
        G = DDP(self.G)
        G.eval()  # Set the generator to evaluation mode initially

        # Initialize exponential moving average for the generator parameters
        ema = ExponentialMovingAverage(G.parameters(), decay=self.ema_rate)

        # Initialize optimizers based on the configuration
        if config.optim.optimizer == 'Adam':
            optimizerD = torch.optim.Adam(D.parameters(), lr=config.optim.params.d_lr, betas=(config.optim.params.beta1, 0.999))
            optimizerG = torch.optim.Adam(G.parameters(), lr=config.optim.params.g_lr, betas=(config.optim.params.beta1, 0.999))
        else:
            raise NotImplementedError("Only Adam optimizer is supported currently")

        # Initialize the training state
        state = dict(G=G, D=D, emaG=ema, step=0)

        # Initialize the privacy engine
        privacy_engine = PrivacyEngine()
        
        # Prepare privacy accounting history if differential privacy is enabled
        if config.dp.sdq is None:
            account_history = None
            alpha_history = None
        else:
            account_history = [tuple(item) for item in config.dp.privacy_history]
            if config.dp.alpha_num == 0:
                alpha_history = None
            else:
                alpha = np.arange(config.dp.alpha_num) / config.dp.alpha_num
                alpha = alpha * (config.dp.alpha_max - config.dp.alpha_min)
                alpha += config.dp.alpha_min 
                alpha_history = list(alpha)

        # Apply differential privacy to the discriminator
        D, optimizerD, dataset_loader = privacy_engine.make_private_with_epsilon(
            module=D,
            optimizer=optimizerD,
            data_loader=sensitive_dataloader,
            target_delta=config.dp.delta,
            target_epsilon=config.dp.epsilon,
            epochs=config.n_epochs,
            max_grad_norm=config.dp.max_grad_norm,
            noise_multiplicity=1,
            account_history=account_history,
            alpha_history=alpha_history,
        )

        # Initialize the Inception model for computing FID scores
        inception_model = InceptionFeatureExtractor()
        inception_model.model = inception_model.model.to(self.device)
        
        # Define shapes for sampling images
        snapshot_sampling_shape = (config.snapshot_batch_size, self.z_dim)
        fid_sampling_shape = (config.fid_batch_size, self.z_dim)

        D_steps = 0  # Counter for discriminator steps

        # Training loop over epochs
        for epoch in range(config.n_epochs):
            with BatchMemoryManager(
                    data_loader=dataset_loader,
                    max_physical_batch_size=config.dp.max_physical_batch_size,
                    optimizer=optimizerD,
                    n_splits=config.n_splits if config.n_splits > 0 else None) as memory_safe_data_loader:

                # Iterate over batches in the dataset
                for _, (train_x, train_y) in enumerate(memory_safe_data_loader):
                    # Preprocess the input data
                    if len(train_y.shape) == 2:
                        train_x = train_x.to(torch.float32) / 255.
                        train_y = torch.argmax(train_y, dim=1)
                    
                    real_images = train_x.to(self.device) * 2. - 1.
                    real_labels = train_y.to(self.device).long()
                    batch_size = real_images.size(0)

                    # Generate fake labels and noise for generating fake images
                    fake_labels = torch.randint(0, self.private_num_classes, (batch_size, ), device=self.device)
                    noise = torch.randn((batch_size, self.z_dim), device=self.device)
                    fake_images = G(noise, fake_labels)

                    # Compute discriminator outputs for real and fake images
                    real_out = D(real_images, real_labels)
                    fake_out = D(fake_images.detach(), fake_labels)

                    # Compute and backpropagate the discriminator loss
                    loss_D = self.d_hinge(real_out, fake_out)
                    loss_D.backward()
                    optimizerD.step()
                    optimizerD.zero_grad(set_to_none=True)

                    # Update the discriminator step counter
                    if not optimizerD._is_last_step_skipped:
                        D_steps += 1

                        # Train the generator every `d_updates` discriminator steps
                        if D_steps % config.d_updates == 0:
                            self.d_copy(self.D.parameters(), self.D_copy.parameters())
                            G.train()
                            self.D_copy.zero_grad()
                            optimizerG.zero_grad()
                            batch_size = config.batch_size // config.n_splits // self.global_size
                            for _ in range(config.n_splits):
                                optimizerD.zero_grad(set_to_none=True)
                                fake_labels = torch.randint(0, self.private_num_classes, (batch_size, ), device=self.device)
                                noise = torch.randn((batch_size, self.z_dim), device=self.device)
                                fake_images = G(noise, fake_labels)
                                output_g = self.D_copy(fake_images, fake_labels)
                                loss_G = self.g_hinge(output_g) / config.n_splits
                                loss_G.backward()
                            optimizerG.step()
                            G.eval()

                            state['step'] += 1
                            state['emaG'].update(G.parameters())

                            # Save snapshots and compute FID scores periodically
                            if state['step'] % config.snapshot_freq == 0 and state['step'] >= config.snapshot_threshold:
                                logging.info(
                                    'Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])

                                with torch.no_grad():
                                    ema.store(G.parameters())
                                    ema.copy_to(G.parameters())
                                    samples = sample_random_image_batch(G, snapshot_sampling_shape, self.device, self.private_num_classes)
                                    ema.restore(G.parameters())
                                
                                if self.global_rank == 0:
                                    save_img(samples, os.path.join(sample_dir, 'sample_%d.png' % state['step']))
                                dist.barrier()
                            if state['step'] % config.fid_freq == 0 and state['step'] >= config.fid_threshold:
                                with torch.no_grad():
                                    ema.store(G.parameters())
                                    ema.copy_to(G.parameters())
                                    fid = compute_fid(config.fid_samples, self.global_size, fid_sampling_shape, G, inception_model, self.fid_stats, self.device, self.private_num_classes)
                                    ema.restore(G.parameters())

                                    if self.global_rank == 0:
                                        logging.info('FID at iteration %d: %.6f' % (state['step'], fid))
                                dist.barrier()
                            if state['step'] % config.save_freq == 0 and state['step'] >= config.save_threshold and self.global_rank == 0:
                                checkpoint_file = os.path.join(
                                    checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                                save_checkpoint(checkpoint_file, state)
                                logging.info(
                                    'Saving  checkpoint at iteration %d' % state['step'])
                            dist.barrier()
                            if state['step'] % config.log_freq == 0 and self.global_rank == 0:
                                logging.info('Loss D: %.4f, Loss G: %.4f, step: %d' %
                                            (loss_D, loss_G, state['step'] + 1))
                if self.global_rank == 0:
                    logging.info('Eps-value after %d epochs: %.4f' % (epoch + 1, privacy_engine.get_epsilon(config.dp.delta)))

        # Save the final checkpoint and compute final FID score
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')
        dist.barrier()

        with torch.no_grad():
            ema.store(G.parameters())
            ema.copy_to(G.parameters())
            samples = sample_random_image_batch(G, snapshot_sampling_shape, self.device, self.private_num_classes)
            fid = compute_fid(config.final_fid_samples, self.global_size, fid_sampling_shape, G, inception_model,
                            self.fid_stats, self.device, self.private_num_classes)
            ema.restore(G.parameters())

        if self.global_rank == 0:
            make_dir(os.path.join(sample_dir, 'final'))
            save_img(samples, os.path.join(os.path.join(sample_dir, 'final'), 'sample.png'))
            logging.info('Final FID %.6f' % (fid))
        dist.barrier()

        self.ema = ema


    def generate(self, config):
        # Log the start of the generation process and the number of samples to be generated
        logging.info("Start to generate {} samples".format(config.data_num))
        
        # Define the working directory and sample directory paths
        workdir = os.path.join(config.log_dir, 'samples{}_acc'.format(config.data_num))
        sample_dir = os.path.join(workdir, 'samples/')
        
        # Create necessary directories if the current rank is 0
        if self.global_rank == 0:
            make_dir(config.log_dir)
            make_dir(workdir)
            make_dir(sample_dir)
        
        # Synchronize all processes after directory creation
        dist.barrier()

        # Define the shape for the sampling batch
        sampling_shape = (config.batch_size, self.z_dim)

        # Wrap the generator model with DistributedDataParallel for multi-GPU support
        G = DDP(self.G)
        
        # Copy the parameters from the EMA model to the generator
        self.ema.copy_to(G.parameters())

        # Initialize lists to store synthetic data and labels if the current rank is 0
        if self.global_rank == 0:
            syn_data = []
            syn_labels = []

        # Generate batches of synthetic data
        for _ in range(config.data_num // (sampling_shape[0] * self.global_size) + 1):
            # Generate a batch of data and labels
            x, y = generate_batch(G, sampling_shape, self.device, self.private_num_classes)
            
            # Synchronize all processes after generating a batch
            dist.barrier()
            
            if self.global_rank == 0:
                # Prepare lists to gather data and labels from all processes
                gather_x = [torch.zeros_like(x) for _ in range(self.global_size)]
                gather_y = [torch.zeros_like(y) for _ in range(self.global_size)]
            else:
                gather_x = None
                gather_y = None
            
            # Gather data and labels from all processes
            dist.gather(x, gather_x)
            dist.gather(y, gather_y)
            
            if self.global_rank == 0:
                # Concatenate gathered data and labels and move them to CPU memory
                syn_data.append(torch.cat(gather_x).detach().cpu().numpy())
                syn_labels.append(torch.cat(gather_y).detach().cpu().numpy())

        # If the current rank is 0, finalize the generation process
        if self.global_rank == 0:
            logging.info("Generation Finished!")
            
            # Concatenate all generated data and labels
            syn_data = np.concatenate(syn_data)
            syn_labels = np.concatenate(syn_labels)

            # Save the generated data and labels to a file
            np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

            # Prepare images to display
            show_images = []
            for cls in range(self.private_num_classes):
                show_images.append(syn_data[syn_labels == cls][:8])
            show_images = np.concatenate(show_images)
            
            # Save the images to a file
            torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
            
            # Return the generated data and labels
            return syn_data, syn_labels
        else:
            # Return None for non-zero ranks
            return None, None

    def d_hinge(self, d_logit_real, d_logit_fake):
        # Calculate the hinge loss for the discriminator
        return torch.mean(F.relu(1. - d_logit_real)) + torch.mean(F.relu(1. + d_logit_fake))

    def g_hinge(self, d_logit_fake):
        # Calculate the hinge loss for the generator
        return -torch.mean(d_logit_fake)

    def d_copy(self, params1, params2):
        # Copy the parameters from one set to another
        for param1, param2 in zip(params1, params2):
            param2.data.copy_(param1.data)
