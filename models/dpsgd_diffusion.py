import os
import logging
import torch
import copy
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import pickle
import torchvision
import tqdm

from models.DP_Diffusion.model.ncsnpp import NCSNpp
from models.DP_Diffusion.utils.util import set_seeds, make_dir, save_checkpoint, sample_random_image_batch, compute_fid
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from models.DP_Diffusion.model.ema import ExponentialMovingAverage
from models.DP_Diffusion.score_losses import EDMLoss, VPSDELoss, VESDELoss, VLoss
from models.DP_Diffusion.denoiser import EDMDenoiser, VPSDEDenoiser, VESDEDenoiser, VDenoiser
from models.DP_Diffusion.samplers import ddim_sampler, edm_sampler
from models.DP_Diffusion.generate_base import generate_batch

import importlib
opacus = importlib.import_module('opacus')

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

from models.synthesizer import DPSynther

class DP_Diffusion(DPSynther):
    def __init__(self, config, device):
        """
        Initializes the model with the provided configuration and device settings.

        Args:
            config (Config): Configuration object containing all necessary parameters.
            device (str): Device to use for computation (e.g., 'cuda:0').
        """
        super().__init__()
        self.local_rank = config.local_rank  # Local rank of the process
        self.global_rank = config.global_rank  # Global rank of the process
        self.global_size = config.global_size  # Total number of processes

        self.denoiser_name = config.denoiser_name  # Name of the denoiser to be used
        self.denoiser_network = config.denoiser_network  # Network architecture for the denoiser
        self.ema_rate = config.ema_rate  # Rate for exponential moving average
        self.network = config.network  # Configuration for the network
        self.sampler = config.sampler  # Sampler configuration
        self.sampler_fid = config.sampler_fid  # FID sampler configuration
        self.sampler_acc = config.sampler_acc  # Accuracy sampler configuration
        self.fid_stats = config.fid_stats  # FID statistics configuration

        self.config = config  # Store the entire configuration
        self.device = 'cuda:%d' % self.local_rank  # Set the device based on local rank

        self.private_num_classes = config.private_num_classes  # Number of private classes
        self.public_num_classes = config.public_num_classes  # Number of public classes
        label_dim = max(self.private_num_classes, self.public_num_classes)  # Determine the maximum label dimension
        self.network.label_dim = label_dim  # Set the label dimension for the network

        # Initialize the denoiser based on the specified name and network
        if self.denoiser_name == 'edm':
            if self.denoiser_network == 'song':
                self.model = EDMDenoiser(NCSNpp(**self.network).to(self.device))  # Initialize EDM denoiser with NCSNpp network
            else:
                raise NotImplementedError("Network type not supported for EDM denoiser")
        elif self.denoiser_name == 'vpsde':
            if self.denoiser_network == 'song':
                self.model = VPSDEDenoiser(self.config.beta_min, self.config.beta_max - self.config.beta_min,
                                        self.config.scale, NCSNpp(**self.network).to(self.device))  # Initialize VPSDE denoiser with NCSNpp network
            else:
                raise NotImplementedError("Network type not supported for VPSDE denoiser")
        elif self.denoiser_name == 'vesde':
            if self.denoiser_network == 'song':
                self.model = VESDEDenoiser(NCSNpp(**self.network).to(self.device))  # Initialize VESDE denoiser with NCSNpp network
            else:
                raise NotImplementedError("Network type not supported for VESDE denoiser")
        elif self.denoiser_name == 'v':
            if self.denoiser_network == 'song':
                self.model = VDenoiser(NCSNpp(**self.network).to(self.device))  # Initialize V denoiser with NCSNpp network
            else:
                raise NotImplementedError("Network type not supported for V denoiser")
        else:
            raise NotImplementedError("Denoiser name not recognized")

        self.model = self.model.to(self.local_rank)  # Move the model to the specified device
        self.model.train()  # Set the model to training mode
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_rate)  # Initialize EMA for the model parameters

        # Load checkpoint if provided
        if config.ckpt is not None:
            state = torch.load(config.ckpt, map_location=self.device)  # Load the checkpoint
            new_state_dict = {}
            for k, v in state['model'].items():
                new_state_dict[k[7:]] = v  # Adjust the keys to match the model's state dictionary
            logging.info(self.model.load_state_dict(new_state_dict, strict=True))  # Load the state dictionary into the model
            self.ema.load_state_dict(state['ema'])  # Load the EMA state dictionary
            del state, new_state_dict  # Clean up memory

    
    def pretrain(self, public_dataloader, config):
        """
        Pre-trains the model using the provided public dataloader and configuration.

        Args:
            public_dataloader (DataLoader): The dataloader for the public dataset.
            config (dict): Configuration dictionary containing various settings and hyperparameters.
        """
        if public_dataloader is None:
            # If no public dataloader is provided, return.
            return
        
        # Set the number of classes in the loss function to the number of private classes.
        config.loss.n_classes = self.public_num_classes
        if config.cond:
            # If conditional training is enabled, set the label unconditioning probability to 0.1. Conditional training with a low unconditioning probability usually performs better.
            config.loss['label_unconditioning_prob'] = 0.1
        else:
            # If conditional training is disabled, set the label unconditioning probability to 1.0.
            config.loss['label_unconditioning_prob'] = 1.0

        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Define directories for storing samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if the global rank is 0.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        # Wrap the model with DistributedDataParallel (DDP) for distributed training.
        model = DDP(self.model, device_ids=[self.local_rank])
        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the training state.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        if self.global_rank == 0:
            # Log the number of trainable parameters and training details if the global rank is 0.
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            n_params = sum([np.prod(p.size()) for p in model_parameters])
            logging.info('Number of trainable parameters in model: %d' % n_params)
            logging.info('Number of total epochs: %d' % config.n_epochs)
            logging.info('Starting training at step %d' % state['step'])
        dist.barrier()

        # Create a distributed data loader for the public dataset.
        dataset_loader = torch.utils.data.DataLoader(
            dataset=public_dataloader.dataset, 
            batch_size=config.batch_size // self.global_size, 
            sampler=DistributedSampler(public_dataloader.dataset), 
            pin_memory=True, 
            drop_last=True, 
            num_workers=16
        )

        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function version not supported")

        # Initialize the Inception model for feature extraction.
        inception_model = InceptionFeatureExtractor()
        inception_model.model = inception_model.model.to(self.device)

        # Define the sampler function for generating images.
        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # Define the shape of the batches for sampling and FID computation.
        snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
                                self.network.num_in_channels, 
                                self.network.image_size, 
                                self.network.image_size)
        fid_sampling_shape = (self.sampler.fid_batch_size, 
                            self.network.num_in_channels, 
                            self.network.image_size, 
                            self.network.image_size)

        # Training loop over the specified number of epochs.
        for epoch in range(config.n_epochs):
            dataset_loader.sampler.set_epoch(epoch)
            for _, (train_x, train_y) in enumerate(dataset_loader):

                # Save snapshots and checkpoints at specified intervals.
                if state['step'] % config.snapshot_freq == 0 and state['step'] >= config.snapshot_threshold and self.global_rank == 0:
                    logging.info('Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])

                    model.eval()
                    with torch.no_grad():
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        sample_random_image_batch(snapshot_sampling_shape, sampler, os.path.join(
                            sample_dir, 'iter_%d' % state['step']), self.device, self.private_num_classes)
                        ema.restore(model.parameters())
                    model.train()

                    save_checkpoint(os.path.join(checkpoint_dir, 'snapshot_checkpoint.pth'), state)
                dist.barrier()

                # Compute FID at specified intervals.
                if state['step'] % config.fid_freq == 0 and state['step'] >= config.fid_threshold:
                    model.eval()
                    with torch.no_grad():
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        fid = compute_fid(config.fid_samples, self.global_size, fid_sampling_shape, sampler, inception_model, self.fid_stats, self.device, self.private_num_classes)
                        ema.restore(model.parameters())

                        if self.global_rank == 0:
                            logging.info('FID at iteration %d: %.6f' % (state['step'], fid))
                    model.train()
                dist.barrier()

                # Save checkpoints at specified intervals.
                if state['step'] % config.save_freq == 0 and state['step'] >= config.save_threshold and self.global_rank == 0:
                    checkpoint_file = os.path.join(
                        checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                    save_checkpoint(checkpoint_file, state)
                    logging.info('Saving checkpoint at iteration %d' % state['step'])
                dist.barrier()

                # Prepare the input data for training.
                if len(train_y.shape) == 2:
                    train_x = train_x.to(torch.float32) / 255.
                    train_y = torch.argmax(train_y, dim=1)
                train_x, train_y = train_x.to(self.device) * 2. - 1., train_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean(loss_fn(model, train_x, train_y))
                loss.backward()
                optimizer.step()

                # Log the loss at specified intervals.
                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('Loss: %.4f, step: %d' % (loss, state['step'] + 1))
                dist.barrier()

                state['step'] += 1
                state['ema'].update(model.parameters())
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))

        # Save the final checkpoint.
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')
        dist.barrier()

        # Apply the EMA weights to the model and store the EMA object.
        ema.copy_to(self.model.parameters())
        self.ema = ema

        # Clean up the model and free GPU memory.
        del model
        torch.cuda.empty_cache()


    def train(self, sensitive_dataloader, config):
        """
        Trains the model using the provided sensitive data loader and configuration.

        Args:
            sensitive_dataloader (DataLoader): DataLoader containing the sensitive data.
            config (Config): Configuration object containing various settings for training.

        Returns:
            None
        """
        if sensitive_dataloader is None or config.n_epochs == 0:
            # If the dataloader is not provided or the number of epochs is zero, exit early.
            return
        
        set_seeds(self.global_rank, config.seed)
        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank
        # Set the number of classes for the loss function.
        config.loss.n_classes = self.private_num_classes

        # Define directories for saving samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if this is the main process.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)
        
        if config.partly_finetune:
            # If partial fine-tuning is enabled, freeze certain layers.
            trainable_parameters = []
            for name, param in self.model.named_parameters():
                layer_idx = int(name.split('.')[2])
                if layer_idx > 3 and 'NIN' not in name:
                    param.requires_grad = False
                    if self.global_rank == 0:
                        logging.info('{} is frozen'.format(name))
                else:
                    trainable_parameters.append(param)
        else:
            # Otherwise, all parameters are trainable.
            trainable_parameters = self.model.parameters()

        # Wrap the model with DPDDP for distributed training with differential privacy.
        model = DPDDP(self.model)
        # Initialize Exponential Moving Average (EMA) for model parameters.
        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(trainable_parameters, **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the state dictionary to keep track of the training process.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        if self.global_rank == 0:
            # Log the number of trainable parameters and other training details.
            model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            n_params = sum([np.prod(p.size()) for p in model_parameters])
            logging.info('Number of trainable parameters in model: %d' % n_params)
            logging.info('Number of total epochs: %d' % config.n_epochs)
            logging.info('Starting training at step %d' % state['step'])

        # Initialize the Privacy Engine for differential privacy.
        privacy_engine = PrivacyEngine()
        if config.dp.privacy_history is None:
            account_history = None
        else:
            account_history = [tuple(item) for item in config.dp.privacy_history]

        # Make the model, optimizer, and data loader private.
        model, optimizer, dataset_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=sensitive_dataloader,
            target_delta=config.dp.delta,
            target_epsilon=config.dp.epsilon,
            epochs=config.n_epochs,
            max_grad_norm=config.dp.max_grad_norm,
            noise_multiplicity=config.loss.n_noise_samples,
            account_history=account_history,
        )

        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function not supported")

        # Initialize the Inception model for feature extraction.
        inception_model = InceptionFeatureExtractor()
        inception_model.model = inception_model.model.to(self.device)

        # Define the sampler function for generating images.
        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # Define the shapes for sampling images.
        snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
                                self.network.num_in_channels, self.network.image_size, self.network.image_size)
        fid_sampling_shape = (self.sampler.fid_batch_size, self.network.num_in_channels,
                            self.network.image_size, self.network.image_size)

        # Start the training loop.
        for epoch in range(config.n_epochs):
            with BatchMemoryManager(
                    data_loader=dataset_loader,
                    max_physical_batch_size=config.dp.max_physical_batch_size,
                    optimizer=optimizer,
                    n_splits=config.n_splits if config.n_splits > 0 else None) as memory_safe_data_loader:

                for _, (train_x, train_y) in enumerate(memory_safe_data_loader):
                    if state['step'] % config.snapshot_freq == 0 and state['step'] >= config.snapshot_threshold and self.global_rank == 0:
                        # Save a snapshot checkpoint and sample a batch of images.
                        logging.info(
                            'Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])

                        model.eval()
                        with torch.no_grad():
                            ema.store(model.parameters())
                            ema.copy_to(model.parameters())
                            sample_random_image_batch(snapshot_sampling_shape, sampler, os.path.join(
                                sample_dir, 'iter_%d' % state['step']), self.device, self.private_num_classes)
                            ema.restore(model.parameters())
                        model.train()

                        save_checkpoint(os.path.join(
                            checkpoint_dir, 'snapshot_checkpoint.pth'), state)
                    dist.barrier()

                    if state['step'] % config.fid_freq == 0 and state['step'] >= config.fid_threshold:
                        # Compute FID score and log it.
                        model.eval()
                        with torch.no_grad():
                            ema.store(model.parameters())
                            ema.copy_to(model.parameters())
                            fid = compute_fid(config.fid_samples, self.global_size, fid_sampling_shape, sampler, inception_model, self.fid_stats, self.device, self.private_num_classes)
                            ema.restore(model.parameters())

                            if self.global_rank == 0:
                                logging.info('FID at iteration %d: %.6f' % (state['step'], fid))
                            dist.barrier()
                        model.train()

                    if state['step'] % config.save_freq == 0 and state['step'] >= config.save_threshold and self.global_rank == 0:
                        # Save a checkpoint at regular intervals.
                        checkpoint_file = os.path.join(
                            checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                        save_checkpoint(checkpoint_file, state)
                        logging.info(
                            'Saving checkpoint at iteration %d' % state['step'])
                    dist.barrier()

                    if len(train_y.shape) == 2:
                        # Preprocess the input data.
                        train_x = train_x.to(torch.float32) / 255.
                        train_y = torch.argmax(train_y, dim=1)
                    
                    x = train_x.to(self.device) * 2. - 1.
                    y = train_y.to(self.device).long()

                    # Perform a forward pass and backpropagation.
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.mean(loss_fn(model, x, y))
                    loss.backward()
                    optimizer.step()

                    if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                        # Log the loss at regular intervals.
                        logging.info('Loss: %.4f, step: %d' %
                                    (loss, state['step'] + 1))
                    dist.barrier()

                    state['step'] += 1
                    if not optimizer._is_last_step_skipped:
                        state['ema'].update(model.parameters())

                # Log the epsilon value after each epoch.
                logging.info('Eps-value after %d epochs: %.4f' %
                            (epoch + 1, privacy_engine.get_epsilon(config.dp.delta)))

        if self.global_rank == 0:
            # Save the final checkpoint.
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')
        dist.barrier()

        # Update the EMA.
        self.ema = ema


    def generate(self, config):
        # Log the start of the generation process with the number of samples to be generated
        logging.info("start to generate {} samples".format(config.data_num))
        
        # Ensure the log directory exists if this is the main process (global_rank == 0)
        if self.global_rank == 0 and not os.path.exists(config.log_dir):
            make_dir(config.log_dir)
        
        # Synchronize all processes
        dist.barrier()

        # Define the shape for the sampling batch
        sampling_shape = (config.batch_size, self.network.num_in_channels, self.network.image_size, self.network.image_size)

        # Wrap the model with DistributedDataParallel for multi-GPU training
        model = DDP(self.model)
        model.eval()  # Set the model to evaluation mode
        
        # Copy the exponential moving average parameters to the model
        self.ema.copy_to(model.parameters())

        # Define a function to handle different types of samplers
        def sampler_acc(x, y=None):
            if self.sampler_acc.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler_acc)
            elif self.sampler_acc.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler_acc)
            else:
                raise NotImplementedError("Sampler type not supported")

        # Initialize lists to store synthetic data and labels if this is the main process
        if self.global_rank == 0:
            syn_data = []
            syn_labels = []

        # Loop to generate the required number of samples
        for _ in range(config.data_num // (sampling_shape[0] * self.global_size) + 1):
            # Generate a batch of samples and labels
            x, y = generate_batch(sampler_acc, sampling_shape, self.device, self.private_num_classes, self.private_num_classes)
            
            # Synchronize all processes
            dist.barrier()
            
            # Prepare tensors for gathering results from all processes
            if self.global_rank == 0:
                gather_x = [torch.zeros_like(x) for _ in range(self.global_size)]
                gather_y = [torch.zeros_like(y) for _ in range(self.global_size)]
            else:
                gather_x = None
                gather_y = None
            
            # Gather the generated samples and labels from all processes
            dist.gather(x, gather_x)
            dist.gather(y, gather_y)
            
            # If this is the main process, collect the gathered data
            if self.global_rank == 0:
                syn_data.append(torch.cat(gather_x).detach().cpu().numpy())
                syn_labels.append(torch.cat(gather_y).detach().cpu().numpy())

        # If this is the main process, finalize the generation process
        if self.global_rank == 0:
            logging.info("Generation Finished!")
            
            # Concatenate all collected synthetic data and labels
            syn_data = np.concatenate(syn_data)[:config.data_num]
            syn_labels = np.concatenate(syn_labels)[:config.data_num]
            
            # Save the synthetic data and labels to a .npz file
            np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)
            
            # Prepare images to display
            show_images = []
            for cls in range(self.private_num_classes):
                show_images.append(syn_data[syn_labels==cls][:8])
            show_images = np.concatenate(show_images)
            
            # Save the sample images to a PNG file
            torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
            
            # Return the synthetic data and labels
            return syn_data, syn_labels
        else:
            # Return None for non-main processes
            return None, None
