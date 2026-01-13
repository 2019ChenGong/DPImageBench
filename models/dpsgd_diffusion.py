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
import random

from models.DP_Diffusion.model.ncsnpp import NCSNpp
from models.DP_Diffusion.utils.util import set_seeds, make_dir, save_checkpoint, sample_random_image_batch, compute_fid
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from models.DP_Diffusion.model.ema import ExponentialMovingAverage
from models.DP_Diffusion.score_losses import EDMLoss, VPSDELoss, VESDELoss, VLoss
from models.DP_Diffusion.denoiser import EDMDenoiser, VPSDEDenoiser, VESDEDenoiser, VDenoiser
from models.DP_Diffusion.samplers import ddim_sampler, edm_sampler
from models.DP_Diffusion.generate_base import generate_batch, generate_batch_grad
from models.dp_merf import DP_MERF as Freq_Model
from torch.utils.data import random_split, TensorDataset, Dataset, DataLoader, ConcatDataset

from models.DP_MERF.rff_mmd_approx import data_label_embedding, get_rff_mmd_loss, noisy_dataset_embedding
from data.dataset_loader import CentralDataset


import importlib
opacus = importlib.import_module('opacus')

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

from models.synthesizer import DPSynther

class DP_Diffusion(DPSynther):
    def __init__(self, config, device, all_config):
        """
        Initializes the model with the provided configuration and device settings.

        Args:
            config (Config): Configuration object containing all necessary parameters.
            device (str): Device to use for computation (e.g., 'cuda:0').
        """
        super().__init__()
        # self.local_rank = config.local_rank  # Local rank of the process
        # self.global_rank = config.global_rank  # Global rank of the process
        # self.all_config = all_config
        # self.device = 'cuda:%d' % self.local_rank  # Set the device based on local rank

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
        self.all_config = all_config
        self.device = 'cuda:%d' % self.local_rank  # Set the device based on local rank

        self.private_num_classes = config.private_num_classes  # Number of private classes
        self.public_num_classes = config.public_num_classes  # Number of public classes
        label_dim = max(self.private_num_classes, self.public_num_classes)  # Determine the maximum label dimension
        if config.ckpt is not None:
            state = torch.load(config.ckpt, map_location=self.device)
            for k, v in state['model'].items():
                label_dim = v.shape[0]-1
                break
        self.network.label_dim = label_dim  # Set the label dimension for the network

        if 'mode' in self.all_config.pretrain and self.all_config.pretrain.mode != 'time':
            self.freq_model = Freq_Model(self.all_config.model.freq, self.device, self.all_config.train.sigma_sensitivity_ratio)

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
        self.ema = None

        # Load checkpoint if provided
        self.optimizer_state = None
        if config.ckpt is not None:
            state = torch.load(config.ckpt, map_location=self.device)  # Load the checkpoint
            new_state_dict = {}
            for k, v in state['model'].items():
                new_state_dict[k[7:]] = v  # Adjust the keys to match the model's state dictionary
            logging.info(self.model.load_state_dict(new_state_dict, strict=True))  # Load the state dictionary into the model
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_rate)
            # try:
            #     self.ema.load_state_dict(state['ema'])  # Load the EMA state dictionary
            #     self.optimizer_state = state['optimizer']
            # except:
            #     pass
            del state, new_state_dict  # Clean up memory

        self.is_pretrain = True  # Flag to indicate pretraining status

    def pretrain(self, public_dataloader, config, run=False):
        """
        Pre-trains the model using the provided public dataloader and configuration.

        Args:
            public_dataloader (DataLoader): The dataloader for the public dataset.
            config (dict): Configuration dictionary containing various settings and hyperparameters.
        """
        if public_dataloader is None or config.n_epochs == 0:
            # If no public dataloader is provided, set pretraining flag to False and return.
            self.is_pretrain = False
            return
        
        # Set the number of classes in the loss function to the number of private classes.
        config.loss.n_classes = self.private_num_classes
        if config.cond:
            # If conditional training is enabled, set the label unconditioning probability.
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
            num_workers=0
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
                    logging.info('Loss: %.4f, step: %d' % (loss.item(), state['step'] + 1))
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
        # self.ema = ema

        # Clean up the model and free GPU memory.
        del model
        torch.cuda.empty_cache()
    
    def pretrain_freq(self, sensitive_dataloader, config):
        """
        Pre-trains the model using the provided public dataloader and configuration.

        Args:
            public_dataloader (DataLoader): The dataloader for the public dataset.
            config (dict): Configuration dictionary containing various settings and hyperparameters.
        """

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

        self.noise_factor = self.all_config.train.freq.dp.sigma
        logging.info("The noise factor is {}".format(self.noise_factor))
        
        n_data = len(sensitive_dataloader.dataset)  # Number of data points in the sensitive dataset
        rff_sigma = [float(sig) for sig in self.freq_model.rff_sigma.split(',')]
        if self.global_rank == 0:
            _, w_freq = get_rff_mmd_loss(self.freq_model.n_feat, self.freq_model.d_rff, rff_sigma[0], self.local_rank, self.freq_model.private_num_classes, self.noise_factor, sensitive_dataloader.batch_size, self.freq_model.mmd_type)

            noisy_emb = noisy_dataset_embedding(sensitive_dataloader, w_freq, self.freq_model.d_rff, self.local_rank, self.freq_model.private_num_classes, self.noise_factor, self.freq_model.mmd_type, pca_vecs=None, cond=True)
            torch.save({'w_freq': w_freq.w.cpu(), 'noisy_emb': noisy_emb.cpu()}, os.path.join(config.log_dir, 'freq_cache.pth'))

 
        dist.barrier()
        merf_cache = torch.load(os.path.join(config.log_dir, 'freq_cache.pth'))
        w_freq, noisy_emb = merf_cache['w_freq'].to(self.local_rank), merf_cache['noisy_emb'].to(self.local_rank)
        from collections import namedtuple
        rff_param_tuple = namedtuple('rff_params', ['w', 'b'])
        w_freq_param = rff_param_tuple(w=w_freq, b=None)

        def rff_mmd_loss(gen_enc, gen_labels):
            gen_emb = data_label_embedding(gen_enc, gen_labels, w_freq_param, self.freq_model.mmd_type)
            return torch.sum((noisy_emb - gen_emb) ** 2)

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

        # Initialize the Inception model for feature extraction.
        inception_model = InceptionFeatureExtractor()
        inception_model.model = inception_model.model.to(self.device)

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
        n_iter = n_data // config.batch_size
        for epoch in range(config.n_epochs):
            for batch_idx in range(n_iter):
                # Prepare the input data for training.
                gen_samples, gen_y = generate_batch_grad(sampler, (config.batch_size // self.global_size, self.network.num_in_channels, self.network.image_size, self.network.image_size), 
                                            self.device, self.private_num_classes, self.private_num_classes)
                gen_one_hots = torch.nn.functional.one_hot(gen_y, num_classes=self.private_num_classes)
                gen_samples = gen_samples.reshape(config.batch_size // self.global_size, -1)
                loss = rff_mmd_loss(gen_samples, gen_one_hots)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Log the loss at specified intervals.
                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('Loss: %.4f, step: %d' % (loss.item(), state['step'] + 1))
                dist.barrier()

                state['step'] += 1
                state['ema'].update(model.parameters())
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))
            
            # Save snapshots and checkpoints at specified intervals.
            if self.global_rank == 0:
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
        
    def warm_up(self, sensitive_train_loader, config):
        time_set = CentralDataset(sensitive_train_loader.dataset, num_classes=self.all_config.sensitive_data.n_classes, **self.all_config.public_data.central)
        time_dataloader = torch.utils.data.DataLoader(dataset=time_set, shuffle=True, drop_last=True, batch_size=self.all_config.pretrain.batch_size_time, num_workers=0)
        if time_set.privacy_history[0] != 0:
            config.dp['privacy_history'] = [time_set.privacy_history]
        if self.global_rank == 0:
            logging.info("Additional privacy cost: {}".format(str(time_set.privacy_history)))
        if 'auxiliary' not in self.all_config.pretrain.mode and self.all_config.pretrain.mode != 'time':
            if self.global_rank == 0:
                # freq_model = Freq_Model(self.all_config.model.merf, self.device, self.all_config.train.sigma_sensitivity_ratio)
                if 'start_time' in self.all_config.model.freq:
                    self.freq_model.train(sensitive_train_loader, self.all_config.train.freq, start_images=time_set.central_x, start_labels=torch.tensor(time_set.central_y).long())
                else:
                    self.freq_model.train(sensitive_train_loader, self.all_config.train.freq)
                syn_data, syn_labels = self.freq_model.generate(self.all_config.gen.freq)
            dist.barrier()
            # return config
            syn = np.load(os.path.join(self.all_config.gen.freq.log_dir, 'gen.npz'))
            syn_data, syn_labels = syn["x"], syn["y"]
            freq_train_set = TensorDataset(torch.from_numpy(syn_data).float(), torch.from_numpy(syn_labels).long())
            freq_train_loader = DataLoader(dataset=freq_train_set, shuffle=True, drop_last=True, batch_size=self.all_config.pretrain.batch_size_freq, num_workers=16)
        if self.all_config.pretrain.mode != 'time':
            config.dp['privacy_history'].append([self.all_config.train.freq.dp.sigma, 1, 1])

        if self.all_config.pretrain.mode == 'freq_time':
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir + '_freq'
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_freq
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_freq
            self.pretrain(freq_train_loader, self.all_config.pretrain, run=True)
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir[:-5] + '_time'
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_time
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_time
            self.pretrain(time_dataloader, self.all_config.pretrain, run=True)
        elif self.all_config.pretrain.mode == 'time_freq':
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir + '_time'
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_time
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_time
            self.pretrain(time_dataloader, self.all_config.pretrain, run=True)
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir[:-5] + '_freq'
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_freq
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_freq
            self.pretrain(freq_train_loader, self.all_config.pretrain, run=True)
        elif self.all_config.pretrain.mode == 'no_auxiliary':
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir + '_time'
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_time
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_time
            self.pretrain(time_dataloader, self.all_config.pretrain, run=True)
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir[:-5] + '_freq'
            self.all_config.pretrain.n_epochs = self.all_config.train.freq.epochs
            self.all_config.pretrain.batch_size = self.all_config.train.freq.batch_size
            self.pretrain_freq(sensitive_train_loader, self.all_config.pretrain)
        elif self.all_config.pretrain.mode == 'dm_auxiliary':
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir + '_time'
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_time
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_time
            self.pretrain(time_dataloader, self.all_config.pretrain, run=True)
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir[:-13] + 'gen_freq'
            self.all_config.pretrain.n_epochs = self.all_config.train.freq.epochs
            self.all_config.pretrain.batch_size = self.all_config.train.freq.batch_size
            
            from models.model_loader import load_model
            model_sur, config_sur = load_model(self.all_config)
            config_sur.gen.log_dir = self.all_config.pretrain.log_dir + "/gen"
            model_sur.pretrain_freq(sensitive_train_loader, self.all_config.pretrain)
            syn_data, syn_labels = model_sur.generate(config_sur.gen, config_sur.model.sampler)
            del model_sur
            dist.barrier()

            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir[:-8] + 'pretrain_freq'
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_freq
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_freq

            syn = np.load(os.path.join(config_sur.gen.log_dir, 'gen.npz'))
            syn_data, syn_labels = syn["x"], syn["y"]
            freq_train_set = TensorDataset(torch.from_numpy(syn_data).float(), torch.from_numpy(syn_labels).long())
            freq_train_loader = torch.utils.data.DataLoader(dataset=freq_train_set, shuffle=True, drop_last=True, batch_size=self.all_config.pretrain.batch_size, num_workers=16)
            self.pretrain(freq_train_loader, self.all_config.pretrain, run=True)
        elif self.all_config.pretrain.mode == 'freq':
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir + '_freq'
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_freq
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_freq
            self.pretrain(freq_train_loader, self.all_config.pretrain, run=True)
        elif self.all_config.pretrain.mode == 'time':
            self.all_config.pretrain.log_dir = self.all_config.pretrain.log_dir + '_time'
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_time
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_time
            self.pretrain(time_dataloader, self.all_config.pretrain, run=True)
        elif self.all_config.pretrain.mode == 'mix':
            self.all_config.pretrain.n_epochs = self.all_config.pretrain.n_epochs_freq
            self.all_config.pretrain.batch_size = self.all_config.pretrain.batch_size_freq
            pretrain_set = ConcatDataset([freq_train_set, self.time_dataloader.dataset])
            self.pretrain(DataLoader(dataset=pretrain_set, shuffle=True, drop_last=True, batch_size=self.all_config.pretrain.batch_size, num_workers=16), self.all_config.pretrain, run=True)
        else:
            raise NotImplementedError
        
        return config

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

        if 'mode' in self.all_config.pretrain and self.all_config.pretrain.mode is not None:
            config = self.warm_up(sensitive_dataloader, config)
        
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
        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate) if self.ema is None else self.ema

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(trainable_parameters, **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the state dictionary to keep track of the training process.
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        if self.global_rank == 0:
            # Log the number of trainable parameters and other training details.
            model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            n_params = sum([np.prod(p.size()) for p in model_parameters])
            logging.info('Number of trainable parameters in model: %d' % n_params)
            logging.info('Number of total epochs: %d' % config.n_epochs)
            logging.info('Starting training at step %d' % state['step'])

        # Initialize the Privacy Engine for differential privacy.
        privacy_engine = PrivacyEngine(accountant='rdp' if 'accountant' not in config else config.accountant)
        if 'privacy_history' in config.dp and config.dp.privacy_history is not None:
            account_history = [tuple(item) for item in config.dp.privacy_history]
        else:
            account_history = None

        # Make the model, optimizer, and data loader private.
        if 'noise_multiplier' in config:
            model, optimizer, dataset_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=sensitive_dataloader,
            noise_multiplier=config.noise_multiplier,
            max_grad_norm=config.dp.max_grad_norm,
            noise_multiplicity=config.loss.n_noise_samples,
        )
        else:
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
            if 'cut_noise' in config:
                loss_fn = EDMLoss(**config.loss).get_loss_stage2
            else:
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
                    if 'cut_noise' in config:
                        loss = torch.mean(loss_fn(model, x, y, max_sigma=config.cut_noise))
                    else:
                        loss = torch.mean(loss_fn(model, x, y))
                    loss.backward()
                    optimizer.step()

                    if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                        # Log the loss at regular intervals.
                        logging.info('Loss: %.4f, step: %d' %
                                    (loss.item(), state['step'] + 1))
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
    
    def eval_mia(self, sensitive_train_dataloader, sensitive_test_dataloader, config):
        """
        Trains the model using the provided sensitive data loader and configuration.

        Args:
            sensitive_dataloader (DataLoader): DataLoader containing the sensitive data.
            config (Config): Configuration object containing various settings for training.

        Returns:
            None
        """
        
        set_seeds(self.global_rank, config.seed)
        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank
        # Set the number of classes for the loss function.
        config.loss.n_classes = self.private_num_classes

        model = self.model

        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss_mia
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function not supported")

        # Start the training loop.
        sigma_list = [0.07]
        model.eval()
        from sklearn.metrics import roc_curve
        for sigma in sigma_list:
            logging.info("sigma: {}".format(sigma))
            losses = []
            label = []
            category = []
            for train_x, train_y in sensitive_test_dataloader:
                x = train_x.to(self.device) * 2. - 1.
                x += torch.randn_like(x) * 0.008
                y = train_y.to(self.device).long()
                loss = loss_fn(model, x, y, sigma)
                losses.append(list(loss.detach().cpu().numpy()))
                label.append([1] * loss.shape[0])
                category.append(list(y.detach().cpu().numpy()))

            for train_x, train_y in sensitive_train_dataloader:
                x = train_x.to(self.device) * 2. - 1.
                y = train_y.to(self.device).long()
                loss = loss_fn(model, x, y, sigma)
                losses.append(list(loss.detach().cpu().numpy()))
                label.append([0] * loss.shape[0])
                category.append(list(y.detach().cpu().numpy()))
                if sum([len(x) for x in label]) >= len(sensitive_test_dataloader.dataset):
                    break
            
            all_losses = np.concatenate(losses)
            all_labels = np.concatenate(label)  # 0 for train, 1 for test
            all_categories = np.concatenate(category)

            # 获取所有类别
            unique_categories = np.unique(all_categories)

            # 定义FPR阈值
            fpr_threshold_list = [0.1, 0.01]
            for cat in unique_categories:
                # 找到当前类别的索引
                cat_mask = (all_categories == cat)
                cat_losses = all_losses[cat_mask]
                cat_labels = all_labels[cat_mask]
                
                # 计算ROC曲线
                fpr, tpr, thresholds = roc_curve(cat_labels, cat_losses)
                
                # 找到FPR最接近阈值的点
                for fpr_threshold in fpr_threshold_list:
                    idx = np.where(fpr >= fpr_threshold)[0]
                    if len(idx) > 0:
                        idx = idx[0]  # 第一个满足条件的点
                        tpr_at_fpr = tpr[idx]
                    else:
                        # 如果没有FPR达到阈值，取最后一个点
                        tpr_at_fpr = tpr[-1] if len(tpr) > 0 else 0.0
                    
                    logging.info(f"Category {cat}: TPR@FPR={fpr_threshold} = {tpr_at_fpr:.4f}")
            fpr, tpr, thresholds = roc_curve(all_labels, all_losses)
            
            # 找到FPR最接近阈值的点
            for fpr_threshold in fpr_threshold_list:
                idx = np.where(fpr >= fpr_threshold)[0]
                if len(idx) > 0:
                    idx = idx[0]  # 第一个满足条件的点
                    tpr_at_fpr = tpr[idx]
                else:
                    # 如果没有FPR达到阈值，取最后一个点
                    tpr_at_fpr = tpr[-1] if len(tpr) > 0 else 0.0
                
                logging.info(f"All: TPR@FPR={fpr_threshold} = {tpr_at_fpr:.4f}")


    def generate(self, config, sampler_config=None):
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
        sampler_config = self.sampler_acc if sampler_config is None else sampler_config
        def sampler_acc(x, y=None):
            if sampler_config.type == 'ddim':
                return ddim_sampler(x, y, model, **sampler_config)
            elif sampler_config.type == 'edm':
                return edm_sampler(x, y, model, **sampler_config)
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
            syn_data = np.concatenate(syn_data)
            syn_labels = np.concatenate(syn_labels)
            
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