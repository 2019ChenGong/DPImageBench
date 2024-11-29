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
        super().__init__()
        self.local_rank = config.local_rank
        self.global_rank = config.global_rank
        self.global_size = config.global_size

        self.denoiser_name = config.denoiser_name
        self.denoiser_network = config.denoiser_network
        self.ema_rate = config.ema_rate
        self.network = config.network
        self.sampler = config.sampler
        self.sampler_fid = config.sampler_fid
        self.sampler_acc = config.sampler_acc
        self.fid_stats = config.fid_stats

        self.config = config
        self.device = 'cuda:%d' % self.local_rank

        self.private_num_classes = config.private_num_classes
        self.public_num_classes = config.public_num_classes
        label_dim = max(self.private_num_classes, self.public_num_classes)
        self.network.label_dim = label_dim

        if self.denoiser_name == 'edm':
            if self.denoiser_network == 'song':
                self.model = EDMDenoiser(
                    NCSNpp(**self.network).to(self.device))
            else:
                raise NotImplementedError
        elif self.denoiser_name == 'vpsde':
            if self.denoiser_network == 'song':
                self.model = VPSDEDenoiser(self.config.beta_min, self.config.beta_max - self.config.beta_min,
                                    self.config.scale, NCSNpp(**self.network).to(self.device))
            else:
                raise NotImplementedError
        elif self.denoiser_name == 'vesde':
            if self.denoiser_network == 'song':
                self.model = VESDEDenoiser(
                    NCSNpp(**self.network).to(self.device))
            else:
                raise NotImplementedError
        elif self.denoiser_name == 'v':
            if self.denoiser_network == 'song':
                self.model = VDenoiser(
                    NCSNpp(**self.network).to(self.device))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        self.model = self.model.to(self.local_rank)
        self.model.train()
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_rate)

        if config.ckpt is not None:
            state = torch.load(config.ckpt, map_location=self.device)
            new_state_dict = {}
            for k, v in state['model'].items():
                new_state_dict[k[7:]] = v
            logging.info(self.model.load_state_dict(new_state_dict, strict=True))
            self.ema.load_state_dict(state['ema'])
            del state, new_state_dict
        self.is_pretrain = True
    
    def pretrain(self, public_dataloader, config):
        if public_dataloader is None:
            self.is_pretrain = False
            return
        
        config.loss.n_classes = self.private_num_classes
        if config.cond:
            config.loss['label_unconditioning_prob'] = 0.1
        else:
            config.loss['label_unconditioning_prob'] = 1.0

        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        model = DDP(self.model, device_ids=[self.local_rank])
        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError

        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        if self.global_rank == 0:
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            n_params = sum([np.prod(p.size()) for p in model_parameters])
            logging.info('Number of trainable parameters in model: %d' % n_params)
            logging.info('Number of total epochs: %d' % config.n_epochs)
            logging.info('Starting training at step %d' % state['step'])
        dist.barrier()

        dataset_loader = torch.utils.data.DataLoader(
        dataset=public_dataloader.dataset, batch_size=config.batch_size//self.global_size, sampler=DistributedSampler(public_dataloader.dataset), pin_memory=True, drop_last=True)

        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError

        inception_model = InceptionFeatureExtractor()
        inception_model.model = inception_model.model.to(self.device)

        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError

        snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
                                self.network.num_in_channels, self.network.image_size, self.network.image_size)
        fid_sampling_shape = (self.sampler.fid_batch_size, self.network.num_in_channels,
                            self.network.image_size, self.network.image_size)

        for epoch in range(config.n_epochs):
            dataset_loader.sampler.set_epoch(epoch)
            for _, (train_x, train_y) in enumerate(dataset_loader):

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

                if state['step'] % config.save_freq == 0 and state['step'] >= config.save_threshold and self.global_rank == 0:
                    checkpoint_file = os.path.join(
                        checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                    save_checkpoint(checkpoint_file, state)
                    logging.info('Saving checkpoint at iteration %d' % state['step'])
                dist.barrier()

                # Preprocess and train
                if len(train_y.shape) == 2:
                    train_x = train_x.to(torch.float32) / 255.
                    train_y = torch.argmax(train_y, dim=1)
                train_x, train_y = train_x.to(self.device) * 2. - 1., train_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean(loss_fn(model, train_x, train_y))
                loss.backward()
                optimizer.step()

                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('Loss: %.4f, step: %d' % (loss, state['step'] + 1))
                dist.barrier()

                state['step'] += 1
                state['ema'].update(model.parameters())
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))

        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')
        dist.barrier()

        ema.copy_to(self.model.parameters())
        self.ema = ema

        del model
        torch.cuda.empty_cache()

    def train(self, sensitive_dataloader, config):
        if sensitive_dataloader is None:
            return
        
        set_seeds(self.global_rank, config.seed)
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank
        config.loss.n_classes = self.private_num_classes

        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        # if config.partly_finetune:
        #     for name, param in self.model.named_parameters():
        #         layer_idx = int(name.split('.')[2])
        #         if layer_idx > 3 and 'NIN' not in name:
        #             param.requires_grad = False
        #             if self.global_rank == 0:
        #                 logging.info('{} is frozen'.format(name))
        
        if config.partly_finetune:
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
            trainable_parameters = self.model.parameters()

        model = DPDDP(self.model)
        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(trainable_parameters, **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError

        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        if self.global_rank == 0:
            model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            n_params = sum([np.prod(p.size()) for p in model_parameters])
            logging.info('Number of trainable parameters in model: %d' % n_params)
            logging.info('Number of total epochs: %d' % config.n_epochs)
            logging.info('Starting training at step %d' % state['step'])

        privacy_engine = PrivacyEngine()
        if config.dp.sdq is None:
            account_history = None
            alpha_history = None
        else:
            account_history = [tuple(item) for item in config.dp.privacy_history]
            if config.dp.alpha_num == 0:
                alpha_history = None
            else:
                alpha = np.arange(config.dp.alpha_num) / config.dp.alpha_num
                alpha = alpha * (config.dp.alpha_max-config.dp.alpha_min)
                alpha += config.dp.alpha_min 
                alpha_history = list(alpha)

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
            alpha_history=alpha_history,
        )

        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError
        
        inception_model = InceptionFeatureExtractor()
        inception_model.model = inception_model.model.to(self.device)

        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError

        snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
                                self.network.num_in_channels, self.network.image_size, self.network.image_size)
        fid_sampling_shape = (self.sampler.fid_batch_size, self.network.num_in_channels,
                            self.network.image_size, self.network.image_size)

        for epoch in range(config.n_epochs):
            with BatchMemoryManager(
                    data_loader=dataset_loader,
                    max_physical_batch_size=config.dp.max_physical_batch_size,
                    optimizer=optimizer,
                    n_splits=config.n_splits if config.n_splits > 0 else None) as memory_safe_data_loader:

                for _, (train_x, train_y) in enumerate(memory_safe_data_loader):
                    if state['step'] % config.snapshot_freq == 0 and state['step'] >= config.snapshot_threshold and self.global_rank == 0:
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
                        checkpoint_file = os.path.join(
                            checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                        save_checkpoint(checkpoint_file, state)
                        logging.info(
                            'Saving  checkpoint at iteration %d' % state['step'])
                    dist.barrier()

                    if len(train_y.shape) == 2:
                        train_x = train_x.to(torch.float32) / 255.
                        train_y = torch.argmax(train_y, dim=1)
                    
                    x = train_x.to(self.device) * 2. - 1.
                    y = train_y.to(self.device).long()

                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.mean(loss_fn(model, x, y))
                    loss.backward()
                    optimizer.step()

                    if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                        logging.info('Loss: %.4f, step: %d' %
                                    (loss, state['step'] + 1))
                    dist.barrier()

                    state['step'] += 1
                    if not optimizer._is_last_step_skipped:
                        state['ema'].update(model.parameters())

                logging.info('Eps-value after %d epochs: %.4f' %
                            (epoch + 1, privacy_engine.get_epsilon(config.dp.delta)))

        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')
        dist.barrier()

        self.ema = ema


    def generate(self, config):
        logging.info("start to generate {} samples".format(config.data_num))
        if self.global_rank == 0 and not os.path.exists(config.log_dir):
            make_dir(config.log_dir)
        dist.barrier()

        sampling_shape = (config.batch_size, self.network.num_in_channels, self.network.image_size, self.network.image_size)

        model = DDP(self.model)
        model.eval()
        self.ema.copy_to(model.parameters())

        def sampler_acc(x, y=None):
            if self.sampler_acc.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler_acc)
            elif self.sampler_acc.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler_acc)
            else:
                raise NotImplementedError

        if self.global_rank == 0:
            syn_data = []
            syn_labels = []
        for _ in range(config.data_num // (sampling_shape[0] * self.global_size) + 1):
            x, y = generate_batch(sampler_acc, sampling_shape, self.device, self.private_num_classes, self.private_num_classes)
            dist.barrier()
            if self.global_rank == 0:
                gather_x = [torch.zeros_like(x) for _ in range(self.global_size)]
                gather_y = [torch.zeros_like(y) for _ in range(self.global_size)]
            else:
                gather_x = None
                gather_y = None
            dist.gather(x, gather_x)
            dist.gather(y, gather_y)
            if self.global_rank == 0:
                syn_data.append(torch.cat(gather_x).detach().cpu().numpy())
                syn_labels.append(torch.cat(gather_y).detach().cpu().numpy())
        

        if self.global_rank == 0:
            logging.info("Generation Finished!")
            syn_data = np.concatenate(syn_data)
            syn_labels = np.concatenate(syn_labels)

            np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

            show_images = []
            for cls in range(self.private_num_classes):
                show_images.append(syn_data[syn_labels==cls][:8])
            show_images = np.concatenate(show_images)
            torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
            return syn_data, syn_labels
        else:
            return None, None