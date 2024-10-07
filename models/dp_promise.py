import os
import logging
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import pickle
import torchvision

from models.DP_Diffusion.model.ncsnpp import NCSNpp
from models.DP_Diffusion.utils.util import set_seeds, make_dir, save_checkpoint, sample_random_image_batch, compute_fid
from models.DP_Diffusion.dnnlib.util import open_url
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
from models.DP_Promise.mechanism import get_noise_multiplier

class DP_Promise(DPSynther):
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
        self.device = device

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
        
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_rate)

        if config.ckpt is not None:
            state = torch.load(config.ckpt, map_location=self.device)
            new_state_dict = {}
            for k, v in state['model'].items():
                new_state_dict[k[7:]] = v
            logging.info(self.model.load_state_dict(new_state_dict, strict=True))
            logging.info(self.ema.load_state_dict(state['ema']))
            del state, new_state_dict
    
    def pretrain(self, public_dataloader, config):
        if public_dataloader is None:
            return

        set_seeds(self.global_rank, config.seed)
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')
        fid_dir = os.path.join(config.log_dir, 'fid')

        if self.global_rank == 0:
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)
            make_dir(fid_dir)
        dist.barrier()

        model = DDP(self.model)
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
        dataset=public_dataloader.dataset, batch_size=config.batch_size//self.global_size, sampler=DistributedSampler(public_dataloader.dataset), num_workers=2,
        pin_memory=True)

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

        with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
            inception_model = pickle.load(f).to(self.device)

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

            for _, (train_x, train_y) in enumerate(dataset_loader):
                if state['step'] % config.snapshot_freq == 0 and state['step'] >= config.snapshot_threshold and self.global_rank == 0:
                    logging.info(
                        'Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])

                    model.eval()
                    with torch.no_grad():
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        sample_random_image_batch(snapshot_sampling_shape, sampler, os.path.join(
                            sample_dir, 'iter_%d' % state['step']), self.device, self.network.label_dim)
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
                        fid = compute_fid(config.fid_samples, self.global_size, fid_sampling_shape, sampler, inception_model, self.fid_stats, self.device, self.network.label_dim)
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
                if config.label_random:
                    y = train_y.to(self.device).long() % self.network.label_dim
                    # y = torch.randint(self.network.label_dim, size=(x.shape[0],), dtype=torch.int32, device=self.device)
                else:
                    y = train_y.to(self.device).long()

                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean(loss_fn(model, x, y))
                loss.backward()
                optimizer.step()

                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('Loss: %.4f, step: %d' % (loss, state['step'] + 1))
                dist.barrier()

                state['step'] += 1
                state['ema'].update(model.parameters())

            logging.info('After %d epochs' % (epoch + 1))

        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')
        dist.barrier()

        ema.copy_to(self.model.parameters())
        self.ema = ema
        

    def train(self, sensitive_dataloader, config):
        if sensitive_dataloader is None:
            return
        set_seeds(self.global_rank, config.seed)
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        if self.global_rank == 0:
            make_dir(config.log_dir)
        dist.barrier()

        data_num = len(sensitive_dataloader.dataset)
        q1 = 1 / (data_num // config.batch_size1)
        q2 = 1 / (data_num // config.batch_size)
        niter1 = config.n_epochs1 * (data_num // config.batch_size1)
        niter2 = config.n_epochs * (data_num // config.batch_size)

        noise_multiplier = get_noise_multiplier(config, q1, q2, niter1, niter2, image_shape=(self.network.num_in_channels, self.network.image_size, self.network.image_size))
        
        # self.stage1_train(sensitive_dataloader, config)
        self.stage2_train(sensitive_dataloader, config, noise_multiplier)


    def generate(self, config):
        logging.info("start to generate {} samples".format(config.data_num))
        workdir = os.path.join(config.log_dir, 'samples{}_acc'.format(config.data_num))
        sample_dir = os.path.join(workdir, 'samples/')
        if self.global_rank == 0:
            make_dir(config.log_dir)
            make_dir(workdir)
            make_dir(sample_dir)
        dist.barrier()

        sampling_shape = (config.batch_size, self.network.num_in_channels, self.network.image_size, self.network.image_size)

        model = DDP(self.model)
        self.ema.copy_to(model.parameters())
        model.eval()

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
            x, y = generate_batch(sampler_acc, sampling_shape, self.device, self.network.label_dim, self.network.label_dim)
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
            for cls in range(self.network.label_dim):
                show_images.append(syn_data[syn_labels==cls][:8])
            show_images = np.concatenate(show_images)
            torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
            return syn_data, syn_labels
        else:
            return None, None
    

    def stage1_train(self, sensitive_dataloader, config):
        log_dir = os.path.join(config.log_dir, 'stage1')
        sample_dir = os.path.join(log_dir, 'samples')
        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        fid_dir = os.path.join(log_dir, 'fid')

        if self.global_rank == 0:
            make_dir(log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)
            make_dir(fid_dir)
        dist.barrier()

        dataset_loader = torch.utils.data.DataLoader(
        dataset=sensitive_dataloader.dataset, batch_size=config.batch_size1//self.global_size, sampler=DistributedSampler(sensitive_dataloader.dataset), num_workers=2,
        pin_memory=True)

        model = DDP(self.model)
        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        if config.optim1.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim1.params)
        elif config.optim1.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim1.params)
        else:
            raise NotImplementedError

        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        if self.global_rank == 0:
            model_parameters = filter(
                lambda p: p.requires_grad, model.parameters())
            n_params = sum([np.prod(p.size()) for p in model_parameters])
            logging.info('Number of trainable parameters in model: %d' % n_params)
            logging.info('Number of total epochs: %d' % config.n_epochs1)
            logging.info('Starting training at step %d' % state['step'])
        dist.barrier()

        if config.loss1.version == 'edm':
            loss_fn = EDMLoss(**config.loss1).get_loss_stage1
        else:
            raise NotImplementedError

        with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
            inception_model = pickle.load(f).to(self.device)

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

        for epoch in range(config.n_epochs1):
            for _, (train_x, train_y) in enumerate(dataset_loader):
                if state['step'] % config.snapshot_freq == 0 and state['step'] >= config.snapshot_threshold and self.global_rank == 0:
                    logging.info(
                        'Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])

                    model.eval()
                    with torch.no_grad():
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        sample_random_image_batch(snapshot_sampling_shape, sampler, os.path.join(
                            sample_dir, 'iter_%d' % state['step']), self.device, self.network.label_dim)
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
                        fid = compute_fid(config.fid_samples, self.global_size, fid_sampling_shape, sampler, inception_model, self.fid_stats, self.device, self.network.label_dim)
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
                loss = torch.mean(loss_fn(model, x, y, config.dp.gaussian_max))
                loss.backward()
                optimizer.step()

                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('Loss: %.4f, step: %d' %
                                (loss, state['step'] + 1))
                dist.barrier()

                state['step'] += 1
                state['ema'].update(model.parameters())

            # logging.info('Eps-value after %d epochs: %.4f' %
            #             (epoch + 1, privacy_engine.get_epsilon(config.dp.delta)))

        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')
        dist.barrier()

        ema.copy_to(self.model.parameters())
        self.ema = ema

    def stage2_train(self, sensitive_dataloader, config, noise_multiplier):
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')
        fid_dir = os.path.join(config.log_dir, 'fid')

        if self.global_rank == 0:
            make_dir(sample_dir)
            make_dir(checkpoint_dir)
            make_dir(fid_dir)
        dist.barrier()

        model = DPDDP(self.model)
        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError

        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        if self.global_rank == 0:
            model_parameters = filter(
                lambda p: p.requires_grad, model.parameters())
            n_params = sum([np.prod(p.size()) for p in model_parameters])
            logging.info('Number of trainable parameters in model: %d' % n_params)
            logging.info('Number of total epochs: %d' % config.n_epochs)
            logging.info('Starting training at step %d' % state['step'])
        dist.barrier()

        privacy_engine = PrivacyEngine()
        
        model, optimizer, dataset_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=sensitive_dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=config.dp.max_grad_norm,
            noise_multiplicity=config.loss.n_noise_samples,
        )

        # model, optimizer, dataset_loader = privacy_engine.make_private_with_epsilon(
        #     module=model,
        #     optimizer=optimizer,
        #     data_loader=sensitive_dataloader,
        #     target_delta=config.dp.delta,
        #     target_epsilon=config.dp.epsilon,
        #     epochs=config.n_epochs,
        #     max_grad_norm=config.dp.max_grad_norm,
        #     noise_multiplicity=config.loss.n_noise_samples,
        # )

        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss_stage2
        else:
            raise NotImplementedError

        with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
            inception_model = pickle.load(f).to(self.device)

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
                    n_splits=config.dp.n_splits if config.dp.n_splits > 0 else None) as memory_safe_data_loader:

                for _, (train_x, train_y) in enumerate(memory_safe_data_loader):
                    if state['step'] % config.snapshot_freq == 0 and state['step'] >= config.snapshot_threshold and self.global_rank == 0:
                        logging.info(
                            'Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])

                        model.eval()
                        with torch.no_grad():
                            ema.store(model.parameters())
                            ema.copy_to(model.parameters())
                            sample_random_image_batch(snapshot_sampling_shape, sampler, os.path.join(
                                sample_dir, 'iter_%d' % state['step']), self.device, self.network.label_dim)
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
                            fid = compute_fid(config.fid_samples, self.global_size, fid_sampling_shape, sampler, inception_model, self.fid_stats, self.device, self.network.label_dim)
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
                    loss = torch.mean(loss_fn(model, x, y, config.dp.gaussian_max))
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

        # def sampler_final(x, y=None):
        #     if self.sampler_fid.type == 'ddim':
        #         return ddim_sampler(x, y, model, **self.sampler_fid)
        #     elif self.sampler_fid.type == 'edm':
        #         return edm_sampler(x, y, model, **self.sampler_fid)
        #     else:
        #         raise NotImplementedError

        # model.eval()
        # with torch.no_grad():
        #     ema.store(model.parameters())
        #     ema.copy_to(model.parameters())
        #     if self.global_rank == 0:
        #         sample_random_image_batch(snapshot_sampling_shape, sampler_final, os.path.join(
        #                             sample_dir, 'final'), self.device, self.network.label_dim)
        #     fid = compute_fid(config.final_fid_samples, self.global_size, fid_sampling_shape, sampler_final, inception_model,
        #                     self.fid_stats, self.device, self.network.label_dim)
        #     ema.restore(self.model.parameters())

        # if self.global_rank == 0:
        #     logging.info('Final FID %.6f' % (fid))
        # dist.barrier()

        self.ema = ema