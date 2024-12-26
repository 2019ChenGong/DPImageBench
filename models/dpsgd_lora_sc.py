import os, sys, argparse
from packaging import version
import logging
import numpy as np
from omegaconf import OmegaConf

import torch
import torchvision

import subprocess
from concurrent.futures import ProcessPoolExecutor

from models.DP_Diffusion.utils.util import make_dir
from models.synthesizer import DPSynther

def execute(script):
    try:
        result = subprocess.run(['python'] + script, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"error: {e.stderr}")
        return e.stderr

class DP_LDM(DPSynther):
    def __init__(self, config, device):
        super().__init__()
        self.local_rank = config.setup.local_rank
        self.global_rank = config.setup.global_rank
        self.global_size = config.setup.global_size

        self.config = config
        self.device = 'cuda:%d' % self.local_rank

        self.is_pretrain = True
    
    def pretrain(self, public_dataloader, config):
        if public_dataloader is None:
            self.is_pretrain = False
            return
        if self.global_rank == 0:
            make_dir(config.log_dir)
        
        self.pretrain_autoencoder(public_dataloader.dataset, config.autoencoder, os.path.join(config.log_dir, 'autoencoder'))
        self.pretrain_unet(public_dataloader.dataset, config.unet, os.path.join(config.log_dir, 'unet'))

        torch.cuda.empty_cache()
    
    def pretrain_autoencoder(self, public_dataset, config, logdir):
        if self.config.pretrain.unet.pretrain_model is not None:
            return
        if self.global_rank == 0:
            make_dir(logdir)

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None:
            gpu_ids = ','.join([i for i in range(self.config.setup.n_gpus_per_node)])
        else:
            gpu_ids = cuda_visible_devices
        config_path = config.config_path
        scripts = [[
            'models/DP_LDM/main.py', 
            '-t', 
            '--logdir', logdir, 
            '--base', config_path, 
            '--gpus', gpu_ids, 
            'data.params.batch_size={}'.format(config.batch_size), 
            'lightning.trainer.max_epochs={}'.format(config.n_epochs), 
            'data.params.train.params.root={}'.format(self.config.public_data.train_path),
            'data.params.validation.params.root={}'.format(self.config.public_data.train_path),
            'data.params.train.params.image_size={}'.format(self.config.public_data.resolution),
            'data.params.validation.params.image_size={}'.format(self.config.public_data.resolution),
            'data.params.train.params.c={}'.format(self.config.public_data.num_channels),
            'data.params.validation.params.c={}'.format(self.config.sensitive_data.num_channels),
            'data.params.train.data_num={}'.format(len(public_dataset)),
            'data.params.validation.data_num={}'.format(len(public_dataset)),
            ]]
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(execute, script) for script in scripts]
            for future in futures:
                try:
                    output = future.result()
                    logging.info(f"Output:\n{output}")
                except Exception as e:
                    logging.info(f"generated an exception: {e}")
        
        self.config.pretrain.unet.pretrain_model = os.path.join(logdir, 'checkpoints', 'last.ckpt')

    def pretrain_unet(self, public_dataset, config, logdir):
        if self.global_rank == 0:
            make_dir(logdir)

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None:
            gpu_ids = ','.join([str(i) for i in range(self.config.setup.n_gpus_per_node)]) + ','
        else:
            gpu_ids = str(cuda_visible_devices) + ','
        config_path = config.config_path
        pretrain_model = self.config.pretrain.unet.pretrain_model
        scripts = [[
            'models/DP_LDM/main.py', 
            '-t', 
            '--logdir', logdir, 
            '--base', config_path, 
            '--gpus', gpu_ids, 
            'data.params.batch_size={}'.format(config.batch_size), 
            'lightning.trainer.max_epochs={}'.format(config.n_epochs), 
            'model.params.first_stage_config.params.ckpt_path={}'.format(pretrain_model), 
            'data.params.train.params.root={}'.format(self.config.public_data.train_path),
            'data.params.validation.params.root={}'.format(self.config.public_data.train_path),
            'data.params.train.params.image_size={}'.format(self.config.public_data.resolution),
            'data.params.validation.params.image_size={}'.format(self.config.public_data.resolution),
            'data.params.train.params.c={}'.format(self.config.public_data.num_channels),
            'data.params.validation.params.c={}'.format(self.config.sensitive_data.num_channels),
            'data.params.train.data_num={}'.format(len(public_dataset)),
            'data.params.validation.data_num={}'.format(len(public_dataset)),
            ]]
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(execute, script) for script in scripts]
            for future in futures:
                try:
                    output = future.result()
                    logging.info(f"Output:\n{output}")
                except Exception as e:
                    logging.info(f"generated an exception: {e}")
        
        self.config.train.pretrain_model = os.path.join(logdir, 'checkpoints', 'last.ckpt')

    def train(self, sensitive_dataloader, config):
        if sensitive_dataloader is None:
            return
        
        if self.global_rank == 0:
            make_dir(config.log_dir)
        
        gpu_ids = '0,'
        config_path = config.config_path
        pretrain_model = self.config.pretrain_model
        scripts = [[
            'models/DP_LORA/main.py', 
            '-t', 
            '--logdir', config.log_dir, 
            '--base', config_path, 
            '--gpus', gpu_ids, 
            '--accelerator', 'gpu', 
            'model.params.ckpt_path={}'.format(pretrain_model), 
            'model.params.dp_config.epsilon={}'.format(config.dp.epsilon), 
            'model.params.dp_config.delta={}'.format(config.dp.delta), 
            'model.params.dp_config.max_grad_norm={}'.format(config.dp.max_grad_norm), 
            'data.params.batch_size={}'.format(config.batch_size), 
            'lightning.trainer.max_epochs={}'.format(config.n_epochs), 
            'data.params.train.params.path={}'.format(self.config.sensitive_data.train_path),
            'data.params.validation.params.path={}'.format(self.config.sensitive_data.train_path),
            'data.params.train.params.resolution={}'.format(self.config.sensitive_data.resolution),
            'data.params.validation.params.resolution={}'.format(self.config.sensitive_data.resolution),
            'data.params.train.params.c={}'.format(self.config.sensitive_data.num_channels),
            'data.params.validation.params.c={}'.format(self.config.sensitive_data.num_channels),
            'data.params.train.data_num={}'.format(len(sensitive_dataloader.dataset)),
            'data.params.validation.data_num={}'.format(len(sensitive_dataloader.dataset)),
            ]]
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(execute, script) for script in scripts]
            for future in futures:
                try:
                    output = future.result()
                    logging.info(f"Output:\n{output}")
                except Exception as e:
                    logging.info(f"generated an exception: {e}")


    def generate(self, config):
        logging.info("start to generate {} samples".format(config.data_num))
        if self.global_rank == 0 and not os.path.exists(config.log_dir):
            make_dir(config.log_dir)
        
        scripts = [[
            'models/DP_LDM/cond_sampling_test.py', 
            '--save_path', config.log_dir, 
            '--yaml', self.config.train.config_path, 
            '--ckpt_path', os.path.join(self.config.train.log_dir, 'checkpoints', 'last.ckpt'), 
            '--num_samples', str(config.data_num), 
            '--num_classes', str(self.config.sensitive_data.n_classes), 
            '--batch_size', str(config.batch_size)
            ]]
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(execute, script) for script in scripts]
            for future in futures:
                try:
                    output = future.result()
                    logging.info(f"Output:\n{output}")
                except Exception as e:
                    logging.info(f"generated an exception: {e}")

        logging.info("Generation Finished!")

        syn = np.load(os.path.join(config.log_dir, 'gen.npz'))
        syn_data, syn_labels = syn["x"], syn["y"]

        show_images = []
        for cls in range(self.config.sensitive_data.n_classes):
            show_images.append(syn_data[syn_labels==cls][:8])
        show_images = np.concatenate(show_images)
        torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
        return syn_data, syn_labels