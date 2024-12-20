import os, sys, argparse
from packaging import version
import logging
import numpy as np
from omegaconf import OmegaConf

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from models.ldm.data.util import VirtualBatchWrapper, DataModuleFromDataset
from models.ldm.util import instantiate_from_config
from models.ldm.gen import generate_batch
from models.ldm.privacy.myopacus import MyDPLightningDataModule
from models.DP_Diffusion.utils.util import make_dir
from models.synthesizer import DPSynther

from models.ldm.callbacks.cuda import CUDACallback                         # noqa: F401
from models.ldm.callbacks.image_logger import ImageLogger                  # noqa: F401
from models.ldm.callbacks.setup import SetupCallback                       # noqa: F401
from models.ldm.data.util import DataModuleFromConfig, WrappedDataset, WrappedDataset_ldm  # noqa: F401

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
        
        self.pretrain_autoencoder(config.unet, config.log_dir)
        self.pretrain_unet(config.unet, config.log_dir)

        torch.cuda.empty_cache()
    
    def pretrain_autoencoder(self, config, logdir):

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None:
            gpu_ids = ','.join([i for i in range(self.config.setup.n_gpus_per_node)])
        else:
            gpu_ids = cuda_visible_devices
        config_path = config.config_path
        scripts = ['models/DP_LDM/main.py', '-t', '--base', config_path, '--gpus', gpu_ids]
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(warm_up, script) for script in scripts]
            for future in futures:
                try:
                    output = future.result()
                    print(f"Output:\n{output}")
                except Exception as e:
                    print(f"generated an exception: {e}")

    def pretrain_unet(self, dataset, config, logdir):
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None:
            gpu_ids = ','.join([i for i in range(self.config.setup.n_gpus_per_node)])
        else:
            gpu_ids = cuda_visible_devices
        config_path = config.config_path
        scripts = ['models/DP_LDM/main.py', '-t', '--base', config_path, '--gpus', gpu_ids]
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(warm_up, script) for script in scripts]
            for future in futures:
                try:
                    output = future.result()
                    print(f"Output:\n{output}")
                except Exception as e:
                    print(f"generated an exception: {e}")

    def train(self, sensitive_dataloader, config):
        if sensitive_dataloader is None:
            return
        
        if self.global_rank == 0:
            make_dir(config.log_dir)
        
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is None:
            gpu_ids = '0,'
        else:
            gpu_ids = cuda_visible_devices[:2]
        config_path = config.config_path
        scripts = ['models/DP_LDM/main.py', '-t', '--base', config_path, '--gpus', gpu_ids, '--accelerator', 'gpu']
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(warm_up, script) for script in scripts]
            for future in futures:
                try:
                    output = future.result()
                    print(f"Output:\n{output}")
                except Exception as e:
                    print(f"generated an exception: {e}")


    def generate(self, config):
        logging.info("start to generate {} samples".format(config.data_num))
        if self.global_rank == 0 and not os.path.exists(config.log_dir):
            make_dir(config.log_dir)
        
        model_path = os.path.join(self.config.train.log_dir, "checkpoints", "last.ckpt")
        syn_data, syn_labels = generate_batch(self.config.train, config.data_num, model_path, num_classes=self.config.sensitive_data.n_classes, batch_size=config.batch_size)

        logging.info("Generation Finished!")

        np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

        show_images = []
        for cls in range(self.config.sensitive_data.n_classes):
            show_images.append(syn_data[syn_labels==cls][:8])
        show_images = np.concatenate(show_images)
        torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
        return syn_data, syn_labels


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))