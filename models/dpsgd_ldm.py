import os
import logging
import torch
import copy
import numpy as np

from models.synthesizer import DPSynther

class DP_Diffusion(DPSynther):
    def __init__(self, config, device):
        super().__init__()
        self.local_rank = config.local_rank
        self.global_rank = config.global_rank
        self.global_size = config.global_size

        self.config = config
        self.device = 'cuda:%d' % self.local_rank

        self.private_num_classes = config.private_num_classes
        self.public_num_classes = config.public_num_classes
        label_dim = max(self.private_num_classes, self.public_num_classes)
        self.network.label_dim = label_dim

        if config.ckpt is not None:
            pass
        self.is_pretrain = True
    
    def pretrain(self, public_dataloader, config):
        if public_dataloader is None:
            self.is_pretrain = False
            return
        
        torch.cuda.empty_cache()

    def train(self, sensitive_dataloader, config):
        if sensitive_dataloader is None:
            return


    def generate(self, config):
        logging.info("start to generate {} samples".format(config.data_num))
        if self.global_rank == 0 and not os.path.exists(config.log_dir):
            make_dir(config.log_dir)
        

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