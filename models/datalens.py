import os

import torch
import torchvision
import torch.nn.functional as F
import numpy as np


from models.synthesizer import DPSynther
from models.DataLens.model import DCGAN

class DataLens(DPSynther):
    def __init__(self, config, device, sess):
        super().__init__()

        self.config = config
        self.device = device

        self.model = DCGAN(sess=sess,
                        image_size=config.image_size,
                        y_dim=config.y_dim,
                        z_dim=config.z_dim,
                        dataset_name=config.dataset,
                        # parameters to tune
                        batch_teachers=config.batch_teachers,
                        pca=config.pca,
                        random_proj=config.random_proj,
                        pca_dim=config.pca_dim,
                        teachers_batch=config.teachers_batch,
                        wgan=config.wgan,
                        config=config)

    def train(self, sensitive_dataloader, config):
        os.mkdir(config.log_dir)
        
        data_x, data_y = [], []
        for x, y in sensitive_dataloader:
            data_x.append(x)
            data_y.append(y)
        
        data_x = torch.cat(data_x).permute(0, 2, 3, 1)
        data_y = torch.cat(data_y).long()
        data_y = F.one_hot(data_y, num_classes=self.model.y_dim)

        config.checkpoint_dir = os.path.join(config.log_dir, config.checkpoint_dir)
        config.teacher_dir = os.path.join(config.log_dir, config.teacher_dir)

        if not os.path.exists(config.checkpoint_dir):
            os.makedirs(config.checkpoint_dir)
        if not os.path.exists(config.teacher_dir):
            os.makedirs(config.teacher_dir)

        if self.config.wgan:
            config.learning_rate = 5e-5
            config.step_size = 5e-4

        epsilon, delta = self.model.train_together((data_x, data_y), config)

    def generate(self, config):
        os.mkdir(config.log_dir)
        n_batch = config.data_num // self.model.batch_size + 1
        dataX, datay = self.model.gen_data(n_batch)
        dataX = dataX[:config.data_num]
        datay = datay[:config.data_num]

        syn_data = dataX.reshape(dataX.shape[0], config.num_channels, config.resolution, config.resolution)
        syn_labels = datay.reshape(-1)
        np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

        show_images = []
        for cls in range(self.model.y_dim):
            show_images.append(syn_data[syn_labels==cls][:8])
        show_images = np.concatenate(show_images)
        torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
        return syn_data, syn_labels