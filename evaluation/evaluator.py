import torch
import pickle
import numpy as np
from scipy import linalg
import logging


from models.DP_Diffusion.dnnlib.util import open_url


class Evaluator(object):
    def __init__(self, config):

        self.device = config.setup.local_rank

        self.sensitive_stats_path = config.sensitive_data.fid_stats
        self.config = config
    
    def eval(self, synthetic_images, synthetic_labels, sensitive_test_loader):
        if self.device != 0:
            return
        
        fid = self.cal_fid(synthetic_images)
        logging.info("The FID of synthetic images is {}".format(fid))
        print("The FID of synthetic images is {}".format(fid))

    def cal_fid(self, synthetic_images, batch_size=500):
        with open_url('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl') as f:
            inception_model = pickle.load(f).to(self.device)

        act = []
        chunks = torch.chunk(torch.from_numpy(synthetic_images), len(synthetic_images) // batch_size)
        print('Starting to sample.')
        for batch in chunks:
            batch = (batch * 255.).to(torch.uint8)

            batch = batch.to(self.device)
            if batch.shape[1] == 1:  # if image is gray scale
                batch = batch.repeat(1, 3, 1, 1)
            elif len(batch.shape) == 3:  # if image is gray scale
                batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)

            with torch.no_grad():
                pred = inception_model(batch.to(self.device), return_features=True).unsqueeze(-1).unsqueeze(-1)

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            act.append(pred)

        act = np.concatenate(act, axis=0)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)

        stats = np.load(self.sensitive_stats_path)
        data_pools_mean = stats['mu']
        data_pools_sigma = stats['sigma']

        m = np.square(mu - data_pools_mean).sum()
        s, _ = linalg.sqrtm(np.dot(sigma, data_pools_sigma), disp=False)
        fd = np.real(m + np.trace(sigma + data_pools_sigma - s * 2))

        return fd
    
    