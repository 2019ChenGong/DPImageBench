"""
This script computes the mean and covariance of the InceptionV3 activations on
generated samples. A separate script should be used for computing the same
statistics on the real dataset. To compute the FID, these stats can be combined
using a third script.
"""
import argparse

from pytorch_fid.inception import InceptionV3
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fid.cifar10_fid_stats_pytorch_fid import stats_from_dataloader, set_seeds


class DatasetWrapper(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        image = self.images[i]
        assert image.shape[0] == 3 and image.shape[1] == image.shape[2], \
               f"Samples not in CxHxW format, instead got {image.shape}"
        image = image.clamp(min=-1, max=1)
        return image


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    images = torch.load(args.samples)["image"]
    dataset = DatasetWrapper(images)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size)

    inception_model = InceptionV3(normalize_input=False).to(device)
    mu, sigma = stats_from_dataloader(dataloader, inception_model, device)

    if args.output:
        np.savez(args.output, mu=mu, sigma=sigma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--batch_size', type=int, default=500, help='Number of samples per batch')
    parser.add_argument('--samples', type=str, help='Path to samples class')
    parser.add_argument('--output', type=str, help='Path to output fid stats (.npz)')
    args = parser.parse_args()

    if not args.output:
        print("[WARN]: --output not provided, generated stats will not be saved")

    set_seeds(0, 0)

    main(args)
