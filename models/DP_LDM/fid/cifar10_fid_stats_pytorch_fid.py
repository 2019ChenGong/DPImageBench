import argparse
import os
import random

import numpy as np
from pytorch_fid.inception import InceptionV3
from tqdm import tqdm

import torch as pt
import torchvision
from torchvision import transforms


def set_seeds(rank, seed):
    random.seed(rank + seed)
    pt.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    pt.cuda.manual_seed(rank + seed)
    pt.cuda.manual_seed_all(rank + seed)
    pt.backends.cudnn.benchmark = True


def stats_from_dataloader(dataloader, model, device='cpu', save_memory=False):
    """
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    model.eval()

    pred_list = []

    if not save_memory:  # compute in single pass, store all embeddings
        pbar = tqdm(dataloader)
        for batch in pbar:
            x = batch[0] if (isinstance(batch, tuple) or isinstance(batch, list)) else batch
            x = x.to(device)

            pbar.set_description(f"x.shape={tuple(x.shape)} | x.min()={x.min()} | x.max()={x.max()}")

            with pt.no_grad():
                pred = model(x)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = pt.nn.adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_list.append(pred)

        pred_arr = np.concatenate(pred_list, axis=0)
        mu = np.mean(pred_arr, axis=0)
        sigma = np.cov(pred_arr, rowvar=False)
        return mu, sigma
    else:  # compute in two passes, no need to store all embeddings
        # first pass: calculate mean
        mu_acc = None
        n_samples = 0
        for batch in tqdm(dataloader):
            x = batch[0] if (isinstance(batch, tuple) or isinstance(batch, list)) else batch
            x = x.to(device)

            with pt.no_grad():
                pred = model(x)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = pt.nn.adaptive_avg_pool2d(pred, output_size=(1, 1))

            n_samples += pred.shape[0]
            pred = pt.sum(pred.squeeze(3).squeeze(2), dim=0)
            mu_acc = mu_acc + pred if mu_acc is not None else pred

        mu = mu_acc / n_samples
        sigma_acc = None
        for batch in tqdm(dataloader):
            x = batch[0] if (isinstance(batch, tuple) or isinstance(batch, list)) else batch
            x = x.to(device)
            with pt.no_grad():
                pred = model(x)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = pt.nn.adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2)
            pred_cent = pred - mu
            sigma_batch = pt.matmul(pt.t(pred_cent), pred_cent)
            sigma_acc = sigma_acc + sigma_batch if sigma_acc is not None else sigma_batch

        sigma = sigma_acc / (n_samples - 1)
        return mu.cpu().numpy(), sigma.cpu().numpy()


def main(args):
    device = 'cuda' if pt.cuda.is_available() else 'cpu'

    data_path = os.path.join(os.getcwd(), '..', 'data/cifar10')

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    transformations = [transforms.Resize(32), transforms.ToTensor()]
    if args.test:
        print("fid for test dataset")
        dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=False,
            transform=transforms.Compose(transformations),
            download=True
        )
        name_f = 'cifar10_test_pytorch_fid.npz'
    else:
        print("fid for train dataset")
        dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            transform=transforms.Compose(transformations),
            download=True
        )
        name_f = 'cifar10_train_pytorch_fid.npz'

    if not os.path.exists(args.fid_dir):
        os.makedirs(args.fid_dir)

    file_path = os.path.join(args.fid_dir, name_f)
    dataloader = pt.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)

    mu_real, sig_real = stats_from_dataloader(dataloader, model, device)
    np.savez(file_path, mu=mu_real, sigma=sig_real)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size per GPU')
    parser.add_argument('--fid_dir', type=str, default='', help='Directory to store fid stats')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    set_seeds(0, 0)

    main(args)
