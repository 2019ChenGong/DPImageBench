import numpy as np
import argparse
from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    m = np.square(mu1 - mu2).sum()
    s, _ = linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)
    fd = np.real(m + np.trace(sigma1 + sigma2 - s * 2))
    return fd


def main(args):
    stats1 = np.load(args.path1)
    stats1_mu = stats1['mu']
    stats1_sigma = stats1['sigma']
    stats2 = np.load(args.path2)
    stats2_mu = stats2['mu']
    stats2_sigma = stats2['sigma']

    fid = calculate_frechet_distance(stats1_mu, stats1_sigma, stats2_mu, stats2_sigma)
    print('FID: %.4f' % fid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=str, required=True)
    parser.add_argument('--path2', type=str, required=True)
    args = parser.parse_args()

    main(args)
