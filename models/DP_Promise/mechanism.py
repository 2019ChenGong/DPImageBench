import argparse

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf

from scipy import optimize
from scipy.stats import norm
from math import sqrt
import numpy as np


# Dual between mu-GDP and (epsilon,delta)-DP
def delta_eps_mu(eps, mu):
    return norm.cdf(-eps / mu +
                    mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


# inverse Dual
def eps_from_mu(mu, delta):

    def f(x):
        return delta_eps_mu(x, mu) - delta

    return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root


def gdp_mech(sample_rate1, sample_rate2, niter1, niter2, sigma,
             alpha_cumprod_S, d, delta):
    mu_1 = sample_rate1 * sqrt(niter1 * (np.exp(4 * d * alpha_cumprod_S / (1 - alpha_cumprod_S)) - 1))
    mu_2 = sample_rate2 * sqrt(niter2 * (np.exp(1 / (sigma ** 2)) - 1))
    mu = sqrt(mu_1 ** 2 + mu_2 ** 2)
    epsilon = eps_from_mu(mu, delta)
    return epsilon



def get_noise_multiplier(config, q1, q2, niter1, niter2, image_shape=(3, 32, 32), epsilon_tolerance: float = 0.01,
) -> float:
    MAX_SIGMA = 1e6

    def eps_cal(sigm_in):
        d = image_shape[0] * image_shape[1] * image_shape[2]

        prob1 = q1
        prob2 = q2

        alpha_cumprod_S = 1 / (1 + config.dp.gaussian_max ** 2)

        epsilon = gdp_mech(
            sample_rate1=prob1,
            sample_rate2=prob2,
            niter1=niter1,
            niter2=niter2,
            sigma=sigm_in,
            alpha_cumprod_S=alpha_cumprod_S,
            d=d,
            delta=config.dp.delta,
        )

        return epsilon

    eps_high = float("inf")

    sigma_low, sigma_high = 0, 10
    target_epsilon = config.dp.epsilon
    while eps_high > target_epsilon:
        sigma_high = 2 * sigma_high
        eps_high = eps_cal(sigma_high)

        if sigma_high > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        
        eps = eps_cal(sigma)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    print("final sigma: ", sigma_high)
    return sigma_high

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    opt, _ = parser.parse_known_args()
    # config = OmegaConf.load(opt.config)

    # delta = config.dp.delta
    # eps = eps_from_config(config)
    # print(f"(epsilon, delta) = ({eps}, {delta})")