"""
This script samples from the given LatentDiffusion model with the DDIM sampler.
The samples are saved to a file with `torch.save` for use in other scripts, such
as FID computation.
"""
import argparse
from math import ceil
import random

import numpy as np
from omegaconf import OmegaConf
import torch

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default=None, help="Path to the config file for the model")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("-o", "--output", type=str, help="Path to the output file (.pt)")
    parser.add_argument("--seed", type=int, default=None, help="Number of DDIM steps")
    parser.add_argument("--ddim_steps", type=int, default=200, help="Number of DDIM steps")
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for DDIM sampling (0.0 yields deterministic sampling)")
    parser.add_argument("--num_samples", type=int, default=5000, help="Total number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=500, help="Number of samples to generate per batch")
    args = parser.parse_args()

    if args.seed is not None:
        set_seeds(args.seed)

    config = OmegaConf.load(args.yaml)
    model = load_model_from_config(config, args.ckpt)
    sampler = DDIMSampler(model)

    shape = (model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size)

    samples_list = list()
    batch_size = args.batch_size
    iters = ceil(args.num_samples / args.batch_size)
    for i in range(iters):
        if args.num_samples % args.batch_size != 0 and i == iters - 1:
            batch_size = args.num_samples % args.batch_size
        samples, _ = sampler.sample(
            S=args.ddim_steps,
            batch_size=batch_size,
            shape=shape,
            eta=args.ddim_eta,
            verbose=False
        )
        samples = model.decode_first_stage(samples)
        samples_list.append(samples.cpu())

    all_samples = torch.vstack(samples_list)
    dic = {"image": all_samples}
    torch.save(dic, args.output)


if __name__ == "__main__":
    main()
