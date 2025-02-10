import argparse, os, sys, datetime, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from math import ceil
import torch
# import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from einops import rearrange
from torchvision.utils import make_grid
from PIL import Image

import argparse
# import os


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
    # Training settings
    parser = argparse.ArgumentParser(
        # description="Opacus MNIST Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-y", "--yaml", type=str, default=None, help="load the yaml from the specified path")
    parser.add_argument("-ckpt", "--ckpt_path", type=str, default=None, help="load the checkpoint from the specified path")
    parser.add_argument("-step", "--ddim_step", type=int, default=200, help="number of steps for ddim sampling")
    parser.add_argument("-eta", "--eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)")
    parser.add_argument("-scale", "--scale", type=float, default=1.0, help="scale for ddim sampling (unconditional_guidance_scale=scale)")
    parser.add_argument("-n", "--num_samples", type=int, default=60000, help="number of samples to generate")
    parser.add_argument("-c", "--num_classes", type=int, default=10, help="class/label numbers you want to generate for, e.g. 0 1 3 5")
    parser.add_argument("-bs", "--batch_size", type=int, default=500, help="mini batch size of sampling (to avoid cuda out of memory, change to a smaller value if needed)")
    parser.add_argument("--save_path", type=str, default='.', help="load the checkpoint from the specified path")

    # args = parser.parse_args()

    args, unknown = parser.parse_known_args()

    configs = [OmegaConf.load(args.yaml)]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    model = load_model_from_config(config, args.ckpt_path)
    ddim_steps = args.ddim_step
    ddim_eta = args.eta
    scale = args.scale
    num_samples = args.num_samples
    classes = [i for i in range(args.num_classes)]
    batch_size = args.batch_size

    n_samples_per_class = int(num_samples / len(classes))
    sampler = DDIMSampler(model)

    shape = [model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    all_samples = list()

    with torch.no_grad():
        for class_label in classes:
            print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
            xc = torch.tensor([class_label])
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

            batch_size_temp = batch_size
            n_iters = ceil(n_samples_per_class / batch_size)
            for idx in range(n_iters):
                if idx == n_iters - 1 and n_samples_per_class % batch_size != 0: batch_size_temp = n_samples_per_class % batch_size
                samples_ddim, _ = sampler.sample(
                    S=ddim_steps,
                    conditioning=c.repeat(batch_size_temp, 1, 1),
                    batch_size=batch_size_temp,
                    shape=shape,
                    verbose=False,
                    eta=ddim_eta
                )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)

    grid = torch.stack(all_samples, 0)
    grid = grid.reshape(args.num_samples, grid.shape[-3], grid.shape[-2], grid.shape[-1])
    # grid_to_plot = rearrange(grid, 'n b c h w -> (n b) c h w')

    # grid_to_plot = make_grid(grid_to_plot, nrow=4)
    # # to image
    # grid_to_plot = 255. * rearrange(grid_to_plot, 'c h w -> h w c').detach().cpu().numpy()
    # plotted_imgs = Image.fromarray(grid_to_plot.astype(np.uint8))
    # plotted_imgs.save("test_class_cond.jpg")

    labels = np.array(classes)
    labels = np.repeat(labels, n_samples_per_class)
    labels = torch.tensor(labels)

    dic = {'image': grid,
           'class_label': labels}
    
    syn_data = grid.detach().cpu().numpy()
    syn_labels = labels.numpy()
    np.savez(os.path.join(args.save_path, "gen.npz"), x=syn_data, y=syn_labels)

    # torch.save(dic, 'conditional_mnist_samples.pt')


if __name__ == "__main__":
    main()