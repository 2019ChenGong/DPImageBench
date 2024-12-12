from math import ceil
import torch
import numpy as np
from PIL import Image

from models.ldm.models.diffusion.ddim import DDIMSampler
from models.ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def generate_batch(config, num_samples, model_path, ddim_steps=200, ddim_eta=1.0, scale=1.0, num_classes=10, batch_size=500):
    # Training settings

    model = load_model_from_config(config, model_path)
    classes = [i for i in range(num_classes)]

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

            batch_size_temp = min(batch_size, n_samples_per_class)
            n_iters = ceil(n_samples_per_class / batch_size)
            for idx in range(n_iters):
                if idx == batch_size_temp - 1: batch_size_temp = n_samples_per_class % batch_size
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
                all_samples.append(x_samples_ddim[:n_samples_per_class])

    grid = torch.stack(all_samples, 0)

    labels = np.array(classes)
    labels = np.repeat(labels, n_samples_per_class)
    labels = torch.tensor(labels)
    
    syn_data = grid.detach().cpu().numpy()
    syn_data = syn_data.reshape(-1, syn_data.shape[-3], syn_data.shape[-2], syn_data.shape[-1])
    syn_labels = labels.numpy()
    return syn_data, syn_labels
