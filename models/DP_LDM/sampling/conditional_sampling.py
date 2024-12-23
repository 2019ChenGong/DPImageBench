import argparse
import random

import numpy as np
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
import torch
from torchvision.utils import save_image

from ..ldm.util import instantiate_from_config
from ..ldm.models.diffusion.ddim import DDIMSampler


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_model_from_config(config, ckpt):
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    try:
        config.model.params.ignore_keys = []
        config.model.params.ckpt_path = None
    except ConfigAttributeError:
        pass
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_uc(model, uc_class):
    x = {model.cond_stage_key: torch.tensor([uc_class], device=model.device)}
    return model.get_learned_conditioning(x)


def get_c(model, args):
    if args.prompt is not None:
        return model.get_learned_conditioning([args.prompt])
    if args.classes is not None:
        all_classes = torch.tensor(args.classes, device=model.device)
        all_conditioning = []
        for batch_classes in all_classes.split(args.batch_size):
            c = model.get_learned_conditioning({model.cond_stage_key: batch_classes})
            all_conditioning.append(c)
        return torch.vstack(all_conditioning)
    else:
        return None


def main(args):
    print("[INFO]: Set all seeds to", args.seed)
    set_seeds(args.seed)

    config = OmegaConf.load(args.yaml)
    model = load_model_from_config(config, args.ckpt)

    num_samples_per_class = args.num_samples // len(args.classes)
    print(f"[INFO]: Will generate {num_samples_per_class} samples per class for {len(args.classes)} classes")

    with model.ema_scope():
        # Get the conditioning information for each label
        conditioning = get_c(model, args)
        print("[INFO]: Got conditioning information of shape", conditioning.shape)

        # Get unconditional conditioning
        uc = get_uc(model, args.uc) if args.uc is not None else None

        # Set up sampler
        sampler = DDIMSampler(model)
        shape = [model.model.diffusion_model.in_channels,
                 model.model.diffusion_model.image_size,
                 model.model.diffusion_model.image_size]
        all_samples = []

        # Sample images
        class_indices = torch.tensor(range(len(args.classes))).repeat_interleave(num_samples_per_class)
        for batch_classes in class_indices.split(args.batch_size):
            batch_size = batch_classes.shape[0]
            batch_conditioning = conditioning[batch_classes]
            batch_uc = uc.repeat_interleave(batch_size, dim=0) if args.uc is not None else None
            samples, _ = sampler.sample(S=args.ddim_steps,
                                        batch_size=batch_size,
                                        shape=shape,
                                        conditioning=batch_conditioning,
                                        unconditional_conditioning=batch_uc,
                                        unconditional_guidance_scale=args.uc_scale,
                                        verbose=False,
                                        eta=args.eta)
            for batch_samples in samples.split(args.decoder_batch_size or batch_size):
                batch_samples = model.decode_first_stage(batch_samples).cpu()
                all_samples.append(batch_samples)

    # Collate images and save
    images = torch.vstack(all_samples)
    if args.output.endswith(".pt"):
        labels = torch.tensor(args.classes).repeat_interleave(num_samples_per_class)
        result = {'image': images, 'class_label': labels}
        torch.save(result, args.output)
    elif args.output.endswith(".png"):
        save_image(images * 0.5 + 0.5, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--yaml", type=str, required=True, help="Path to the model YAML file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to output file (.pt|.png)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=20, help="Total number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=500, help="Number of samples to generate per batch")
    parser.add_argument("--decoder_batch_size", type=int, default=None, help="Number of samples to decode per batch")
    grp_cond = parser.add_argument_group("Conditional Generation")
    grp_cond.add_argument("--classes", type=int, nargs="+", default=[0], help="Classes to generate samples for")
    grp_cond.add_argument("--prompt", type=str, nargs="?", default=None, help="A prompt to generate samples for")
    grp_cond.add_argument("--uc", type=int, default=None, help="Class to use for unconditional guidance")
    grp_cond.add_argument("--uc_scale", type=float, default=None, help="Unconditional guidance scale")
    grp_ddim = parser.add_argument_group("DDIM Options")
    grp_ddim.add_argument("--ddim_steps", type=int, default=200, help="number of steps for ddim sampling")
    grp_ddim.add_argument("--eta", type=float, default=1.0, help="eta for ddim sampling")
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
