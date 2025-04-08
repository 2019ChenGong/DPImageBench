import torch
import numpy as np
from tqdm import tqdm
import logging

from .api import API
from models.PE.pe.arg_utils import str2bool

from improved_diffusion import dist_util

from models.DP_Diffusion.model.ncsnpp import NCSNpp
from models.DP_Diffusion.model.ema import ExponentialMovingAverage
from models.DP_Diffusion.denoiser import EDMDenoiser
from models.DP_Diffusion.samplers import ddim_sampler, edm_sampler
from models.DP_Diffusion.generate_base import generate_batch
from .improved_diffusion.gaussian_diffusion import create_gaussian_diffusion


def _round_to_uint8(image):
    return np.around(np.clip(image, a_min=0, a_max=255)).astype(np.uint8)


class SDAPI(API):
    def __init__(self, ckpt, denoiser_name, denoiser_network,
                 ema_rate, network, sampler, batch_size, use_data_parallel=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize the denoiser based on the specified name and network
        self.network = network
        if denoiser_name == 'edm':
            if denoiser_network == 'song':
                self._model = EDMDenoiser(NCSNpp(**network).to(dist_util.dev()))  # Initialize EDM denoiser with NCSNpp network
            else:
                raise NotImplementedError("Network type not supported for EDM denoiser")
        else:
            raise NotImplementedError("Denoiser name not recognized")

        self._ema = ExponentialMovingAverage(self._model.parameters(), decay=ema_rate)  # Initialize EMA for the model parameters
        self._model.eval()

        # Load checkpoint if provided
        assert ckpt is not None

        state = torch.load(ckpt, map_location="cpu")  # Load the checkpoint
        new_state_dict = {}
        for k, v in state['model'].items():
            new_state_dict[k[7:]] = v  # Adjust the keys to match the model's state dictionary
        logging.info(self._model.load_state_dict(new_state_dict, strict=True))  # Load the state dictionary into the model
        self._ema.load_state_dict(state['ema'])  # Load the EMA state dictionary
        self._ema.restore(self._model.parameters())
        del state, new_state_dict, self._ema  # Clean up memory
        self._model = self._model.to(dist_util.dev())
        if use_data_parallel:
            self._model = torch.nn.DataParallel(self._model)

        def sampler_fn(x, y=None, start_sigma=None, start_t=None):
            if sampler.type == 'ddim':
                return ddim_sampler(x, y, self._model, start_sigma=start_sigma, start_t=start_t, **sampler)
            elif sampler.type == 'edm':
                return edm_sampler(x, y, self._model, **sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        self._sampler = sampler_fn
        self._batch_size = batch_size
        self._image_size = network.image_size
        self._class_cond = True


        self._diffusion = create_gaussian_diffusion(
            steps=1000,
            learn_sigma=True,
            noise_schedule="cosine",
            timestep_respacing=str(sampler.num_steps),)
        self._sigma_list = self._diffusion.sqrt_one_minus_alphas_cumprod / self._diffusion.sqrt_alphas_cumprod
        # logging.info(str(self._sigma_list))

    def image_random_sampling(self, num_samples, size, prompts, labels=None):
        """
        Generates a specified number of random image samples based on a given
        prompt and size using OpenAI's Image API.

        Args:
            num_samples (int):
                The number of image samples to generate.
            size (str, optional):
                The size of the generated images in the format
                "widthxheight". Options include "256x256", "512x512", and
                "1024x1024".
            prompts (List[str]):
                The text prompts to generate images from. Each promot will be
                used to generate num_samples/len(prompts) number of samples.

        Returns:
            numpy.ndarray:
                A numpy array of shape [num_samples x width x height x
                channels] with type np.uint8 containing the generated image
                samples as numpy arrays.
            numpy.ndarray:
                A numpy array with length num_samples containing labels for
                each image.
        """
        samples = []
        labels = []
        sampling_shape = (self._batch_size, self.network.num_in_channels, self._image_size, self._image_size)
        with torch.no_grad():
            for i in range(num_samples//self._batch_size+1):
                # x, y = generate_batch(self._sampler, sampling_shape, dist_util.dev(), self.network.label_dim, self.network.label_dim)
                x, y = generate_batch(self._sampler, sampling_shape, dist_util.dev(), 1, 1)
                # x, y = generate_batch(self._sampler, sampling_shape, dist_util.dev(), None, None)
                samples.append(x.detach().cpu())
                labels.append(y.detach().cpu())
                logging.info(f"Created {(i+1)*self._batch_size} samples")
        samples = torch.cat(samples)[:num_samples].numpy()
        labels = torch.cat(labels)[:num_samples].numpy()

        samples = _round_to_uint8(samples * 255.)
        samples = samples.transpose(0, 2, 3, 1)
        torch.cuda.empty_cache()
        return samples, labels

    def image_variation(self, images, additional_info,
                        num_variations_per_image, size, variation_degree):
        """
        Generates a specified number of variations for each image in the input
        array using OpenAI's Image Variation API.

        Args:
            images (numpy.ndarray):
                A numpy array of shape [num_samples x width x height
                x channels] containing the input images as numpy arrays of type
                uint8.
            additional_info (numpy.ndarray):
                A numpy array with the first dimension equaling to
                num_samples containing labels provided by
                image_random_sampling.
            num_variations_per_image (int):
                The number of variations to generate for each input image.
            size (str):
                The size of the generated image variations in the
                format "widthxheight". Options include "256x256", "512x512",
                and "1024x1024".
            variation_degree (int):
                The diffusion step to add noise to the images to before running
                the denoising steps. The value should between 0 and
                timestep_respacing-1. 0 means the step that is closest to
                noise. timestep_respacing-1 means the step that is closest to
                clean image. A smaller value will result in more variation.

        Returns:
            numpy.ndarray:
                A numpy array of shape [num_samples x num_variations_per_image
                x width x height x channels] containing the generated image
                variations as numpy arrays of type uint8.
        """
        # width, height = list(map(int, size.split('x')))
        # if width != self._image_size or height != self._image_size:
        #     raise ValueError(
        #         f'width and height must be equal to {self._image_size}')
        images = images.astype(np.float32) / 127.5 - 1.0
        images = images.transpose(0, 3, 1, 2)
        variations = []
        for _ in tqdm(range(num_variations_per_image)):
            sub_variations = self._image_variation(
                images=images,
                labels=additional_info,
                variation_degree=variation_degree)
            variations.append(sub_variations)
        variations = np.stack(variations, axis=1)

        # variations = _round_to_uint8((variations + 1.0) * 127.5)
        # variations = variations.transpose(0, 1, 3, 4, 2)
        torch.cuda.empty_cache()
        return variations

    def _image_variation(self, images, labels, variation_degree):
        samples = []
        images_list = torch.split(torch.Tensor(images), split_size_or_sections=self._batch_size)
        labels_list = torch.split(torch.LongTensor(labels), split_size_or_sections=self._batch_size)
        for i in range(len(images_list)):
            with torch.no_grad():
                # x = self._sampler(images_list[i].to(dist_util.dev()), labels_list[i].to(dist_util.dev()), start_t=variation_degree/1000)
                x = self._sampler(images_list[i].to(dist_util.dev()), labels_list[i].to(dist_util.dev()), start_sigma=self._sigma_list[len(self._sigma_list)-1-variation_degree])
            samples.append(x.detach().cpu())
            logging.info(f"Created {(i+1)*self._batch_size} samples")
        samples = torch.cat(samples).clamp(-1., 1.).numpy()

        samples = _round_to_uint8((samples + 1.0) * 127.5)
        samples = samples.transpose(0, 2, 3, 1)
        return samples


# def sample(sampler, num_samples, start_t, batch_size, use_ddim,
#            image_size, clip_denoised, class_cond,
#            start_image=None, labels=None):
#     all_images = []
#     all_labels = []
#     batch_cnt = 0
#     cnt = 0
#     while cnt < num_samples:
#         current_batch_size = \
#             (batch_size if start_image is None
#              else min(batch_size,
#                       start_image.shape[0] - batch_cnt * batch_size))
#         shape = (current_batch_size, 3, image_size, image_size)
#         model_kwargs = {}
#         if class_cond:
#             if labels is None:
#                 classes = torch.randint(
#                     low=0, high=NUM_CLASSES, size=(current_batch_size,),
#                     device=dist_util.dev()
#                 )
#             else:
#                 classes = labels[batch_cnt * batch_size:
#                                  (batch_cnt + 1) * batch_size].to(dist_util.dev())
#             model_kwargs["y"] = classes
#         sample = sampler(
#             clip_denoised=clip_denoised,
#             model_kwargs=model_kwargs,
#             start_t=max(start_t, 0),
#             start_image=(None if start_image is None
#                          else start_image[batch_cnt * batch_size:
#                                           (batch_cnt + 1) * batch_size]),
#             use_ddim=use_ddim,
#             noise=torch.randn(*shape, device=dist_util.dev()),
#             image_size=image_size)
#         batch_cnt += 1

#         all_images.append(sample.detach().cpu().numpy())

#         if class_cond:
#             all_labels.append(classes.detach().cpu().numpy())

#         cnt += sample.shape[0]
#         logging.info(f"Created {cnt} samples")

#     all_images = np.concatenate(all_images, axis=0)
#     all_images = all_images[: num_samples]

#     if class_cond:
#         all_labels = np.concatenate(all_labels, axis=0)
#         all_labels = all_labels[: num_samples]
#     else:
#         all_labels = np.zeros(shape=(num_samples,))
#     return all_images, all_labels


# class Sampler(torch.nn.Module):
#     """
#     A wrapper around the model and diffusion modules that handles the entire
#     sampling process, so as to reduce the communiation rounds between GPUs when
#     using DataParallel.
#     """
#     def __init__(self, model, diffusion):
#         super().__init__()
#         self._model = model
#         self._diffusion = diffusion

#     def forward(self, clip_denoised, model_kwargs, start_t, start_image,
#                 use_ddim, noise, image_size):
#         sample_fn = (
#             self._diffusion.p_sample_loop if not use_ddim
#             else self._diffusion.ddim_sample_loop)
#         sample = sample_fn(
#             self._model,
#             (noise.shape[0], 3, image_size, image_size),
#             clip_denoised=clip_denoised,
#             model_kwargs=model_kwargs,
#             start_t=max(start_t, 0),
#             start_image=start_image,
#             noise=noise,
#             device=noise.device)
#         return sample
