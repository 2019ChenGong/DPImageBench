import argparse
import logging
import os
import numpy as np
import imageio
from torchvision.utils import make_grid
import torchvision
import torch
import scipy
from models.DPSDA.dpsda.feature_extractor import extract_features
from models.DPSDA.dpsda.metrics import make_fid_stats
from models.DPSDA.dpsda.metrics import compute_fid
from models.DPSDA.dpsda.dp_counter import dp_nn_histogram
from models.DPSDA.dpsda.arg_utils import str2bool
from models.DPSDA.apis import get_api_class_from_name

import logging


from models.synthesizer import DPSynther


def get_noise_multiplier(epsilon, num_steps, delta, min_noise_multiplier=1e-1, max_noise_multiplier=500, max_epsilon=1e7):

    def delta_Gaussian(eps, mu):
        """Compute delta of Gaussian mechanism with shift mu or equivalently noise scale 1/mu"""
        if mu == 0:
            return 0
        return scipy.stats.norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * scipy.stats.norm.cdf(-eps / mu - mu / 2)

    def eps_Gaussian(delta, mu):
        """Compute eps of Gaussian mechanism with shift mu or equivalently noise scale 1/mu"""
        def f(x):
            return delta_Gaussian(x, mu) - delta
        return scipy.optimize.root_scalar(f, bracket=[0, max_epsilon], method='brentq').root

    def compute_epsilon(noise_multiplier, num_steps, delta):
        return eps_Gaussian(delta, np.sqrt(num_steps) / noise_multiplier)

    def objective(x):
        return compute_epsilon(noise_multiplier=x, num_steps=num_steps, delta=delta) - epsilon

    output = scipy.optimize.root_scalar(objective, bracket=[min_noise_multiplier, max_noise_multiplier], method='brentq')
    if not output.converged:
        raise ValueError("Failed to converge")

    return output.root


class DPSDA(DPSynther):
    def __init__(self, config, device):
        super().__init__()
        api_class = get_api_class_from_name(config.api)
        # self.api = api_class(**config.api_params)
        api_args = []
        for k in config.api_params:
            api_args.append('--' + k)
            api_args.append(str(config.api_params[k]))
        # print(api_args)
        self.api = api_class.from_command_line_args(api_args)
        self.feature_extractor = config.feature_extractor
        self.samples = None
        self.labels = None

    def train(self, sensitive_dataloader, config):
        os.mkdir(config.log_dir)
        tmp_folder = config.tmp_folder

        self.noise_factor = get_noise_multiplier(epsilon=config.dp.epsilon, delta=config.dp.delta, num_steps=len(config.num_samples_schedule) - 1)

        logging.info("The noise factor is {}".format(self.noise_factor))

        all_private_samples = []
        all_private_labels = []
        for x, y in sensitive_dataloader:
            if len(y.shape) == 2:
                x = x.to(torch.float32) / 255.
                y = torch.argmax(y, dim=1)
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            all_private_samples.append(x.cpu().numpy())
            all_private_labels.append(y.cpu().numpy())
        all_private_samples = np.concatenate(all_private_samples)
        all_private_labels = np.concatenate(all_private_labels)

        all_private_samples = np.around(np.clip(all_private_samples * 255, a_min=0, a_max=255)).astype(np.uint8)
        all_private_samples = np.transpose(all_private_samples, (0, 2, 3, 1))

        private_classes = list(sorted(set(list(all_private_labels))))
        private_num_classes = len(private_classes)

        logging.info('Extracting features')
        all_private_features = extract_features(
            data=all_private_samples,
            tmp_folder=tmp_folder,
            model_name=self.feature_extractor,
            num_workers=2,
            res=config.private_image_size,
            batch_size=config.feature_extractor_batch_size)
        logging.info(f'all_private_features.shape: {all_private_features.shape}')

        # Generating initial samples.
        logging.info('Generating initial samples')

        labels = None

        samples, additional_info = self.api.image_random_sampling(
            prompts=config.initial_prompt,
            num_samples=config.num_samples_schedule[0],
            size=config.image_size,
            labels=labels)
        log_samples(
            samples=samples,
            additional_info=additional_info,
            folder=f'{config.log_dir}/{0}',
            plot_images=False)

        start_t = 1

        for t in range(start_t, len(config.num_samples_schedule)):
            logging.info(f't={t}')
            assert samples.shape[0] % private_num_classes == 0
            num_samples_per_class = samples.shape[0] // private_num_classes

            if config.lookahead_degree == 0:
                packed_samples = np.expand_dims(samples, axis=1)
            else:
                logging.info('Running image variation')
                packed_samples = self.api.image_variation(
                    images=samples,
                    additional_info=additional_info,
                    num_variations_per_image=config.lookahead_degree,
                    size=config.image_size,
                    variation_degree=config.variation_degree_schedule[t])

            packed_features = []
            logging.info('Running feature extraction')
            for i in range(packed_samples.shape[1]):
                sub_packed_features = extract_features(
                    data=packed_samples[:, i],
                    tmp_folder=tmp_folder,
                    num_workers=2,
                    model_name=self.feature_extractor,
                    res=config.private_image_size,
                    batch_size=config.feature_extractor_batch_size)
                logging.info(
                    f'sub_packed_features.shape: {sub_packed_features.shape}')
                packed_features.append(sub_packed_features)
            packed_features = np.mean(packed_features, axis=0)

            logging.info('Computing histogram')
            count = []
            for class_i, class_ in enumerate(private_classes):
                sub_count, sub_clean_count = dp_nn_histogram(
                    public_features=packed_features[
                        num_samples_per_class * class_i:
                        num_samples_per_class * (class_i + 1)],
                    private_features=all_private_features[
                        all_private_labels == class_],
                    noise_multiplier=self.noise_factor,
                    num_nearest_neighbor=config.num_nearest_neighbor,
                    mode=config.nn_mode,
                    threshold=config.count_threshold)
                log_count(
                    sub_count,
                    sub_clean_count,
                    f'{config.log_dir}/{t}/count_class{class_}.npz')
                count.append(sub_count)
            count = np.concatenate(count)
            for class_i, class_ in enumerate(private_classes):
                visualize(
                    samples=samples[
                        num_samples_per_class * class_i:
                        num_samples_per_class * (class_i + 1)],
                    packed_samples=packed_samples[
                        num_samples_per_class * class_i:
                        num_samples_per_class * (class_i + 1)],
                    count=count[
                        num_samples_per_class * class_i:
                        num_samples_per_class * (class_i + 1)],
                    folder=f'{config.log_dir}/{t}',
                    suffix=f'class{class_}')

            logging.info('Generating new indices')
            assert config.num_samples_schedule[t] % config.private_num_classes == 0
            new_num_samples_per_class = (
                config.num_samples_schedule[t] // config.private_num_classes)
            new_indices = []
            for class_i in private_classes:
                sub_count = count[
                    num_samples_per_class * class_i:
                    num_samples_per_class * (class_i + 1)]
                sub_new_indices = np.random.choice(
                    np.arange(num_samples_per_class * class_i,
                            num_samples_per_class * (class_i + 1)),
                    size=new_num_samples_per_class,
                    p=sub_count / np.sum(sub_count))
                new_indices.append(sub_new_indices)
            new_indices = np.concatenate(new_indices)
            new_samples = samples[new_indices]
            additional_info = additional_info[new_indices]

            logging.info('Generating new samples')
            samples = self.api.image_variation(
                images=new_samples,
                additional_info=additional_info,
                num_variations_per_image=1,
                size=config.image_size,
                variation_degree=config.variation_degree_schedule[t])
            samples = np.squeeze(samples, axis=1)

            # if args.compute_fid:
            #     logging.info('Computing FID')
            #     new_new_fid = compute_fid(
            #         new_new_samples,
            #         tmp_folder=args.tmp_folder,
            #         num_fid_samples=args.num_fid_samples,
            #         dataset_res=args.private_image_size,
            #         dataset=args.fid_dataset_name,
            #         dataset_split=args.fid_dataset_split,
            #         model_name=args.fid_model_name,
            #         batch_size=args.fid_batch_size)
            #     logging.info(f'fid={new_new_fid}')
            #     log_fid(args.result_folder, new_new_fid, t)

            if t == len(config.num_samples_schedule) - 1:
                log_samples(
                    samples=samples,
                    additional_info=additional_info,
                    folder=f'{config.log_dir}/{t}',
                    plot_images=False)
        
        self.samples = np.transpose(samples.astype('float'), (0, 3, 1, 2)) / 255.
        self.labels = np.concatenate([[cls] * num_samples_per_class for cls in private_classes])

    def generate(self, config):
        os.mkdir(config.log_dir)
        syn_data = self.samples
        syn_labels = self.labels

        np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

        show_images = []
        num_class = len(set(list(syn_labels)))
        for cls in range(num_class):
            show_images.append(syn_data[syn_labels==cls][:8])
        show_images = np.concatenate(show_images)
        torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)
        return syn_data, syn_labels


def log_samples(samples, additional_info, folder, plot_images):
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savez(
        os.path.join(folder, 'samples.npz'),
        samples=samples,
        additional_info=additional_info)
    if plot_images:
        for i in range(samples.shape[0]):
            imageio.imwrite(os.path.join(folder, f'{i}.png'), samples[i])

def log_count(count, clean_count, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.savez(path, count=count, clean_count=clean_count)


def visualize(samples, packed_samples, count, folder, suffix=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    samples = samples.transpose((0, 3, 1, 2))
    packed_samples = packed_samples.transpose((0, 1, 4, 2, 3))

    ids = np.argsort(count)[::-1][:5]
    print(count[ids])
    vis_samples = []
    for i in range(len(ids)):
        vis_samples.append(samples[ids[i]])
        for j in range(packed_samples.shape[1]):
            vis_samples.append(packed_samples[ids[i]][j])
    vis_samples = np.stack(vis_samples)
    vis_samples = make_grid(
        torch.Tensor(vis_samples),
        nrow=packed_samples.shape[1] + 1,
        padding=0).numpy().transpose((1, 2, 0))
    vis_samples = round_to_uint8(vis_samples)
    imageio.imsave(
        os.path.join(folder, f'visualize_top_{suffix}.png'), vis_samples)

    ids = np.argsort(count)[:5]
    print(count[ids])
    vis_samples = []
    for i in range(len(ids)):
        vis_samples.append(samples[ids[i]])
        for j in range(packed_samples.shape[1]):
            vis_samples.append(packed_samples[ids[i]][j])
    vis_samples = np.stack(vis_samples)
    vis_samples = make_grid(
        torch.Tensor(vis_samples),
        nrow=packed_samples.shape[1] + 1,
        padding=0).numpy().transpose((1, 2, 0))
    vis_samples = round_to_uint8(vis_samples)
    imageio.imsave(
        os.path.join(folder, f'visualize_bottom_{suffix}.png'), vis_samples)


def round_to_uint8(image):
    return np.around(np.clip(image, a_min=0, a_max=255)).astype(np.uint8)