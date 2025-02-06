import functools
import gzip
import io
import json
import os
import pickle
import re
import sys
import tarfile
import zipfile
from pathlib import Path
import os
import argparse
from tqdm import tqdm
import PIL
import numpy as np
from typing import Callable, Optional, Tuple, Union
import functools
import json

import torch
import torchvision
from torchvision import transforms

from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor

new_cwd = os.path.dirname(os.getcwd())
sys.path.insert(0, new_cwd)
from data.stylegan3.dataset import ImageFolderDataset

import os
from PIL import Image

# resize images into the needed resolution
def resize_images(source_dir, target_dir, size=(32, 32)):

    for root, dirs, files in os.walk(source_dir):

        relative_path = os.path.relpath(root, source_dir)

        target_subdir = os.path.join(target_dir, relative_path)

        os.makedirs(target_subdir, exist_ok=True)

        for file in files:
            file_ext = file.lower().split('.')[-1]
            if file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff']:
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_subdir, file)
                with Image.open(source_file) as img:
                    img = img.convert('RGB')
                    img_resized = img.resize(size, Image.LANCZOS)
                    img_resized.save(target_file)
                # try:
                #     with Image.open(source_file) as img:
                #         img = img.convert('RGB')
                #         img_resized = img.resize(size, Image.LANCZOS)
                #         img_resized.save(target_file)
                # except Exception as e:
                #     print(f" {source_file} : {e}")


def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

class target_trans(object):
    def __init__(self, attr_index):
        self.idx = attr_index
        
    def __call__(self, attrs):
        return attrs[self.idx]

    def __repr__(self):
        return self.__class__.__name__

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        try:
            img = PIL.Image.fromarray(img, 'RGB')
        except:
            print(img.shape)
        img = img.resize((width, height), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

# preprocess and save the images
def data_save(data_loader, num_files, transform_image, archive_root_dir, save_bytes, close_dest):
    labels = []
    for idx, (image, label) in tqdm(enumerate(data_loader), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'
        
        label = int(label.long().numpy().astype('uint8')[0])

        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        image = (image * 255)[0].permute(1, 2, 0).numpy().astype('uint8')

        # Apply crop and resize.
        img = transform_image(image)
        image = image[..., :1]

        # Transform may drop images.
        if img is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            'width': img.shape[1],
            'height': img.shape[0],
            'channels': channels
        }

        dataset_attrs = cur_image_attrs
        width = dataset_attrs['width']
        height = dataset_attrs['height']
        if width != height:
            error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
        if dataset_attrs['channels'] not in [1, 3]:
            error('Input images must be stored as RGB or grayscale')
        # if width != 2 ** int(np.floor(np.log2(width))):
        #     error('Image width/height after scale and crop are required to be power-of-two')

        # Save the image as an uncompressed PNG.
        img = PIL.Image.fromarray(img, { 1: 'L', 3: 'RGB' }[channels])
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        labels.append([archive_fname, label] if label is not None else None)

    metadata = {
        'labels': labels if all(x is not None for x in labels) else None
    }
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

def get_activations(dl, model, device, max_samples):
    pred_arr = []
    total_processed = 0

    print('Starting to sample.')
    for batch in dl:
        # ignore labels
        if isinstance(batch, list):
            batch = batch[0]
        batch = batch.to(device)
        if batch.shape[1] == 1:  # if image is gray scale
            batch = batch.repeat(1, 3, 1, 1)
        elif len(batch.shape) == 3:  # if image is gray scale
            batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)
        
        pred = model.get_feature_batch(batch).cpu().numpy()

        pred_arr.append(pred)
        total_processed += pred.shape[0]
        if max_samples is not None and total_processed > max_samples:
            print('Max of %d samples reached.' % max_samples)
            break

    pred_arr = np.concatenate(pred_arr, axis=0)
    if max_samples is not None:
        pred_arr = pred_arr[:max_samples]

    return pred_arr

def fid_save(fid_path, dataset, batch_size):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    queue = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, pin_memory=True, num_workers=1)
    
    inception_model = InceptionFeatureExtractor()
    inception_model.model = inception_model.model.to(device)

    act = get_activations(queue, inception_model, device=device, max_samples=None)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    np.savez(fid_path, mu=mu, sigma=sigma)

def main(config):
    for data_name in config.data_name:   
        train_name = "train_{}".format(config.resolution)
        test_name = "test_{}".format(config.resolution)
        fid_name = "fid_stats_{}".format(config.resolution)

        data_dir = os.path.join(config.data_dir, data_name)
        os.makedirs(data_dir, exist_ok=True)
        if data_name == "mnist":
            sensitive_train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
            sensitive_test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
        elif data_name == "fmnist":
            sensitive_train_set = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
            sensitive_test_set = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
        elif data_name == "eurosat":
            sensitive_set = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "EuroSAT_RGB"), transform=transforms.ToTensor())
            torch.manual_seed(0)
            train_size = 23000
            test_size = 4000
            sensitive_train_set, sensitive_test_set = torch.utils.data.random_split(sensitive_set, [train_size, test_size])
        elif data_name == "cifar10":
            sensitive_train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
            sensitive_test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
        elif data_name == "cifar100":
            sensitive_train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
            sensitive_test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
        elif data_name == "celeba":
            train_name = train_name + '_' + config.celeba_attr
            test_name = test_name + '_' + config.celeba_attr

            with open(os.path.join(data_dir, "celeba", "list_attr_celeba.txt"), 'r') as f:
                attr_index = f.readlines()[1].strip().split().index(config.celeba_attr)
            target_transform = target_trans(attr_index)
            sensitive_train_set = torchvision.datasets.CelebA(root=data_dir, split="train",  download=False, transform=transforms.ToTensor(), target_transform=target_transform)
            sensitive_test_set = torchvision.datasets.CelebA(root=data_dir, split="test", download=False, transform=transforms.ToTensor(), target_transform=target_transform)
        elif data_name == "camelyon":
            sensitive_train_set = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "camelyon17_32", "train"), transform=transforms.ToTensor())
            sensitive_test_set = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, "camelyon17_32", "test"), transform=transforms.ToTensor())
        elif data_name == "places365":
            # _ = torchvision.datasets.Places365(root=data_dir, small=True, download=True)
            resize_images(os.path.join(data_dir, "data_256_standard"), os.path.join(data_dir, "data_{}_standard".format(config.resolution)), (config.resolution, config.resolution))
            return
        elif data_name == "emnist":
            _ = torchvision.datasets.EMNIST(root=data_dir, train=True, split="letters", download=True)
            return
        elif config.train_path != '' and config.test_path != '':
            sensitive_train_set = torchvision.datasets.ImageFolder(root=config.train_path, transform=transforms.ToTensor())
            sensitive_test_set = torchvision.datasets.ImageFolder(root=config.test_path, transform=transforms.ToTensor())
        else:
            raise NotImplementedError('{} is not yet implemented.'.format(data_name))

        sensitive_train_loader = torch.utils.data.DataLoader(dataset=sensitive_train_set, batch_size=1)
        sensitive_test_loader = torch.utils.data.DataLoader(dataset=sensitive_test_set, batch_size=1)

        if config.max_image is None:
            max_idx = len(sensitive_train_set)
        else:
            max_idx = min(config.max_image, len(sensitive_train_set))

        transform_image = make_transform(config.transform, config.resolution, config.resolution)
        
        if config.max_image is None:
            max_idx = len(sensitive_train_set)
        else:
            max_idx = min(config.max_image, len(sensitive_train_set))
        archive_root_dir, save_bytes, close_dest = open_dest(os.path.join(data_dir, train_name + '.zip'))
        data_save(sensitive_train_loader, max_idx, transform_image, archive_root_dir, save_bytes, close_dest)

        if config.max_image is None:
            max_idx = len(sensitive_test_set)
        else:
            max_idx = min(config.max_image, len(sensitive_test_set))
        archive_root_dir, save_bytes, close_dest = open_dest(os.path.join(data_dir, test_name + '.zip'))
        data_save(sensitive_test_loader, max_idx, transform_image, archive_root_dir, save_bytes, close_dest)

        dataset = ImageFolderDataset(os.path.join(data_dir, train_name + '.zip'))

        # calculate the feature of training image for FID metric
        fid_save(os.path.join(data_dir, fid_name + '.npz'), dataset, config.fid_batch_size)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', nargs="*", default=["mnist", "fmnist", "cifar10", "cifar100", "celeba", "camelyon", "imagenet", "places365", "emnist"], help='List of datasets to use. Default is all provided datasets.')
    parser.add_argument('--resolution', default=32, type=int, help='Resolution of the images. Default is 32.')
    parser.add_argument('--c', default=3, type=int, help='Number of color channels in the images. Default is 3 (RGB).')
    parser.add_argument('--fid_batch_size', default=500, type=int, help='Batch size for FID calculation. Default is 500.')
    parser.add_argument('--data_dir', default='../dataset', help='Directory where the datasets are stored. Default is ../dataset.')
    parser.add_argument('--transform', default="center-crop", help='Type of transform to apply to the data. Default is center-crop.')
    parser.add_argument('--celeba_attr', default="Male", help='Attribute to filter CelebA dataset. Default is Male.')
    parser.add_argument('--max_image', default=None, type=int, help='Maximum number of images to load. Default is None (load all).')
    parser.add_argument('--train_path', default='', help='Path to the custom training set. Leave empty if using default datasets.')
    parser.add_argument('--test_path', default='', help='Path to the custom test set. Leave empty if using default datasets.')
    config = parser.parse_args()
    main(config)
