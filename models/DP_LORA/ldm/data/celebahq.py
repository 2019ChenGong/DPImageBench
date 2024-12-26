import os
import pickle

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import center_crop, resize, pil_to_tensor

from taming.data.imagenet import download


class CelebAHQ(Dataset):
    NUM_BINARY_ATTRS = 40

    def __init__(self,
                 datadir="CelebAHQ",
                 split="all",
                 target_type=None,
                 size=256,
                 class_attrs=None,
                 **kwargs):
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        self.root = os.path.join(cachedir, datadir)
        self.split = split
        self.target_type = target_type
        self.image_size = size
        self.class_attrs = class_attrs

        self._prepare_split()
        self._prepare_celebahq_to_celeba()
        self._prepare_target()

    def __len__(self):
        return len(self.split_celebahq)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, "images", f"{index}.jpg")
        image = Image.open(img_path)
        image = pil_to_tensor(image)
        image = resize(image, self.image_size, antialias=True)
        image = center_crop(image, self.image_size)
        image = image / 127.5 - 1
        image = image.permute(1, 2, 0).contiguous()
        item = {"image": image}

        if self.target_type == "attr":
            item["attr"] = self.idx_to_attr[index]
        elif self.target_type == "class":
            item["class_label"] = self.get_class(index)
        elif self.target_type == "caption":
            item["caption"] = self.idx_to_caption[index]

        return item

    def _prepare_split(self):
        if self.split == "train":
            with open(os.path.join(self.root, "train_filenames.pickle"), "rb") as f:
                self.split_celebahq = [f"img{i:>08}.png" for i in pickle.load(f)]
        elif self.split == "test":
            with open(os.path.join(self.root, "test_filenames.pickle"), "rb") as f:
                self.split_celebahq = [f"img{i:>08}.png" for i in pickle.load(f)]
        elif self.split == "all":
            self.split_celebahq = os.listdir(os.path.join(self.root, "images"))
        else:
            raise ValueError("split must be one of ('train', 'test', 'all')")

    def _prepare_celebahq_to_celeba(self):
        """Initializes mapping from CelebAHQ indices to CelebA indices"""
        with open(os.path.join(self.root, "image_list.txt")) as f:
            f.readline()  # Skip the header line
            self.celebahq_to_celeba = {}
            for line in f:
                line = line.split()
                self.celebahq_to_celeba[f"img{line[0]:>08}.png"] = line[2]

    def _prepare_target(self):
        if self.target_type == "attr":
            self._prepare_attr()
        elif self.target_type == "class":
            self._prepare_class()
        elif self.target_type == "caption":
            self._prepare_caption()
        elif self.target_type is not None:
            raise ValueError("target_type must be one of (None, 'attr', 'class', 'caption')")

    def _prepare_attr(self):
        URL = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pblRyaVFSWGxPY0U"
        attr_path = os.path.join(self.root, "list_attr_celeba.txt")
        if not os.path.exists(attr_path):
            download(URL, attr_path)

        with open(attr_path) as f:
            f.readline()
            self.attr_names = f.readline().split()
            self.attr_name_to_idx = dict(zip(self.attr_names, range(len(self.attr_names))))
            celeba_to_attr = {k: v.split() for k, v in (line.split(maxsplit=1) for line in f.readlines())}

        self.idx_to_attr = torch.empty((len(self), CelebAHQ.NUM_BINARY_ATTRS), dtype=torch.int)
        for index, celebahq_idx in enumerate(self.split_celebahq):
            attrs_list = celeba_to_attr[self.celebahq_to_celeba[celebahq_idx]]
            for j in range(len(attrs_list)):
                self.idx_to_attr[index, j] = 1 if attrs_list[j] == "1" else 0

    def _prepare_class(self):
        if not self.class_attrs:
            raise ValueError("class_attrs must be set when target_type='class'")
        self._prepare_attr()

    def _prepare_caption(self):
        self.idx_to_caption = []
        for celebahq_idx in self.split_celebahq:
            index = int(celebahq_idx[3:11])
            with open(os.path.join(self.root, "celeba-caption", f"{index}.txt")) as f:
                self.idx_to_caption.append(f.readline())

    def get_class(self, index):
        class_label = 0
        for j, attr_name in enumerate(self.class_attrs):
            attrs = self.idx_to_attr[index]
            class_label += int(attrs[self.attr_name_to_idx[attr_name]]) * 2**j
        return class_label
