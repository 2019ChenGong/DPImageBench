import os
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
from torchvision import datasets


class SVHNBase(datasets.SVHN):
    def __init__(self, datadir="SVHN", size=None, interpolation="bicubic", flip_p=0.5, **kwargs):
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        super().__init__(root=os.path.join(cachedir, datadir), **kwargs)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)

        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)

        example = {}
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["class_label"] = label

        return example


class SVHNTrain(SVHNBase):
    def __init__(self, **kwargs):
        super().__init__(datadir="SVHN_Train", split="train", download=True, **kwargs)


class SVHNVal(SVHNBase):
    def __init__(self, **kwargs):
        super().__init__(datadir="SVHN_Val", split="test", download=True, **kwargs)
