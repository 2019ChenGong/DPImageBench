import os
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
from torchvision import datasets


class CIFAR10Base(datasets.CIFAR10):
    def __init__(self, size=None, interpolation="bicubic", flip_p=0.5, **kwargs):
        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        super().__init__(**kwargs)

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

        image = np.array(image).astype(np.uint8)

        example = {}
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["class_label"] = label

        return example


class CIFAR10Train(CIFAR10Base):
    def __init__(self, **kwargs):
        super().__init__(root=os.path.join(os.getcwd(), "data/cifar10"), train=True, download=True, **kwargs)


class CIFAR10Val(CIFAR10Base):
    def __init__(self, **kwargs):
        super().__init__(root=os.path.join(os.getcwd(), "data/cifar10"), train=False, download=True, **kwargs)
