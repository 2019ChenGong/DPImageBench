import os
import numpy as np
from PIL import Image
from torchvision import datasets


class MNISTBase(datasets.MNIST):
    def __init__(self, datadir, size=None, **kwargs):
        self.size = size
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        super().__init__(root=os.path.join(cachedir, datadir), **kwargs)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)

        img = np.array(image).astype(np.uint8)
        img_three_channels = np.stack((img, img, img), axis=2)

        img = img_three_channels

        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size))

        image = np.array(image).astype(np.uint8)

        example = {}
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["class_label"] = label

        return example


class MNISTTrain(MNISTBase):
    def __init__(self, **kwargs):
        super().__init__(datadir="MNIST_Train", train=True, download=True, **kwargs)


class MNISTVal(MNISTBase):
    def __init__(self, **kwargs):
        super().__init__(datadir="MNIST_Val", train=False, download=True, **kwargs)
