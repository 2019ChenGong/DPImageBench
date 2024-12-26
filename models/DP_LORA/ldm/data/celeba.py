import os
from torchvision.datasets import CelebA
from torchvision.transforms import Resize, CenterCrop, ToTensor, Compose


class CelebABase(CelebA):
    def __init__(self, datadir, config, **kwargs):
        # cachedir = os.environ.get("XDG_CACHE_HOME")
        cachedir = "/root/autodl-tmp/"
        super().__init__(
            root=os.path.join(cachedir, datadir),
            target_type=[],
            transform=Compose([Resize(config.size), CenterCrop(config.size), ToTensor()]),
            download=True,
            **kwargs
        )

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        # Rescale from [0, 1] to [-1, 1]
        image = (image * 2) - 1
        # Reshape from C x W x H to H x W x C
        image = image.permute(1, 2, 0).contiguous()
        return {"image": image}


class CelebATrain(CelebABase):
    def __init__(self, **kwargs):
        super().__init__(datadir="CelebA", split="train", **kwargs)


class CelebAValidation(CelebABase):
    def __init__(self, **kwargs):
        super().__init__(datadir="CelebA", split="valid", **kwargs)
