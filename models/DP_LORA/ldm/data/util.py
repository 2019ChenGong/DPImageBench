from functools import partial
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from ldm.privacy.myopacus import my_wrap_data_loader


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class VirtualBatchWrapper(pl.LightningDataModule):
    """Produces data loaders guaranteed to return batches below the given size.

    The sampler object attached to each data loader has a `last_batch_queue`
    attribute which stores whether the next returned physical batch is
    the last one in the virtual batch produced by the underlying sampler. This
    can be used to signal to the optimizer not to perform a gradient update.

    For example, suppose the underlying sampler produces a batch of size 100
    but max_batch_size is set to 20. Instead of returning the single batch with
    100 elements, these data loaders will separate this 'virtual' batch into 5
    'physical' batches each of size 20. After returning each of the first 4
    physical batches, the sampler's `last_batch_queue` will hold False as the
    first element. After returning the 5th and final batch, the first element
    will be set to True.
    """
    def __init__(self, data_module, max_batch_size):
        super().__init__()
        self.wrapped_module = data_module
        self.max_batch_size = max_batch_size

    def train_dataloader(self):
        return my_wrap_data_loader(data_loader=self.wrapped_module.train_dataloader(),
                                   max_batch_size=self.max_batch_size)

    def val_dataloader(self):
        return my_wrap_data_loader(data_loader=self.wrapped_module.val_dataloader(),
                                   max_batch_size=self.max_batch_size)

    def test_dataloader(self):
        return my_wrap_data_loader(data_loader=self.wrapped_module.test_dataloader(),
                                   max_batch_size=self.max_batch_size)


def get_sample_rate(data_loader):
    """Tries to infer the sample rate from the given data loader."""
    if hasattr(data_loader, "sample_rate") and data_loader.sample_rate:
        return data_loader.sample_rate
    elif hasattr(data_loader, "batch_size") and data_loader.batch_size:
        return data_loader.batch_size / len(data_loader.dataset)
    else:
        raise RuntimeError("Cannot infer sample rate from data loader")
