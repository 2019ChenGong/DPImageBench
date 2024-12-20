# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from queue import Queue
from typing import Any, List, Optional

import numpy as np
from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)
import torch
from torch.utils.data import BatchSampler, Dataset, DataLoader, Sampler
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import _collate_fn_t

from opacus.lightning import DPDataLoader, DPLightningDataModule

logger = logging.getLogger(__name__)


def build_empty_batch(data):
    """
    Creates an object that represents an empty batch.

    Args:
        data: a single example from the dataset

    Returns:
        An object which can be used as an empty batch
    """
    if isinstance(data, dict):
        return {k: build_empty_batch(data[k]) for k in data}
    elif torch.is_tensor(data):
        return torch.zeros((0, *data.shape), dtype=data.dtype)
    elif isinstance(data, np.ndarray):
        torch_data = torch.as_tensor(data)
        return torch.zeros((0, *torch_data.shape), dtype=torch_data.dtype)
    elif isinstance(data, np.bool_) or isinstance(data, np.number):
        torch_data = torch.as_tensor(data)
        return torch.zeros((0,), dtype=torch_data.dtype)
    elif isinstance(data, str):
        return tuple()
    elif isinstance(data, int):
        return torch.zeros((0,))
    elif isinstance(data, float):
        return torch.zeros((0,), dtype=torch.float64)
    else:
        raise NotImplementedError(f"Unhandled data type {type(data)}")


def wrap_collate_with_empty(
    *,
    collate_fn: Optional[_collate_fn_t],
    empty_batch: Any
):
    """
    Wraps given collate function to handle empty batches.

    Args:
        collate_fn: collate function to wrap
        empty_batch: an object that should be returned for an empty batch

    Returns:
        New collate function, which is equivalent to input ``collate_fn`` for
        non-empty batches and outputs the given object if the input batch
        is of size 0
    """

    def collate(batch):
        if len(batch) > 0:
            return collate_fn(batch)
        else:
            return empty_batch

    return collate


class MyDPDataLoader(DPDataLoader):
    def __init__(
        self,
        dataset: Dataset,
        *,
        sample_rate: float,
        collate_fn: Optional[_collate_fn_t] = None,
        drop_last: bool = False,
        generator=None,
        distributed: bool = False,
        **kwargs,
    ):
        """

        Args:
            dataset: See :class:`torch.utils.data.DataLoader`
            sample_rate: probability with which each element of the dataset is included
                in the next batch.
            num_workers: See :class:`torch.utils.data.DataLoader`
            collate_fn: See :class:`torch.utils.data.DataLoader`
            pin_memory: See :class:`torch.utils.data.DataLoader`
            drop_last: See :class:`torch.utils.data.DataLoader`
            timeout: See :class:`torch.utils.data.DataLoader`
            worker_init_fn: See :class:`torch.utils.data.DataLoader`
            multiprocessing_context: See :class:`torch.utils.data.DataLoader`
            generator: Random number generator used to sample elements
            prefetch_factor: See :class:`torch.utils.data.DataLoader`
            persistent_workers: See :class:`torch.utils.data.DataLoader`
            distributed: set ``True`` if you'll be using DPDataLoader in a DDP environment
                Selects between ``DistributedUniformWithReplacementSampler`` and
                ``UniformWithReplacementSampler`` sampler implementations
        """

        self.sample_rate = sample_rate
        self.distributed = distributed

        if distributed:
            batch_sampler = DistributedUniformWithReplacementSampler(
                total_size=len(dataset),  # type: ignore[assignment, arg-type]
                sample_rate=sample_rate,
                generator=generator,
            )
        else:
            batch_sampler = UniformWithReplacementSampler(
                num_samples=len(dataset),  # type: ignore[assignment, arg-type]
                sample_rate=sample_rate,
                generator=generator,
            )

        if collate_fn is None:
            collate_fn = default_collate

        if drop_last:
            logger.warning(
                "Ignoring drop_last as it is not compatible with DPDataLoader."
            )

        super(DPDataLoader, self).__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=wrap_collate_with_empty(
                collate_fn=collate_fn,
                empty_batch=build_empty_batch(dataset[0])
            ),
            generator=generator,
            **kwargs,
        )


class MyDPLightningDataModule(DPLightningDataModule):
    def train_dataloader(self):
        dataloader = self.datamodule.train_dataloader()
        return MyDPDataLoader.from_data_loader(dataloader, distributed=False)


class MyBatchSplittingSampler(Sampler[List[int]]):
    """
    Samples according to the underlying instance of ``Sampler``, but splits
    the index sequences into smaller chunks.

    Used to split large logical batches into physical batches of a smaller size.
    """

    def __init__(self, *, sampler: Sampler[List[int]], max_batch_size: int):
        """

        Args:
            sampler: Wrapped Sampler instance
            max_batch_size: Max size of emitted chunk of indices
        """
        self.sampler = sampler
        self.max_batch_size = max_batch_size
        self.last_batch_queue = Queue()

    def __iter__(self):
        for batch_idxs in self.sampler:
            if len(batch_idxs) == 0:
                self.last_batch_queue.put(True)
                yield []
                continue

            split_idxs = np.array_split(
                batch_idxs, math.ceil(len(batch_idxs) / self.max_batch_size)
            )
            split_idxs = [s.tolist() for s in split_idxs]
            for i, x in enumerate(split_idxs[:-1]):
                self.last_batch_queue.put(False)
                yield x
            self.last_batch_queue.put(True)
            yield split_idxs[-1]

    def __len__(self):
        if isinstance(self.sampler, BatchSampler):
            return int(
                len(self.sampler) * (self.sampler.batch_size / self.max_batch_size)
            )
        elif isinstance(self.sampler, UniformWithReplacementSampler) or isinstance(
            self.sampler, DistributedUniformWithReplacementSampler
        ):
            expected_batch_size = self.sampler.sample_rate * self.sampler.num_samples
            return int(len(self.sampler) * (expected_batch_size / self.max_batch_size))

        return len(self.sampler)


def my_wrap_data_loader(*, data_loader: DataLoader, max_batch_size: int):
    """
    Replaces batch_sampler in the input data loader with ``MyBatchSplittingSampler``.

    NOTE: It is the responsibility of the user of this function to call
          DPOptimizer.signal_skip_step(True) on intermediary batches, and
          DPOptimizer.signal_skip_step(False) on the final physical batch.


    Args:
        data_loader: DataLoader instance to be wrapped
        max_batch_size: max physical batch size we want to emit

    Returns:
        New DataLoader instance with batch_sampler wrapped in ``MyBatchSplittingSampler``
    """

    new_data_loader = DataLoader(
        dataset=data_loader.dataset,
        batch_sampler=MyBatchSplittingSampler(
            sampler=data_loader.batch_sampler,
            max_batch_size=max_batch_size,
        ),
        num_workers=data_loader.num_workers,
        collate_fn=data_loader.collate_fn,
        pin_memory=data_loader.pin_memory,
        timeout=data_loader.timeout,
        worker_init_fn=data_loader.worker_init_fn,
        multiprocessing_context=data_loader.multiprocessing_context,
        generator=data_loader.generator,
        prefetch_factor=data_loader.prefetch_factor,
        persistent_workers=data_loader.persistent_workers,
    )
    if hasattr(data_loader, "sample_rate"):
        new_data_loader.sample_rate = data_loader.sample_rate
    return new_data_loader
