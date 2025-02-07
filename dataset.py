# Copyright 2024 Jungwoo Park (affjljoo3581)
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

from __future__ import annotations

import itertools
from functools import partial
from typing import Any

import jax
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import webdataset as wds
from torch.utils.data import DataLoader, default_collate



def create_transforms(random_crop,
                      image_size,
                      auto_augment,
                      color_jitter,
                      random_erasing,
                      test_crop_ratio
                      ) -> tuple[nn.Module, nn.Module]:
    if random_crop == "rrc":
        train_transforms = [T.RandomResizedCrop(image_size, interpolation=3)]
    elif random_crop == "src":
        train_transforms = [
            T.Resize(image_size, interpolation=3),
            T.RandomCrop(image_size, padding=4, padding_mode="reflect"),
        ]
    elif random_crop == "none":
        train_transforms = [
            T.Resize(image_size, interpolation=3),
            T.CenterCrop(image_size),
        ]

    train_transforms += [
        T.RandomHorizontalFlip(),
        auto_augment_factory(image_size, auto_augment),
        T.ColorJitter(color_jitter, color_jitter, color_jitter),
        T.RandomErasing(random_erasing, value="random"),
        T.PILToTensor(),
    ]
    valid_transforms = [
        T.Resize(int(image_size / test_crop_ratio), interpolation=3),
        T.CenterCrop(image_size),
        T.PILToTensor(),
    ]
    return T.Compose(train_transforms), T.Compose(valid_transforms)




def collate_and_pad(batch: list[Any], batch_size: int = 1) -> Any:
    pad = tuple(torch.full_like(x, fill_value=-1) for x in batch[0])
    return default_collate(batch + [pad] * (batch_size - len(batch)))





def create_dataloaders(
        train_dataset_shards,
        valid_dataset_shards,
        train_batch_size,
        valid_batch_size,
        train_loader_workers,
        valid_loader_workers,
        shuffle_seed,
        random_crop,
        image_size,
        grad_accum=1,


):


    train_dataloader, train_origin_dataloader, valid_dataloader = None, None, None
    train_transform, valid_transform = create_transforms(random_crop,
                                                         image_size,
                                                         auto_augment,
                                                         color_jitter,
                                                         random_erasing,
                                                         test_crop_ratio
                                                         )

    train_generated_transform, valid_transform = create_transforms(random_crop,
                                                               image_size,
                                                               auto_generated_augment,
                                                               color_generated_jitter,
                                                               random_generated_erasing,
                                                               test_crop_ratio
                                                               )
    total_batch_size = train_batch_size // jax.process_count() //grad_accum



    dataset = wds.DataPipeline(
        wds.SimpleShardList(train_dataset_shards, seed=shuffle_seed),
        itertools.cycle,
        wds.detshuffle(),
        wds.slice(jax.process_index(), None, jax.process_count()),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        wds.detshuffle(),
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.to_tuple("jpg", "cls", handler=wds.ignore_and_continue),
        wds.map_tuple(train_transform, torch.tensor),
    )

    train_origin_dataloader = DataLoader(
        dataset,
        batch_size=total_batch_size,
        num_workers=train_loader_workers,
        drop_last=True,
        prefetch_factor=10,
        persistent_workers=True,
    )

    if valid_dataset_shards is not None:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(valid_dataset_shards),
            wds.slice(jax.process_index(), None, jax.process_count()),
            wds.split_by_worker,
            # wds.cached_tarfile_to_samples(),
            wds.tarfile_to_samples(handler=wds.ignore_and_continue),
            wds.decode("pil"),
            wds.to_tuple("jpg", "cls"),
            wds.map_tuple(valid_transform, torch.tensor),
        )
        # valid_dataloader = DataLoader(
        #     dataset,
        #     batch_size=(batch_size := valid_batch_size // jax.process_count()),
        #     num_workers=valid_loader_workers,
        #     collate_fn=partial(collate_and_pad, batch_size=batch_size),
        #     drop_last=False,
        #     prefetch_factor=10,
        #     persistent_workers=True,
        # )
        valid_dataloader = DataLoader(
            dataset,
            batch_size=(batch_size := valid_batch_size // jax.process_count()),
            num_workers=valid_loader_workers,
            collate_fn=partial(collate_and_pad, batch_size=batch_size),
            drop_last=False,
            prefetch_factor=40,
            persistent_workers=False,
        )

    return train_origin_dataloader, valid_dataloader


if __name__ == "__main__":
    from utils import read_yaml, preprocess_config

    yaml = read_yaml('configs/test.yaml')
    yaml = preprocess_config(yaml)
    train_dataloader, valid_dataloader = create_dataloaders(**yaml['dataset'])
