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
import jax
import webdataset as wds
from torch.utils.data import DataLoader


def create_dataloaders(
        train_dataset_shards,
        train_batch_size,
        train_loader_workers,
        shuffle_seed,

        valid_dataset_shards=None,
        valid_batch_size=None,
        valid_loader_workers=None,
        grad_accum=1,


):
    train_dataloader, valid_dataloader = None, None


    total_batch_size = train_batch_size // jax.process_count() //grad_accum
    dataset = wds.DataPipeline(
        wds.SimpleShardList(train_dataset_shards, seed=shuffle_seed),
        itertools.cycle,
        wds.detshuffle(),
        wds.slice(jax.process_index(), None, jax.process_count()),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.ignore_and_continue), #handler=wds.ignore_and_continue
        wds.detshuffle(),
        wds.decode("pil", handler=wds.ignore_and_continue),
        wds.to_tuple("token.pyd", "cls", "siglip_feature.pyd", "dino_feature.pyd", handler=wds.ignore_and_continue),
        # wds.to_tuple("siglip_feature.pyd", handler=wds.ignore_and_continue),
        # wds.to_tuple("cls", handler=wds.ignore_and_continue),
        # wds.to_tuple("token.pyd","cls", handler=wds.ignore_and_continue),
        # wds.map_tuple(train_transform, torch.tensor),
    )


    train_dataloader= DataLoader(
        dataset,
        batch_size=total_batch_size,
        num_workers=train_loader_workers,
        drop_last=True,
        prefetch_factor=10,
        persistent_workers=True,
    )


    if valid_dataset_shards is not None:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(valid_dataset_shards, seed=shuffle_seed),
            itertools.cycle,
            wds.detshuffle(),
            wds.slice(jax.process_index(), None, jax.process_count()),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.ignore_and_continue),  # handler=wds.ignore_and_continue
            wds.detshuffle(),
            wds.decode("pil", handler=wds.ignore_and_continue),
            # wds.to_tuple("token.pyd", "cls", "siglip_feature.pyd", "dino_feature.pyd", handler=wds.ignore_and_continue),
            wds.to_tuple( "siglip_feature.pyd", handler=wds.ignore_and_continue),
            # wds.map_tuple(train_transform, torch.tensor),
        )

        valid_dataloader = DataLoader(
            dataset,
            batch_size=(batch_size := valid_batch_size // jax.process_count()),
            num_workers=valid_loader_workers,
            drop_last=True,
            prefetch_factor=10,
            persistent_workers=True,
        )



    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    from utils.utils import read_yaml, preprocess_config
    import matplotlib.pyplot as plt

    yaml = read_yaml('../configs/test.yaml')
    yaml = preprocess_config(yaml)
    print(yaml)
    train_dataloader, valid_dataloader = create_dataloaders(**yaml['dataset'])
    i=0
    prev=None
    now=None
    for data in train_dataloader:
        token,label,siglip_feature,dino_feature=data
        # siglip_feature = data
        # print(siglip_feature)
        jax.tree_util.tree_map(lambda x:print(x.shape,x.max(),x.min()),data)
        print()
        break






        # break
