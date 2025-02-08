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

import argparse
import importlib
import json
import os
import re
import threading
from collections import defaultdict
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import webdataset as wds
import yaml
from chex import Array, ArrayTree
from jax.experimental import mesh_utils
from jax.tree_util import DictKey
from jax.sharding import Mesh,PartitionSpec as PS


class AverageMeter:
    def __init__(self, use_latest: list[str] = []):
        self.buffer = defaultdict(list)
        self.use_latest = use_latest

    def update(self, **kwargs: float):
        for k, v in kwargs.items():
            self.buffer[k].append(v)

    def summary(self, prefix: str = "") -> dict[str, float]:
        buffer = {k: np.array(v) for k, v in self.buffer.items()}
        self.buffer.clear()

        return {
            f"{prefix}{k}": v[-1] if k in self.use_latest else np.mean(v)
            for k, v in buffer.items()
        }


def save_checkpoint_in_background(
        filename, params_bytes: bytes, postfix: str = "ema"
):
    filename = f"{filename}-{postfix}.msgpack"
    def thread_fn():
        with wds.gopen(filename, "wb") as fp:
            fp.write(params_bytes)

    threading.Thread(target=thread_fn).start()



def save_checkpoint_in_background2(
        output_dir,name, params_bytes: bytes, postfix: str = "last"
):
    def thread_fn():
        filename = os.path.join(output_dir, f"{name}-{postfix}.msgpack")
        with wds.gopen(filename, "wb") as fp:
            fp.write(params_bytes)

    threading.Thread(target=thread_fn).start()


class Mixup(nn.Module):
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0

    def apply_mixup(self, images: Array, labels: Array) -> tuple[Array, Array]:
        ratio = jax.random.beta(self.make_rng("mixup"), *(self.mixup_alpha,) * 2)
        randperm = jax.random.permutation(self.make_rng("mixup"), images.shape[0])
        images = ratio * images + (1 - ratio) * images[randperm]
        labels = ratio * labels + (1 - ratio) * labels[randperm]
        return images, labels

    def apply_cutmix(self, images: Array, labels: Array) -> tuple[Array, Array]:
        ratio = jax.random.beta(self.make_rng("mixup"), *(self.cutmix_alpha,) * 2)
        image_mask = self.random_bounding_box(ratio, images.shape[2], images.shape[1])
        label_mask = image_mask.mean((1, 2))

        randperm = jax.random.permutation(self.make_rng("mixup"), images.shape[0])
        images = image_mask * images + (1 - image_mask) * images[randperm]
        labels = label_mask * labels + (1 - label_mask) * labels[randperm]
        return images, labels

    def random_bounding_box(self, ratio: Array, width: int, height: int) -> Array:
        size = (1 - ratio) ** 0.5
        xstart, ystart = jax.random.uniform(self.make_rng("mixup"), (2,))
        xrange, yrange = jnp.linspace(0, 1, width), jnp.linspace(0, 1, height)

        xmask = (xstart - 0.5 * size <= xrange) & (xrange < xstart + 0.5 * size)
        ymask = (ystart - 0.5 * size <= yrange) & (yrange < ystart + 0.5 * size)
        return ~(xmask[None, None, :, None] & ymask[None, :, None, None])

    def __call__(self, images: Array, labels: Array) -> tuple[Array, Array]:
        if self.mixup_alpha == 0 and self.cutmix_alpha == 0:
            return images, labels
        if self.mixup_alpha > 0 and self.cutmix_alpha == 0:
            return self.apply_mixup(images, labels)
        if self.mixup_alpha == 0 and self.cutmix_alpha > 0:
            return self.apply_cutmix(images, labels)

        # If both mixup and cutmix are enabled, only one operation will be selected and
        # applied. Since jax does not support conditional branching on JIT, mixup and
        # cutmix are performed first and only one output will be selected.
        images1, labels1 = self.apply_mixup(images, labels)
        images2, labels2 = self.apply_cutmix(images, labels)

        cond = jax.random.uniform(self.make_rng("mixup")) > 0.5
        return jnp.where(cond, images1, images2), jnp.where(cond, labels1, labels2)


def fixed_sincos2d_embeddings(ncols: int, nrows: int, dim: int) -> Array:
    freqs = 1 / (10000 ** jnp.linspace(0, 1, dim // 4))
    x = jnp.outer(jnp.arange(0, nrows, dtype=jnp.float32), freqs)
    y = jnp.outer(jnp.arange(0, ncols, dtype=jnp.float32), freqs)

    x = jnp.broadcast_to(x[None, :, :], (ncols, nrows, dim // 4))
    y = jnp.broadcast_to(y[:, None, :], (ncols, nrows, dim // 4))
    return jnp.concatenate((jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)), axis=2)


def modified_lamb(
        learning_rate: optax.ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-6,
        eps_root: float = 0.0,
        weight_decay: float = 0.0,
        mask: optax.MaskOrFn = None,
) -> optax.GradientTransformation:
    return optax.chain(
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        optax.add_decayed_weights(weight_decay=weight_decay, mask=mask),
        # Change to use trust ratio on weight decay parameters only.
        optax.masked(optax.scale_by_trust_ratio(), mask=mask),
        optax.scale_by_learning_rate(learning_rate),
    )


def get_layer_index_fn(path: tuple[DictKey, ...], _: Any, num_layers: int = 12) -> int:
    if path[0].key == "models" and path[1].key.startswith("layer_"):
        return int(re.match(r"layer_(\d+)", path[1].key).group(1)) + 1
    if path[0].key == "models" and path[1].key == "embed":
        return 0
    return num_layers


def load_pretrained_params(args: argparse.Namespace, params: ArrayTree) -> ArrayTree:
    with wds.gopen(args.pretrained_ckpt) as fp:
        new_params = flax.serialization.msgpack_restore(fp.read())

    # The positional embeddings will be resized when there is a difference in image
    # resolutions between pretraining and finetuning stage.
    if (
            args.posemb == "learnable"
            and new_params["models"]["embed"]["wpe"].shape
            != params["models"]["embed"]["wpe"].shape
    ):
        new_params["models"]["embed"]["wpe"] = jax.image.resize(
            new_params["models"]["embed"]["wpe"],
            params["models"]["embed"]["wpe"].shape,
            method="bicubic",
        )

    # Reinitialize the classifier head if the models was pretrained on different dataset
    # and `args.label_mapping` is not specified.
    if (
            "head" not in new_params["models"]
            or args.label_mapping is None
            and new_params["models"]["head"]["kernel"].shape
            != params["models"]["head"]["kernel"].shape
    ):
        new_params["models"]["head"] = params["models"]["head"]

    # If `args.label_mapping` is specified, then the same labels will automatically
    # replaced with the pretrained ones.
    if args.label_mapping:
        with wds.gopen(args.label_mapping) as fp:
            label_mapping = json.load(fp)
            src, dst = label_mapping["src"], label_mapping["dst"]

        kernel = np.zeros_like(params["models"]["head"]["kernel"])
        kernel[:, dst] = new_params["models"]["head"]["kernel"][:, src]

        bias = np.full_like(params["models"]["head"]["bias"], fill_value=-10.0)
        bias[dst] = new_params["models"]["head"]["bias"][src]

        new_params["models"]["head"] = {"kernel": kernel, "bias": bias}
    return new_params


def read_yaml(config_path):
    with open(config_path, 'r') as f:
        res = yaml.safe_load(f, )
        return res


def get_obj_from_str(string: str):
    module, cls = string.rsplit('.', 1)
    return getattr(importlib.import_module(module), cls)




def tree_path_to_string(path, sep=None):
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    if sep is None:
        return tuple(keys)
    return sep.join(keys)

def named_tree_map(f, tree, *rest, is_leaf=None, sep=None):
    """ An extended version of jax.tree_util.tree_map, where the mapped function
        f takes both the name (path) and the tree leaf as input.
    """
    return jax.tree_util.tree_map_with_path(
        lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
        tree, *rest,
        is_leaf=is_leaf
    )

def match_partition_rules(rules, params):
    """ Returns a pytree of PartitionSpec according to rules. Supports handling
        Flax TrainState and Optax optimizer state.
    """

    def get_partition_spec(name, leaf):
        # print(name,)
        if len(leaf.shape) == 0 or np.prod(leaf.shape) == 1:
            """ Don't partition scalar values. """
            return PS()
        for rule, ps in rules:
            if re.search(rule, name) is not None:
                return ps
        raise ValueError(f'Partition rule not found for param: {name}')

    return named_tree_map(get_partition_spec, params, sep='/')





def replace_env_variables(text):

    if isinstance(text,str):
        # 匹配 $VAR_NAME 或 ${VAR_NAME} 格式的环境变量
        pattern = re.compile(r'\$(\w+|\{(\w+)\})')
        # 查找并替换所有环境变量
        def replace_match(match):
            var_name = match.group(1) if match.group(1) else match.group(2)
            # 获取环境变量值，如果不存在则返回空字符串
            return os.environ.get(var_name, '')

        # 使用正则表达式替换所有匹配项

        text=pattern.sub(replace_match, text)

        try:
            text=eval(text)
        except Exception as e:
            pass


        return text
    else:
        return text


def preprocess_config(yaml):
    yaml=jax.tree_util.tree_map(replace_env_variables,yaml)
    return yaml

def get_partition_rules_caformer():
    return (
        # ('scale', PS('mp')),
        # ('bias', PS('mp')),

        ('fc/fc1/kernel', PS('fsdp', 'mp')),
        ('fc/fc2/kernel', PS('mp', None)),
        #
        # ('downsample/conv/kernel', PS(None, None, 'fsdp', 'mp')),
        ('downsample/conv/kernel', PS(None, None, 'mp', 'fsdp')),
        # ('stem/conv/kernel', PS(None, None, 'fsdp', 'mp')),
        #
        # ('MetaFormerStage_[01]/.*/pwconv1/kernel', PS(None, None, 'fsdp', 'mp')),
        ('MetaFormerStage_[01]/.*/pwconv1/kernel', PS(None, None, 'mp', 'fsdp')),
        # ('MetaFormerStage_[01]/.*/dwconv/kernel', PS(None, None, 'fsdp', 'mp')),
        ('MetaFormerStage_[01]/.*/pwconv2/kernel', PS(None, None, 'mp', 'fsdp')),
        # ('MetaFormerStage_[01]/.*/pwconv2/kernel', PS(None, None, 'fsdp', 'mp')),

        ('MetaFormerStage_[01]/.*/fc1/kernel', PS(None,None,'fsdp', 'mp')),
        ('MetaFormerStage_[01]/.*/fc2/kernel', PS(None,None,'mp', 'fsdp')),

        ('MetaFormerStage_[23]/.*/qkv/kernel', PS('fsdp', 'mp')),
        ('MetaFormerStage_[23]/.*/proj/kernel', PS('mp', 'fsdp')),

        ('MetaFormerStage_[23]/.*/fc1/kernel',PS('fsdp','mp')),
        ('MetaFormerStage_[23]/.*/fc2/kernel', PS('mp', 'fsdp')),

        ('.*', PS(None)),
    )






def get_partition_rules_vit():
    return (
        # ('scale', PS('mp')),
        # ('bias', PS('mp')),
        #
        # ('fc/fc1/kernel', PS('fsdp', 'mp')),
        # ('fc/fc2/kernel', PS('mp', 'fsdp')),
        # ('downsample/conv/kernel', PS(None, None, 'fsdp', 'mp')),
        # ('stem/conv/kernel', PS(None, None, 'fsdp', 'mp')),
        #
        # ('MetaFormerStage_[01]/.*/pwconv1/kernel', PS(None, None, 'fsdp', 'mp')),
        # ('MetaFormerStage_[01]/.*/dwconv/kernel', PS(None, None, 'fsdp', 'mp')),
        # ('MetaFormerStage_[01]/.*/pwconv2/kernel', PS(None, None, 'fsdp', 'mp')),
        #
        # ('MetaFormerStage_[01]/.*/fc1/kernel', PS(None,None,'fsdp', 'mp')),
        # ('MetaFormerStage_[01]/.*/fc2/kernel', PS(None,None,'mp', 'fsdp')),
        #
        # ('MetaFormerStage_[23]/.*/qkv/kernel', PS('fsdp', 'mp')),
        # ('MetaFormerStage_[23]/.*/proj/kernel', PS('mp', 'fsdp')),
        #

        # (DictKey(key='models.layer_1.attn.wk.bias'),)(12, 64)
        # (DictKey(key='models.layer_1.attn.wk.kernel'), )(768, 12, 64)
        # (DictKey(key='models.layer_1.attn.wo.bias'), )(768, )
        # (DictKey(key='models.layer_1.attn.wo.kernel'), )(12, 64, 768)
        # (DictKey(key='models.layer_1.attn.wq.bias'), )(12, 64)
        # (DictKey(key='models.layer_1.attn.wq.kernel'), )(768, 12, 64)
        # (DictKey(key='models.layer_1.attn.wv.bias'), )(12, 64)
        # (DictKey(key='models.layer_1.attn.wv.kernel'), )(768, 12, 64)
        # (DictKey(key='models.layer_1.ff.w1.bias'), )(3072, )
        # (DictKey(key='models.layer_1.ff.w1.kernel'), )(768, 3072)
        # (DictKey(key='models.layer_1.ff.w2.bias'), )(768, )
        # (DictKey(key='models.layer_1.ff.w2.kernel'), )(3072, 768)
        #

        # ('attn/wq/kernel', PS('mp', 'fsdp')),
        # ('attn/wk/kernel', PS('mp', 'fsdp')),
        # ('attn/wv/kernel', PS('mp', 'fsdp')),
        # ('attn/wo/kernel', PS(None, 'fsdp','mp',)),

        ('attn/wq/kernel', PS('mp', 'fsdp')),
        ('attn/wk/kernel', PS('mp', 'fsdp')),
        ('attn/wv/kernel', PS('mp', 'fsdp')),
        ('attn/wo/kernel', PS( 'fsdp', 'mp', )),

        ('ff/w1/kernel', PS('fsdp', 'mp')),
        ('ff/w2/kernel', PS('mp', 'fsdp')),

        # ('ff/w2/kernel', PS('fsdp','mp', )),

        #
        #
        # ('models/head/kernel', PS('mp', 'fsdp')),
        ('.*', PS(None)),
    )



def get_jax_mesh(axis_dims, names):
    if axis_dims.startswith('!'):
        # Allow splitting a physical mesh axis if needed
        mesh_axis_splitting = True
        axis_dims = axis_dims[1:]
    else:
        mesh_axis_splitting = False

    if ':' in axis_dims:
        dims = []
        dim_names = []
        for axis in axis_dims.split(','):
            print(axis)
            name, dim = axis.split(':')
            assert name in names
            dims.append(int(dim))
            dim_names.append(name)
        assert(set(dim_names) == set(names))
    else:
        dims = [int(x) for x in axis_dims.split(',')]
        dim_names = names
    assert len(dims) == len(names)
    mesh_shape = np.arange(jax.device_count()).reshape(dims).shape
    if mesh_axis_splitting:
        physical_mesh = np.array(jax.devices()).reshape(mesh_shape)
    else:
        physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
    return Mesh(physical_mesh, dim_names)

# mesh_dim='dp:2,fsdp:-1,mp:1'
def get_jax_mesh2(axis_dims):
    return get_jax_mesh(axis_dims, ('dp', 'fsdp', 'mp'))
