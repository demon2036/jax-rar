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
import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax
from chex import ArrayTree, PRNGKey
from flax.training import train_state
from flax.training.common_utils import shard_prng_key



class TrainState(train_state.TrainState):
    dropout_rng: PRNGKey

    micro_step: int = 0
    micro_in_mini: int = 1
    grad_accum: ArrayTree | None = None

    ema_params: Any = None
    ema_decay: float = 0.9998
    ref_model_params:Any=None

    def split_rngs(self) -> tuple[ArrayTree, ArrayTree]:
        dropout_rng, new_dropout_rng = jax.random.split(self.dropout_rng)
        rngs = { "dropout": dropout_rng,}
        updates = { "dropout_rng": new_dropout_rng, }
        return rngs, updates
