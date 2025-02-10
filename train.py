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
import os
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import tqdm
import wandb
from flax.linen import partitioning as nn_partitioning
from jax import NamedSharding
from jax._src.mesh import Mesh
from jax._src.partition_spec import PartitionSpec
from jax.sharding import PartitionSpec as P
from torch.utils.data import DataLoader

from dataset.dataset import create_dataloaders
from jax_fid import inception

from sampler import Sampler
from state.state_pjit import init_state, get_jax_tokenizer
from training import training_step
from utils.utils import AverageMeter, read_yaml, preprocess_config, get_jax_mesh2


# os.environ['GOPEN_VERBOSE'] = '1'
# jax.distributed.initialize()


def _build_global_shape_and_sharding(
        local_shape: tuple[int, ...], global_mesh: Mesh
) -> tuple[tuple[int, ...], NamedSharding]:
    sharding = NamedSharding(global_mesh, PartitionSpec(global_mesh.axis_names))

    global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]

    return global_shape, sharding


def _form_global_array(path, array: np.ndarray, global_mesh: Mesh) -> jax.Array:
    """Put local sharded array into local devices"""

    global_shape, sharding = _build_global_shape_and_sharding(np.shape(array), global_mesh)
    try:
        local_device_arrays = np.split(array, len(global_mesh.local_devices), axis=0)
    except ValueError as array_split_error:
        raise ValueError(
            f"Unable to put to devices shape {array.shape} with "
            f"local device count {len(global_mesh.local_devices)} "
            f"at {jtu.keystr(path)}"
        ) from array_split_error

    local_device_buffers = jax.device_put(local_device_arrays, global_mesh.local_devices)
    return jax.make_array_from_single_device_arrays(global_shape, sharding, local_device_buffers)


def evaluate(state: TrainState, dataloader: DataLoader, validation_adv_step_jited, mesh) -> dict[str, float]:
    average_meter = AverageMeter()
    for batch in tqdm.tqdm(dataloader, leave=False, dynamic_ncols=True):
        batch = jax.tree_util.tree_map(lambda x: jnp.array(np.asarray(x)), batch)
        batch = jtu.tree_map_with_path(partial(_form_global_array, global_mesh=mesh), batch)
        metrics = validation_adv_step_jited(state, batch)
        average_meter.update(**metrics)

    metrics = average_meter.summary("val/")
    num_samples = metrics.pop("val/num_samples")
    return jax.tree_util.tree_map(lambda x: x / num_samples, metrics)


# def get_cos_sim(image_features, labels):
#     sim = torch.nn.CosineSimilarity(dim=-1)(image_features.unsqueeze(1), image_features.unsqueeze(0))
#     sim = (sim + 1) / 2
#     label_mask = labels[:, None] != labels[None, :]
#     mask_sin = sim * label_mask
#     return mask_sin
# data=next(train_dataloader_iter)
# print(get_cos_sim(data[-1],data[1]))

def main(configs):
    training_steps = configs['steps'] * configs['training_epoch'] // configs['dataset']['train_batch_size']
    warmup_steps = configs['steps'] * configs['warmup_epoch'] // configs['dataset']['train_batch_size']
    eval_interval = configs['steps'] * configs['eval_epoch'] // configs['dataset']['train_batch_size']
    epoch_per_step = configs['steps'] // configs['dataset']['train_batch_size']
    log_interval = configs['log_interval']
    grad_accum_steps = configs['train_state'].get('grad_accum_steps', 1)
    resume = configs.get('resume', False)

    # os.environ['JAX_PLATFORMS']='cpu'
    # os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
    # jax.config.update('jax_platform_name', 'cpu')
    # pass
    """"""
    dp = configs.pop('dp', -1)
    fsdp = configs.pop('fsdp', 1)
    tp = configs.pop('tp', 1)

    mesh_dim = f'{dp},{fsdp},{tp}'  #  '-1,1,1'
    postfix = "ema"
    name = configs['name']
    output_dir = configs['output_dir']
    filename = os.path.join(output_dir, f"{name}-{postfix}")
    print(filename)

    mesh = get_jax_mesh2(mesh_dim)
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("dp", 'fsdp', 'mp'))
    data_spec = [["dp", 'fsdp', 'mp']]
    # data_spec=["dp",'fsdp','mp']
    data_spec = P(*data_spec)
    print(data_spec)
    sharding = jtu.tree_map(lambda p: NamedSharding(mesh, p), data_spec)

    logical_axis_rules = [
        ['batch', ['dp', 'fsdp']],
        ['activation_embed', 'mp'],
        ['mlp', 'mp'],
        ['vocab', 'fsdp'],
        ['embed', 'fsdp'],
        ['heads', 'mp'],
    ]

    with mesh, nn_partitioning.axis_rules(logical_axis_rules):
        pass

        state, init_step, train_state_sharding, rar_config, model = init_state(configs['train_state'],
                                                                               warmup_steps=warmup_steps,
                                                                               training_steps=training_steps,
                                                                               mesh=mesh,
                                                                               # restore_state_config=configs[
                                                                               #     'restore_state'] if 'restore_state' in configs else None,
                                                                               # remote_model_path=filename, resume=resume
                                                                               )

        fid_model = inception.InceptionV3(pretrained=True)
        fid_model_params = fid_model.init(jax.random.PRNGKey(1), jnp.ones((1, 256, 256, 3)))
        tokenizer, tokenizer_params = get_jax_tokenizer()
        sampler = Sampler(model, tokenizer, tokenizer_params, rar_config, 128, fid_model, fid_model_params)

        training_step_pjit = jax.jit(training_step,
                                     donate_argnums=(0,),
                                     out_shardings=(train_state_sharding, None),
                                     in_shardings=(train_state_sharding, sharding,),
                                     )

        fid=sampler.sample_and_eval(state.ema_params)

        train_dataloader, valid_dataloader = create_dataloaders(**configs['dataset'], grad_accum=grad_accum_steps)
        train_dataloader_iter = iter(train_dataloader)



        average_meter, max_val_acc1 = AverageMeter(use_latest=["learning_rate"]), 0.0

        if jax.process_index() == 0:
            wandb.init(name=configs['name'], project=configs['project'], config=configs)


        for step in tqdm.tqdm(range(init_step, training_steps + 1), initial=init_step, total=training_steps + 1):
            for _ in range(grad_accum_steps):
                batch = jax.tree_util.tree_map(lambda x: jnp.array(np.asarray(x)), next(train_dataloader_iter))
                batch = jtu.tree_map_with_path(partial(_form_global_array, global_mesh=mesh), batch)
                state, metrics = training_step_pjit(state, batch, )
                average_meter.update(**metrics)
                print(metrics)



            if (
                    jax.process_index() == 0
                    and log_interval > 0
                    and step % log_interval == 0
            ):
                metrics = average_meter.summary(prefix="train/")
                # metrics["processed_samples"] = step * configs['dataset']['train_batch_size']
                wandb.log(metrics, step)


            
            if eval_interval > 0 and (
                    step % eval_interval == 0 or step == training_steps
            ):

                fid=sampler.sample_and_eval(state.ema_params)

                """
                if valid_dataloader is None:
                    continue
                del batch
                state = off_load_memory_state(state)
                metrics = evaluate(state, valid_dataloader, validation_adv_step_jited, mesh)
                state = reload_device_state(state)

                if "val/advacc1" in metrics:
                    now_acc1 = metrics["val/advacc1"]
                else:
                    now_acc1 = metrics["val/acc1"]
                print(now_acc1,max_val_acc1)
                if now_acc1 > max_val_acc1:
                    ckpt = {'models': state}
                    save_args = orbax_utils.save_args_from_target(ckpt)
                    checkpointer.save(filename, ckpt, save_args=save_args, force=True)
                    max_val_acc1 = now_acc1
                    del ckpt

                metrics["val/acc1/best"] = max_val_acc1
                metrics["processed_samples"] = step * configs['dataset']['train_batch_size']
                if jax.process_index() == 0:
                    wandb.log(metrics, step)

            if use_orbax_save:
                checkpointer.wait_until_finished()

    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-path", type=str,
                        default='configs/test.yaml')
    args = parser.parse_args()
    yaml = read_yaml(args.yaml_path)

    yaml = preprocess_config(yaml)
    jax.distributed.initialize()
    print(yaml)
    main(yaml)
