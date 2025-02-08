import copy
import json
import time
from functools import partial

import flax
import jax
import numpy as np
import optax
import timm
from jax._src.pjit import pjit
from optax.schedules import _join
from orbax.checkpoint._src.handlers.standard_checkpoint_handler import StandardRestoreArgs
from orbax.checkpoint._src.serialization.type_handlers import ArrayRestoreArgs

import train_modules
from pre_define import CRITERION_COLLECTION, OPTIMIZER_COLLECTION
from training import TrainState
from utils import read_yaml, get_obj_from_str, Mixup, preprocess_config, match_partition_rules, \
    get_partition_rules_vit, get_partition_rules_caformer
import os
import jax.numpy as jnp
from convert_model_pytorch import convert_torch_to_flax_conv_next, convert_torch_to_flax_meta_former
import orbax.checkpoint as ocp
from timm.models import MetaFormer, ConvNeXt




def load_pretrained_params(pretrained_ckpt, abstract_state):
    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    ckpt = {'models': abstract_state}
    restore_kwargs = {
        "restore_args": jax.tree_map(
            lambda _: ocp.RestoreArgs(restore_type=np.ndarray), ckpt
        )
    }
    state = checkpointer.restore(pretrained_ckpt, item=ckpt, **restore_kwargs)['models']
    params = state.ema_params
    return params


def load_pretrain(pretrained_model='convnext_base.fb_in1k', default_params=None):
    model_torch = timm.create_model(pretrained_model, pretrained=True)
    params = {k: v.numpy() for k, v in model_torch.state_dict().items()}
    params = flax.traverse_util.unflatten_dict(params, sep=".")

    if isinstance(model_torch, ConvNeXt):
        model_jax_params = convert_torch_to_flax_conv_next(params, sep='', default_params=default_params)
    elif isinstance(model_torch, MetaFormer):
        model_jax_params = convert_torch_to_flax_meta_former(params, sep='', )
    else:
        raise NotImplemented()

    model_jax_params = jax.tree_util.tree_map(jnp.asarray, model_jax_params)
    return {'models': model_jax_params}



def warmup_stable_cosine_decay_schedule(
        init_value: float,
        peak_value: float,
        warmup_steps: int,
        decay_steps: int,
        end_value: float = 0.0,
        exponent: float = 1.0,
) :
    r"""Linear warmup followed by cosine decay.

    Args:
      init_value: Initial value for the scalar to be annealed.
      peak_value: Peak value for scalar to be annealed at end of warmup.
      warmup_steps: Positive integer, the length of the linear warmup.
      decay_steps: Positive integer, the total length of the schedule. Note that
        this includes the warmup time, so the number of steps during which cosine
        annealing is applied is ``decay_steps - warmup_steps``.
      end_value: End value of the scalar to be annealed.
      exponent: The default decay is ``0.5 * (1 + cos(pi t/T))``, where ``t`` is
        the current timestep and ``T`` is ``decay_steps``. The exponent modifies
        this to be ``(0.5 * (1 + cos(pi * t/T))) ** exponent``. Defaults to 1.0.

    Returns:
      schedule
        A function that maps step counts to values
    """
    alpha = 0.0 if peak_value == 0.0 else end_value / peak_value

    stable_steps=int(decay_steps*2/3)

    schedules = [
        optax.schedules.linear_schedule(
            init_value=init_value,
            end_value=peak_value,
            transition_steps=warmup_steps,
        ),
        optax.schedules.constant_schedule(
           value=peak_value
        ),
        optax.schedules.cosine_decay_schedule(
            init_value=peak_value,
            decay_steps=decay_steps - warmup_steps-stable_steps,
            alpha=alpha,
            exponent=exponent,
        ),
    ]
    return _join.join_schedules(schedules, [warmup_steps,warmup_steps+stable_steps])


def create_train_state(train_state_config, image_size: int = 224,
                       warmup_steps=1, training_steps=10, mesh=None, logical_axis_rules=None
                       ):  # -> TrainState:

    model_config = train_state_config['models']
    optimizer_config = train_state_config['optimizer']
    train_module_config = train_state_config['train_module']
    grad_accum_steps=train_state_config.pop('grad_accum_steps', 1)


    model = get_obj_from_str(model_config['target'])(**model_config['model_kwargs'])
    print(f'{train_module_config=}')

    train_module = get_obj_from_str(train_module_config.pop('target'))  #(**model_config['model_kwargs'])

    module = train_module(
        model=model,
        mixup=Mixup(train_module_config.pop('mixup', ), train_module_config.pop('cutmix')),
        label_smoothing=train_module_config.pop('label_smoothing') if train_module_config['criterion'] != "bce" else 0,
        criterion=CRITERION_COLLECTION[train_module_config.pop('criterion')], **train_module_config
    )
    if jax.process_index() == 0:
        print(module)

    # Initialize the models weights with dummy inputs. Using the init RNGS and inputs, we
    # will tabulate the summary of models and its parameters. Furthermore, empty gradient
    # accumulation arrays will be prepared if the gradient accumulation is enabled.
    example_inputs = {
        "tokens": jnp.zeros((1,256), dtype=jnp.int32),
        # "labels": jnp.zeros((1,), dtype=jnp.int32),
        "labels": jnp.ones((1,256), dtype=jnp.int32)  #jnp.array([1,2], dtype=jnp.int32),
    }
    init_rngs = {"params": jax.random.PRNGKey(train_state_config['init_seed'])}


    # if args.grad_accum > 1:
    #     grad_accum = jax.tree_map(jnp.zeros_like, params)
    lr = optimizer_config['optimizer_kwargs'].pop('learning_rate')
    end_lr = optimizer_config['optimizer_kwargs'].pop('end_learning_rate', 1e-5)
    init_value = optimizer_config['optimizer_kwargs'].pop('init_value', 1e-6)
    schedule = optimizer_config.get('schedule', 'cosine')
    clip_grad = optimizer_config.get('clip_grad', 1.0)

    # Create learning rate scheduler and optimizer with gradient clipping. The learning
    # rate will be recorded at `hyperparams` by `optax.inject_hyperparameters`.

    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
    def create_optimizer_fn(
            learning_rate: optax.Schedule,
    ) -> optax.GradientTransformation:

        tx = OPTIMIZER_COLLECTION[optimizer_config['target']](
            learning_rate=learning_rate,
            **optimizer_config['optimizer_kwargs'],
            mask=partial(jax.tree_util.tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
        )
        print(f'{clip_grad=}')
        if clip_grad is not None:
            tx = optax.chain(optax.clip_by_global_norm(clip_grad), tx)
        return tx


    if schedule == 'constant':
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=lr,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=training_steps,
            end_value=lr,
        )

    elif schedule=="wsd":
        learning_rate = warmup_stable_cosine_decay_schedule(
            init_value=init_value,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=training_steps,
            end_value=end_lr,
        )


    else:
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=init_value,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=training_steps,
            end_value=end_lr,
        )

    tx = create_optimizer_fn(learning_rate, )
    def init_by_params_fn(params, ) -> TrainState:
        if grad_accum_steps > 1:
            print(f'{grad_accum_steps=}')
            grad_accum = jax.tree_map(jnp.zeros_like, params)

        state = TrainState.create(
            apply_fn=module.apply,
            params=params,
            tx=tx,
            mixup_rng=jax.random.PRNGKey(train_state_config['mixup_seed']),
            dropout_rng=jax.random.PRNGKey(train_state_config['dropout_seed']),
            adv_rng=jax.random.PRNGKey(2036),
            ema_decay=train_state_config['ema_decay'],
            ema_params=copy.deepcopy(params) if train_state_config['ema_decay'] > 0 else None,
            micro_step=0,
            micro_in_mini=grad_accum_steps,
            grad_accum=grad_accum if grad_accum_steps > 1 else None,
        )
        return state




    def init_fn(init_rngs,example_inputs ) -> TrainState:
        params = module.init(init_rngs, **example_inputs, det=False)["params"]
        if grad_accum_steps > 1:
            print(f'{grad_accum_steps=}')
            grad_accum = jax.tree_map(jnp.zeros_like, params)

        state = TrainState.create(
            apply_fn=module.apply,
            params=params,
            tx=tx,
            mixup_rng=jax.random.PRNGKey(train_state_config['mixup_seed']),
            dropout_rng=jax.random.PRNGKey(train_state_config['dropout_seed']),
            adv_rng=jax.random.PRNGKey(2036),
            ema_decay=train_state_config['ema_decay'],
            ema_params=copy.deepcopy(params) if train_state_config['ema_decay'] > 0 else None,
            micro_step=0,
            micro_in_mini=grad_accum_steps,
            grad_accum=grad_accum if grad_accum_steps > 1 else None,
        )
        return state


    state_shapes = jax.eval_shape(init_fn, init_rngs,example_inputs, )
    train_state_partition = match_partition_rules(get_partition_rules_caformer(), state_shapes)
    # jax.sharding.NamedSharding(mesh,train_state_partition)
    train_state_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), train_state_partition)
    # print(type(train_state_sharding))

    # while True:
    #     pass

    opt_state = jax.tree_util.tree_map(
        lambda x: x.with_memory_kind(kind="pinned_host"), train_state_sharding.opt_state)

    params = jax.tree_util.tree_map(
        lambda x: x.with_memory_kind(kind="pinned_host"), train_state_sharding.params)
    train_state_off_load_sharding = train_state_sharding.replace(params=params,
                                                                 opt_state=opt_state)

    return state_shapes,train_state_sharding,init_fn,init_by_params_fn,init_rngs,example_inputs




def resume_checkpoint(pretrained_ckpt,state_shapes,train_state_sharding):
    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    ckpt = {'models': state_shapes}

    def set_sharding(sharding) -> ArrayRestoreArgs:
        return ocp.ArrayRestoreArgs(sharding=sharding)

    restore_args = {'models': jax.tree_util.tree_map(set_sharding, train_state_sharding)}
    state = checkpointer.restore(pretrained_ckpt, item=ckpt, restore_args=restore_args)
    return state


def init_state(train_state_config, image_size: int = 224, warmup_steps=1, training_steps=10,
         mesh=None, logical_axis_rules=None,restore_state_config=None,resume=False,remote_model_path=None):


    if restore_state_config is not None:
        train_state_config_cpy=copy.deepcopy(train_state_config)
        train_state_config_cpy_flatten=flax.traverse_util.flatten_dict(train_state_config_cpy,sep='/')
        restore_state_config_flatten=flax.traverse_util.flatten_dict(restore_state_config,sep='/')
        restore_state_config_flatten=train_state_config_cpy_flatten | restore_state_config_flatten
        restore_state_config=flax.traverse_util.unflatten_dict(restore_state_config_flatten,sep='/')
    else:
        restore_state_config = copy.deepcopy(train_state_config)





    (restore_state_shapes,
     restore_state_sharding,*_)=create_train_state(restore_state_config, image_size, warmup_steps, training_steps,
                                         mesh, logical_axis_rules)


    (state_shapes,
     train_state_sharding,init_fn,
     init_by_params_fn,init_rngs,example_inputs)=create_train_state(train_state_config, image_size, warmup_steps, training_steps,
                                         mesh, logical_axis_rules)




    if resume:
        print(remote_model_path)
        state=resume_checkpoint(remote_model_path,state_shapes,train_state_sharding)['models']
        # state=state.replace(params=copy.deepcopy(state.ema_params))
        return state,int(state.step) + 1,train_state_sharding

    pretrained_ckpt = restore_state_config.pop('pretrained_ckpt', None)

    if pretrained_ckpt is not None:

        if 'gs://' in pretrained_ckpt:
            state = resume_checkpoint(pretrained_ckpt, restore_state_shapes, restore_state_sharding)['models']
            params=state.ema_params
            del state
        else:
            params = load_pretrain(pretrained_ckpt)

        state=jax.jit(init_by_params_fn,out_shardings=train_state_sharding,donate_argnums=(0,))(params)
    else:
        state=jax.jit(init_fn,out_shardings=train_state_sharding)(init_rngs,example_inputs)

    return state,1,train_state_sharding






























def create_train_state2(train_state_config, image_size: int = 224,
                       warmup_steps=1, training_steps=10, mesh=None, logical_axis_rules=None
                       ):  # -> TrainState:

    model_config = train_state_config['models']
    optimizer_config = train_state_config['optimizer']
    train_module_config = train_state_config['train_module']
    grad_accum_steps=train_state_config.pop('grad_accum_steps', 1)


    model = get_obj_from_str(model_config['target'])(**model_config['model_kwargs'])
    print(f'{train_module_config=}')

    train_module = get_obj_from_str(train_module_config.pop('target'))  #(**model_config['model_kwargs'])

    module = train_module(
        model=model,
        mixup=Mixup(train_module_config.pop('mixup', ), train_module_config.pop('cutmix')),
        label_smoothing=train_module_config.pop('label_smoothing') if train_module_config['criterion'] != "bce" else 0,
        criterion=CRITERION_COLLECTION[train_module_config.pop('criterion')], **train_module_config
    )
    if jax.process_index() == 0:
        print(module)

    # Initialize the models weights with dummy inputs. Using the init RNGS and inputs, we
    # will tabulate the summary of models and its parameters. Furthermore, empty gradient
    # accumulation arrays will be prepared if the gradient accumulation is enabled.
    example_inputs = {
        "images": jnp.zeros((1, 3, image_size, image_size), dtype=jnp.uint8),
        # "labels": jnp.zeros((1,), dtype=jnp.int32),
        "labels": jnp.ones((1,), dtype=jnp.int32)  #jnp.array([1,2], dtype=jnp.int32),
    }

    init_rngs = {"params": jax.random.PRNGKey(train_state_config['init_seed'])}


    # if args.grad_accum > 1:
    #     grad_accum = jax.tree_map(jnp.zeros_like, params)
    lr = optimizer_config['optimizer_kwargs'].pop('learning_rate')
    end_lr = optimizer_config['optimizer_kwargs'].pop('end_learning_rate', 1e-5)
    init_value = optimizer_config['optimizer_kwargs'].pop('init_value', 1e-6)
    schedule = optimizer_config.get('schedule', 'cosine')
    clip_grad = optimizer_config.get('clip_grad', 1.0)

    # Create learning rate scheduler and optimizer with gradient clipping. The learning
    # rate will be recorded at `hyperparams` by `optax.inject_hyperparameters`.

    @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
    def create_optimizer_fn(
            learning_rate: optax.Schedule,
    ) -> optax.GradientTransformation:

        tx = OPTIMIZER_COLLECTION[optimizer_config['target']](
            learning_rate=learning_rate,
            **optimizer_config['optimizer_kwargs'],
            mask=partial(jax.tree_util.tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
        )
        print(f'{clip_grad=}')
        if clip_grad is not None:
            tx = optax.chain(optax.clip_by_global_norm(clip_grad), tx)
        return tx


    if schedule == 'constant':
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=lr,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=training_steps,
            end_value=lr,
        )

    elif schedule=="wsd":
        learning_rate = warmup_stable_cosine_decay_schedule(
            init_value=init_value,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=training_steps,
            end_value=end_lr,
        )


    else:
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=init_value,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=training_steps,
            end_value=end_lr,
        )

    tx = create_optimizer_fn(learning_rate, )

    def init_fn(init_rngs,example_inputs ) -> TrainState:
        params = module.init(init_rngs, **example_inputs, det=False)["params"]
        if grad_accum_steps > 1:
            print(f'{grad_accum_steps=}')
            grad_accum = jax.tree_map(jnp.zeros_like, params)

        state = TrainState.create(
            apply_fn=module.apply,
            params=params,
            tx=tx,
            mixup_rng=jax.random.PRNGKey(train_state_config['mixup_seed']),
            dropout_rng=jax.random.PRNGKey(train_state_config['dropout_seed']),
            adv_rng=jax.random.PRNGKey(2036),
            ema_decay=train_state_config['ema_decay'],
            ema_params=copy.deepcopy(params) if train_state_config['ema_decay'] > 0 else None,
            micro_step=0,
            micro_in_mini=grad_accum_steps,
            grad_accum=grad_accum if grad_accum_steps > 1 else None,
        )
        return state


    state_shapes = jax.eval_shape(init_fn, init_rngs,example_inputs, )
    train_state_partition = match_partition_rules(get_partition_rules_caformer(), state_shapes)
    # jax.sharding.NamedSharding(mesh,train_state_partition)
    train_state_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), train_state_partition)




    return state_shapes,train_state_sharding,init_fn,init_rngs,example_inputs



def init_state_restore(train_state_config, image_size: int = 224, warmup_steps=1, training_steps=10,
         mesh=None, logical_axis_rules=None,restore_state_config=None,resume=False,remote_model_path=None):


    if restore_state_config is not None:
        train_state_config_cpy=copy.deepcopy(train_state_config)
        train_state_config_cpy_flatten=flax.traverse_util.flatten_dict(train_state_config_cpy,sep='/')
        restore_state_config_flatten=flax.traverse_util.flatten_dict(restore_state_config,sep='/')
        restore_state_config_flatten=train_state_config_cpy_flatten | restore_state_config_flatten
        restore_state_config=flax.traverse_util.unflatten_dict(restore_state_config_flatten,sep='/')
    else:
        restore_state_config = copy.deepcopy(train_state_config)





    (restore_state_shapes,
     restore_state_sharding,*_)=create_train_state2(restore_state_config, image_size, warmup_steps, training_steps,
                                         mesh, logical_axis_rules)


    (state_shapes,
     train_state_sharding,init_fn,
     init_rngs,example_inputs)=create_train_state2(train_state_config, image_size, warmup_steps, training_steps,
                                         mesh, logical_axis_rules)

    # train_state_sharding = jax.tree_util.tree_map(
    #     lambda x: x.with_memory_kind(kind="pinned_host"), train_state_sharding)


    if resume:
        print(remote_model_path)
        state=resume_checkpoint(remote_model_path,state_shapes,train_state_sharding)['models']
        # state=state.replace(params=copy.deepcopy(state.ema_params))
        return state





if __name__ == "__main__":
    os.environ['GCS_DATASET_DIR'] = 'hello'

    yaml = read_yaml('configs/adv/convnext-b-3step-200ep-ft.yaml')
    yaml = preprocess_config(yaml)

    # print(os.environ.get('GCS_DATASET_DIR'))

    # print(yaml)
    # print(json.dumps(yaml, indent=5))
    #
    # while True:
    #     pass

    state = create_train_state(yaml['train_state'])
    # print(state)
    state = state.replicate()
