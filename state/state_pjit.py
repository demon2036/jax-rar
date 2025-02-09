






import copy
from copy import deepcopy
from functools import partial

import flax
import jax
import optax
import timm
import torch
from huggingface_hub import hf_hub_download
from jax._src.pjit import pjit
from orbax.checkpoint._src.handlers.standard_checkpoint_handler import StandardRestoreArgs

# from training import TrainState
# from utils import read_yaml, get_obj_from_str, Mixup, preprocess_config, match_partition_rules, \
#     get_partition_rules_caformer
import os
import jax.numpy as jnp

from convert_model_torch_to_flax.convert_rar_torch_to_flax import convert_torch_to_flax_rar
from convert_model_torch_to_flax.convert_vqgan_torch_to_flax import convert_vqgan_state_dict
from models import PretrainedTokenizer
from pytorch_rar import demo_util
from sampler import RARConfig
from state.train_state import TrainState
from utils.utils import get_obj_from_str, match_partition_rules, get_partition_rules_caformer, read_yaml, \
    preprocess_config
import orbax.checkpoint as ocp


def resume_checkpoint(pretrained_ckpt,state_shapes,train_state_sharding):
    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    ckpt = {'models': state_shapes}

    def set_sharding(sharding) -> ocp.ArrayRestoreArgs:
        return ocp.ArrayRestoreArgs(sharding=sharding)

    restore_args = {'models': jax.tree_util.tree_map(set_sharding, train_state_sharding)}
    state = checkpointer.restore(pretrained_ckpt, item=ckpt, restore_args=restore_args)
    return state






def get_torch_model_from_rar_size(rar_model_size='rar_xxl'):
    # Choose one from ["rar_b_imagenet", "rar_l_imagenet", "rar_xl_imagenet", "rar_xxl_imagenet"]
    # rar_model_size = ["rar_b", "rar_l", "rar_xl", "rar_xxl"][-1]

    assert rar_model_size in ["rar_b", "rar_l", "rar_xl", "rar_xxl"]

    # local_dir= '../'
    local_dir= '/root/'

    class ConfigTokenizer:
        channel_mult = [1, 1, 2, 2, 4]
        num_resolutions = 5
        dropout = 0.0
        hidden_channels = 128
        num_channels = 3
        num_res_blocks = 2
        resolution = 256
        # resolution = 32
        z_channels = 256

    config_tokenizer = ConfigTokenizer()

    # download the maskgit-vq tokenizer
    hf_hub_download(repo_id="fun-research/TiTok", filename=f"maskgit-vqgan-imagenet-f16-256.bin",
                    local_dir=local_dir
                    )
    # download the rar generator weight
    hf_hub_download(repo_id="yucornetto/RAR", filename=f"{rar_model_size}.bin",
                    local_dir=local_dir
                    )

    config = demo_util.get_config("./pytorch_rar/configs/training/generator/rar.yaml")
    config.experiment.generator_checkpoint = f"{rar_model_size}.bin"
    config.model.generator.hidden_size = {"rar_b": 768, "rar_l": 1024, "rar_xl": 1280, "rar_xxl": 1408}[rar_model_size]
    config.model.generator.num_hidden_layers = {"rar_b": 24, "rar_l": 24, "rar_xl": 32, "rar_xxl": 40}[rar_model_size]
    config.model.generator.num_attention_heads = 16
    config.model.generator.intermediate_size = {"rar_b": 3072, "rar_l": 4096, "rar_xl": 5120, "rar_xxl": 6144}[
        rar_model_size]

    config.experiment.generator_checkpoint = f"{local_dir+rar_model_size}.bin"
    config.model.vq_model.pretrained_tokenizer_weight=f"{local_dir+'maskgit-vqgan-imagenet-f16-256.bin'}"


    model_params = convert_torch_to_flax_rar(torch.load(config.experiment.generator_checkpoint ))
    # tokenizer_params = convert_vqgan_state_dict(torch.load(config.model.vq_model.pretrained_tokenizer_weight))
    # model_params = convert_torch_to_flax_rar(generator.state_dict())
    # tokenizer_params = convert_vqgan_state_dict(tokenizer.state_dict())
    # tokenizer = PretrainedTokenizer(config=config_tokenizer)
    # rar_config=RARConfig(hidden_size=config.model.generator.hidden_size,num_hidden_layers=config.model.generator.num_hidden_layers)
    return model_params#,tokenizer_params,model,tokenizer,rar_config


def get_jax_tokenizer():
    local_dir = '/root/'

    class ConfigTokenizer:
        channel_mult = [1, 1, 2, 2, 4]
        num_resolutions = 5
        dropout = 0.0
        hidden_channels = 128
        num_channels = 3
        num_res_blocks = 2
        resolution = 256
        # resolution = 32
        z_channels = 256

    config_tokenizer = ConfigTokenizer()
    # download the maskgit-vq tokenizer
    hf_hub_download(repo_id="fun-research/TiTok", filename=f"maskgit-vqgan-imagenet-f16-256.bin",
                    local_dir=local_dir
                    )

    config = demo_util.get_config("./pytorch_rar/configs/training/generator/rar.yaml")
    config.model.vq_model.pretrained_tokenizer_weight = f"{local_dir + 'maskgit-vqgan-imagenet-f16-256.bin'}"
    tokenizer_params = convert_vqgan_state_dict(torch.load(config.model.vq_model.pretrained_tokenizer_weight))
    tokenizer = PretrainedTokenizer(config=config_tokenizer)
    return tokenizer,tokenizer_params


def load_pretrain(pretrained_model='convnext_base.fb_in1k', default_params=None):
    model_jax_params=get_torch_model_from_rar_size(pretrained_model)

    model_jax_params = jax.tree_util.tree_map(jnp.asarray, model_jax_params)
    return {'models': model_jax_params}


def create_train_state2(train_state_config,
                       warmup_steps=1, training_steps=10, mesh=None, logical_axis_rules=None
                       ):  # -> TrainState:

    model_config = train_state_config['model']
    optimizer_config = train_state_config['optimizer']
    train_module_config = train_state_config['train_module']
    grad_accum_steps=train_state_config.pop('grad_accum_steps', 1)



    config= get_obj_from_str(model_config['config']['target'])(**model_config['config']['kwargs'])
    model = get_obj_from_str(model_config['target'])(config=config)
    print(f'{train_module_config=}')
    train_module = get_obj_from_str(train_module_config.pop('target'))  #(**model_config['model_kwargs'])
    module = train_module(model=model,ref_model=deepcopy(model))

    if jax.process_index() == 0:
        print(module)

    # Initialize the models weights with dummy inputs. Using the init RNGS and inputs, we
    # will tabulate the summary of models and its parameters. Furthermore, empty gradient
    # accumulation arrays will be prepared if the gradient accumulation is enabled.
    example_inputs = {
        "tokens": jnp.zeros((1,256), dtype=jnp.int32),
        "labels": jnp.ones((1,), dtype=jnp.int32) ,
        "siglip_feature": jnp.ones((1,1024), dtype=jnp.float32)  ,
        "dino_feature": jnp.ones((1, 1024), dtype=jnp.float32),
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
        tx=optax.sgd(1)

        # tx = OPTIMIZER_COLLECTION[optimizer_config['target']](
        #     learning_rate=learning_rate,
        #     **optimizer_config['optimizer_kwargs'],
        #     mask=partial(jax.tree_util.tree_map_with_path, lambda kp, *_: kp[-1].key == "kernel"),
        # )
        # print(f'{clip_grad=}')
        # if clip_grad is not None:
        #     tx = optax.chain(optax.clip_by_global_norm(clip_grad), tx)
        return tx


    if schedule == 'constant':
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=lr,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=training_steps,
            end_value=lr,
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
            ref_model_params=copy.deepcopy(params),
            tx=tx,
            dropout_rng=jax.random.PRNGKey(train_state_config['dropout_seed']),
            ema_decay=train_state_config['ema_decay'],
            ema_params=copy.deepcopy(params) if train_state_config['ema_decay'] > 0 else None,
            micro_step=0,
            micro_in_mini=grad_accum_steps,
            grad_accum=grad_accum if grad_accum_steps > 1 else None,
        )
        return state




    def init_fn(init_rngs,example_inputs ) -> TrainState:
        params = module.init(init_rngs, **example_inputs, det=False)["params"]
        model_params=params['model']
        # ref_model_params=params['ref_model']

        if grad_accum_steps > 1:
            print(f'{grad_accum_steps=}')
            grad_accum = jax.tree_map(jnp.zeros_like, params)

        state = TrainState.create(
            apply_fn=module.apply,
            params=model_params,
            ref_model_params=copy.deepcopy(model_params),
            tx=tx,
            dropout_rng=jax.random.PRNGKey(train_state_config['dropout_seed']),
            ema_decay=train_state_config['ema_decay'],
            ema_params=copy.deepcopy(model_params) if train_state_config['ema_decay'] > 0 else None,
            micro_step=0,
            micro_in_mini=grad_accum_steps,
            grad_accum=grad_accum if grad_accum_steps > 1 else None,
        )
        return state


    state_shapes = jax.eval_shape(init_fn, init_rngs,example_inputs, )
    train_state_partition = match_partition_rules(get_partition_rules_caformer(), state_shapes)
    train_state_sharding = jax.tree_util.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), train_state_partition)


    model_config_kwargs= model_config['config']['kwargs']
    rar_config=RARConfig(hidden_size=model_config_kwargs['embed_dim'],num_hidden_layers=model_config_kwargs['depth'])


    return state_shapes,train_state_sharding,init_fn,init_by_params_fn,init_rngs,example_inputs,rar_config,model




def init_state(train_state_config,warmup_steps=1, training_steps=10,
         mesh=None, logical_axis_rules=None,restore_state_config=None,resume=False,remote_model_path=None):


    # state_shapes,train_state_sharding,init_fn,init_by_params_fn,init_rngs,example_inputs=create_train_state2(train_state_config,mesh=mesh)
    # state = jax.jit(init_fn, out_shardings=train_state_sharding)(init_rngs, example_inputs)




    if restore_state_config is not None:
        train_state_config_cpy=copy.deepcopy(train_state_config)
        train_state_config_cpy_flatten=flax.traverse_util.flatten_dict(train_state_config_cpy,sep='/')
        restore_state_config_flatten=flax.traverse_util.flatten_dict(restore_state_config,sep='/')
        restore_state_config_flatten=train_state_config_cpy_flatten | restore_state_config_flatten
        restore_state_config=flax.traverse_util.unflatten_dict(restore_state_config_flatten,sep='/')
    else:
        restore_state_config = copy.deepcopy(train_state_config)


    (restore_state_shapes,
     restore_state_sharding,*_)=create_train_state2(restore_state_config, warmup_steps, training_steps,
                                         mesh, logical_axis_rules)


    (state_shapes,
     train_state_sharding,init_fn,
     init_by_params_fn,init_rngs,example_inputs,rar_config,model,)=create_train_state2(train_state_config, warmup_steps, training_steps,
                                         mesh, logical_axis_rules)


    if resume:
        print(remote_model_path)
        state=resume_checkpoint(remote_model_path,state_shapes,train_state_sharding)['models']
        # state=state.replace(params=copy.deepcopy(state.ema_params))
        return state,int(state.step) + 1,train_state_sharding,rar_config,model

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

    return state,1,train_state_sharding,rar_config,model









if __name__ == "__main__":
    os.environ['GCS_DATASET_DIR'] = 'hello'

    yaml = read_yaml('../configs/test.yaml')
    yaml = preprocess_config(yaml)

    state = init_state(yaml['train_state'])
