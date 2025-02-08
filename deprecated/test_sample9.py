import time
from functools import partial
from typing import Any

import einops
import torch
import tqdm
from PIL import Image
import numpy as np
import demo_util
from huggingface_hub import hf_hub_download

from maskgit_vqgan import PretrainedTokenizer
from test import convert_vqgan_state_dict
from rar import FlaxRAR, convert_torch_to_flax_rar, init_cache
from utils.train_utils import create_pretrained_tokenizer

import jax
# jax.config.update('jax_platform_name', 'cpu')
import flax
import jax.numpy as jnp
import chex


# Choose one from ["rar_b_imagenet", "rar_l_imagenet", "rar_xl_imagenet", "rar_xxl_imagenet"]



@chex.dataclass
class SampleState:
    decoding_step: jnp.int32
    token_buffer: jnp.ndarray
    position_ids: jnp.ndarray
    cache: Any
    attn_mask: jnp.ndarray
    key: jnp.ndarray


class RARConfig:
    def __init__(self, hidden_size=768, num_attention_heads=16, num_hidden_layers=24):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        # self.head_dim


def sample( key,params,tokenizer_params, model,tokenizer_jax, config, batch_size=1,randomize_temperature=1.02):
    image_seq_len = 256

    #condition=None,

    key,key_sample=jax.random.split(key)

    condition = jax.random.randint(key, (batch_size, 1), 0, 1000)
    guidance_scale = 16.0
    scale_pow = 2.75
    num_samples = batch_size

    step = jnp.arange(0, image_seq_len)
    scale_step = (1 - jnp.cos(
        ((step / image_seq_len) ** scale_pow) * jnp.pi)) * 1 / 2

    cfg_scale = (guidance_scale - 1) * scale_step + 1

    max_cache_length = 258
    cache = init_cache(config, num_samples * 2 if guidance_scale != 0 else num_samples,
                       max_cache_length=max_cache_length, dtype=jnp.float32)

    attn_mask = jnp.zeros((1, max_cache_length), dtype=jnp.int32)
    prefill_jit = jax.jit(partial(model.apply, method=FlaxRAR.prefill))

    condition_jax = condition + 1024 + 1
    none_condition = model.apply({'params': params}, condition_jax, method=model.get_none_condition)
    if guidance_scale != 0:
        c = jnp.concat([condition_jax, none_condition], axis=0)
        logits, cache = prefill_jit({'params': params}, c,
                                    cache, attn_mask=attn_mask)
        cond_logits, uncond_logits = logits[:num_samples], logits[num_samples:]
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale[0]
    else:
        logits, cache = prefill_jit({'params': params}, condition_jax, cache, attn_mask=attn_mask)

    token_buffer = jnp.zeros((batch_size, 256), jnp.int32)

    position_ids = jnp.arange(0, 1)[None, ...]

    logits = logits / randomize_temperature
    next_token=jax.random.categorical(key_sample,logits[:, -1])#.reshape((-1,1))
    # next_token = jnp.argmax(logits[:, -1], axis=-1)

    token_buffer = token_buffer.at[:, 0].set(next_token)
    decode_jit = jax.jit(partial(model.apply, method=FlaxRAR.decode))
    attn_mask = attn_mask.at[:, :2].set(1)
    print('go')

    sample_state = SampleState(decoding_step=0, token_buffer=token_buffer,
                               position_ids=position_ids,cache=cache,attn_mask=attn_mask,key=key)

    def loop_body(i, sample_state):
        sample_state.attn_mask = sample_state.attn_mask.at[:, i + 2].set(1)
        last_token=sample_state.token_buffer[:,i]
        last_token=last_token.reshape((sample_state.token_buffer.shape[0],1))

        if guidance_scale != 0:
            logits, cache = decode_jit({'params': params},
                                       jnp.concat([last_token,last_token], axis=0),
                                       jnp.concat([condition_jax, none_condition], axis=0),
                                       sample_state.position_ids, sample_state.cache, sample_state.attn_mask,)


            cond_logits, uncond_logits = logits[:num_samples], logits[num_samples:]
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale[i + 1]
        else:
            logits, cache = decode_jit({'params': params},last_token, condition_jax,
                                       sample_state.position_ids, sample_state.cache, sample_state.attn_mask)

        sample_state.key, key_sample = jax.random.split(sample_state.key)
        logits = logits / randomize_temperature
        next_token = jax.random.categorical(key_sample, logits[:, -1])  # .reshape((-1,1))

        # next_token = jnp.argmax(logits[:, -1], axis=-1)
        sample_state.token_buffer = sample_state.token_buffer.at[:, i + 1].set(next_token)
        sample_state.position_ids += 1
        sample_state.cache=cache

        return sample_state

    sample_state=jax.lax.fori_loop(0,255,loop_body,sample_state)
    token_buffer=sample_state.token_buffer

    reconstructed = tokenizer_jax.apply({'params':tokenizer_params}, token_buffer, method=PretrainedTokenizer.decode, deterministic=True)
    generated_image = jnp.array(reconstructed * 255.0, dtype=np.uint8)

    return generated_image

    # return token_buffer



def init_model():
    # Choose one from ["rar_b_imagenet", "rar_l_imagenet", "rar_xl_imagenet", "rar_xxl_imagenet"]
    rar_model_size = ["rar_b", "rar_l", "rar_xl", "rar_xxl"][0]
    local_dir= '../'

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
    hf_hub_download(repo_id="fun-research/TiTok", filename=f"../maskgit-vqgan-imagenet-f16-256.bin",
                    local_dir=local_dir
                    )
    # download the rar generator weight
    hf_hub_download(repo_id="yucornetto/RAR", filename=f"{rar_model_size}.bin",
                    local_dir=local_dir
                    )

    config = demo_util.get_config("../configs/training/generator/rar.yaml")
    config.experiment.generator_checkpoint = f"{rar_model_size}.bin"
    config.model.generator.hidden_size = {"rar_b": 768, "rar_l": 1024, "rar_xl": 1280, "rar_xxl": 1408}[rar_model_size]
    config.model.generator.num_hidden_layers = {"rar_b": 24, "rar_l": 24, "rar_xl": 32, "rar_xxl": 40}[rar_model_size]
    config.model.generator.num_attention_heads = 16
    config.model.generator.intermediate_size = {"rar_b": 3072, "rar_l": 4096, "rar_xl": 5120, "rar_xxl": 6144}[
        rar_model_size]

    config.experiment.generator_checkpoint = f"{local_dir+rar_model_size}.bin"
    config.model.vq_model.pretrained_tokenizer_weight=f"{local_dir+'maskgit-vqgan-imagenet-f16-256.bin'}"
    # maskgit-vq as tokenizer
    tokenizer = create_pretrained_tokenizer(config)
    generator = demo_util.get_rar_generator(config)

    model = FlaxRAR(config=config)
    model_params = convert_torch_to_flax_rar(generator.state_dict())
    tokenizer_params = convert_vqgan_state_dict(tokenizer.state_dict())
    tokenizer = PretrainedTokenizer(config=config_tokenizer)

    # print(tokenizer_params.keys())

    return model_params,tokenizer_params,model,tokenizer


def main3():
    # 初始化模型
    model_params,tokenizer_params,model,tokenizer_jax = init_model()

    # 构造随机输入：input_ids（整数 token）和 condition（假设为 [B, 1]）
    batch_size = 8
    rng = jax.random.PRNGKey(2)
    sample_rng, dropout_rng = jax.random.split(rng)


    # cls=3
    # condition = jax.random.randint(sample_rng, (batch_size, 1), cls,
    #                                cls)

    # 构造 rngs 字典
    sample_jit=jax.jit(partial(sample,model=model,config=RARConfig(),batch_size=batch_size,tokenizer_jax=tokenizer_jax))
    start=time.time()
    # token_buffer=sample_jit(model_params,sample_rng)
    sample_img=sample_jit(sample_rng,model_params,tokenizer_params)
    sample_img.block_until_ready()
    print(time.time()-start)

    # generated_image = np.array(reconstructed * 255.0,dtype=np.uint8)
    generated_image=np.array(sample_img)
    print(generated_image.shape)
    Image.fromarray(generated_image[0]).save(f"assets/rar_generated_{1}.png")





if __name__ == "__main__":
    # main()
    # main3()
    # main2()
    main3()