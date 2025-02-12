from functools import partial
from typing import Any
import chex
import flax
import jax
import jax.numpy as jnp
from models.maskgit_vqgan import PretrainedTokenizer
from models.rar import FlaxRAR, init_cache


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



def sample( key,params,tokenizer_params, model,tokenizer_jax, config, batch_size=1,
            guidance_scale = 8.0,
            scale_pow = 1.2,
            randomize_temperature=1.02):
    image_seq_len = 256

    def choice(logits,key):
        logits=logits/randomize_temperature
        logits=flax.linen.softmax(logits,axis=-1)
        return jax.random.choice(key, a=jnp.arange(0, logits.shape[0]), p=logits)

    vmap_choice = jax.vmap(choice)


    origin_key=key
    key,key_prefill,key_decode=jax.random.split(key[0],3)

    labels = jax.random.randint(key, (batch_size, 1), 0, 1000)


    num_samples = batch_size

    step = jnp.arange(0, image_seq_len)
    scale_step = (1 - jnp.cos(
        ((step / image_seq_len) ** scale_pow) * jnp.pi)) * 1 / 2

    cfg_scale = (guidance_scale - 1) * scale_step + 1
    max_cache_length = 256
    cache = init_cache(config, num_samples * 2 if guidance_scale != 0 else num_samples,
                       max_cache_length=max_cache_length, dtype=jnp.bfloat16)

    attn_mask = jnp.zeros((1, max_cache_length), dtype=jnp.int32)
    prefill_jit = jax.jit(partial(model.apply, method=FlaxRAR.prefill))

    condition_jax = labels + 1024 + 1
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

    next_token = vmap_choice(logits[:, -1], jax.random.split(key_prefill, logits.shape[0]))
    # logits = logits / randomize_temperature
    # next_token=jax.random.categorical(key_prefill,logits[:, -1])
    # next_token = jnp.argmax(logits[:, -1], axis=-1)

    token_buffer = token_buffer.at[:, 0].set(next_token)
    decode_jit = jax.jit(partial(model.apply, method=FlaxRAR.decode))
    attn_mask = attn_mask.at[:, :2].set(1)
    print('go')

    sample_state = SampleState(decoding_step=0, token_buffer=token_buffer,
                               position_ids=position_ids,cache=cache,attn_mask=attn_mask,key=key_decode)

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
        next_token = vmap_choice(logits[:, -1], jax.random.split(key_sample, logits.shape[0]))
        # logits = logits / randomize_temperature
        # next_token = jax.random.categorical(key_sample, logits[:, -1])  # .reshape((-1,1))

        # next_token = jnp.argmax(logits[:, -1], axis=-1)
        sample_state.token_buffer = sample_state.token_buffer.at[:, i + 1].set(next_token)
        sample_state.position_ids += 1
        sample_state.cache=cache

        return sample_state

    sample_state=jax.lax.fori_loop(0,255,loop_body,sample_state)
    token_buffer=sample_state.token_buffer
    del cache

    reconstructed = tokenizer_jax.apply({'params':tokenizer_params}, token_buffer, method=PretrainedTokenizer.decode, deterministic=True)
    generated_image = jnp.array(jnp.clip(reconstructed * 255.0,0,255), dtype=jnp.uint8)
    return origin_key.at[0].set(key),generated_image,labels




































def sample_cfg( key,params,tokenizer_params,
            model,tokenizer_jax, config, batch_size=1,
            guidance_scale = 8.0,
            scale_pow = 1.2,
            randomize_temperature=1.02):
    image_seq_len = 256

    def choice(logits,key):
        logits=logits/randomize_temperature
        logits=flax.linen.softmax(logits,axis=-1)
        return jax.random.choice(key, a=jnp.arange(0, logits.shape[0]), p=logits)

    vmap_choice = jax.vmap(choice)


    origin_key=key
    key,key_prefill,key_decode,key_choice,key_cfg=jax.random.split(key[0],5)

    labels = jax.random.randint(key, (batch_size, 1), 0, 1000)


    num_samples = batch_size

    step = jnp.arange(0, image_seq_len)
    scale_step = (1 - jnp.cos(
        ((step / image_seq_len) ** scale_pow) * jnp.pi)) * 1 / 2

    cfg_scale = (guidance_scale - 1) * scale_step + 1


    choice = jax.random.randint(key_choice, (batch_size,), 16, 192)[..., None]
    # x = jnp.arange(0, image_seq_len)[None, ...]
    var_cfg = jax.random.randint(key_cfg, (batch_size,), 8, 100)[..., None] #* 10
    # print(jnp.where(jnp.arange(0, 256)[None, ...] < choice, x, var_cfg))
    cfg_scale=jnp.where(jnp.arange(0, image_seq_len)[None, ...] < choice, cfg_scale[None,...], var_cfg)


    print(f'{cfg_scale[:,0].shape=}')



    max_cache_length = 256
    cache = init_cache(config, num_samples * 2 if guidance_scale != 0 else num_samples,
                       max_cache_length=max_cache_length, dtype=jnp.bfloat16)

    attn_mask = jnp.zeros((1, max_cache_length), dtype=jnp.int32)
    prefill_jit = jax.jit(partial(model.apply, method=FlaxRAR.prefill))

    condition_jax = labels + 1024 + 1
    none_condition = model.apply({'params': params}, condition_jax, method=model.get_none_condition)
    if guidance_scale != 0:
        c = jnp.concat([condition_jax, none_condition], axis=0)
        logits, cache = prefill_jit({'params': params}, c,
                                    cache, attn_mask=attn_mask)
        cond_logits, uncond_logits = logits[:num_samples], logits[num_samples:]
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale[:,0]
    else:
        logits, cache = prefill_jit({'params': params}, condition_jax, cache, attn_mask=attn_mask)

    token_buffer = jnp.zeros((batch_size, 256), jnp.int32)

    position_ids = jnp.arange(0, 1)[None, ...]

    next_token = vmap_choice(logits[:, -1], jax.random.split(key_prefill, logits.shape[0]))
    # logits = logits / randomize_temperature
    # next_token=jax.random.categorical(key_prefill,logits[:, -1])
    # next_token = jnp.argmax(logits[:, -1], axis=-1)

    token_buffer = token_buffer.at[:, 0].set(next_token)
    decode_jit = jax.jit(partial(model.apply, method=FlaxRAR.decode))
    attn_mask = attn_mask.at[:, :2].set(1)
    print('go')

    sample_state = SampleState(decoding_step=0, token_buffer=token_buffer,
                               position_ids=position_ids,cache=cache,attn_mask=attn_mask,key=key_decode)

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
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale[:,i + 1]
        else:
            logits, cache = decode_jit({'params': params},last_token, condition_jax,
                                       sample_state.position_ids, sample_state.cache, sample_state.attn_mask)

        sample_state.key, key_sample = jax.random.split(sample_state.key)
        next_token = vmap_choice(logits[:, -1], jax.random.split(key_sample, logits.shape[0]))
        # logits = logits / randomize_temperature
        # next_token = jax.random.categorical(key_sample, logits[:, -1])  # .reshape((-1,1))

        # next_token = jnp.argmax(logits[:, -1], axis=-1)
        sample_state.token_buffer = sample_state.token_buffer.at[:, i + 1].set(next_token)
        sample_state.position_ids += 1
        sample_state.cache=cache

        return sample_state

    sample_state=jax.lax.fori_loop(0,255,loop_body,sample_state)
    token_buffer=sample_state.token_buffer
    del cache

    reconstructed = tokenizer_jax.apply({'params':tokenizer_params}, token_buffer, method=PretrainedTokenizer.decode, deterministic=True)
    generated_image = jnp.array(jnp.clip(reconstructed * 255.0,0,255), dtype=jnp.uint8)
    return origin_key.at[0].set(key),generated_image,labels