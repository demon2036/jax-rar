import os
from functools import partial
from typing import Any

import PIL
import tqdm
from PIL import Image
import numpy as np
from jax.experimental import mesh_utils

from jax_fid import fid, inception
from pytorch_rar import demo_util
from huggingface_hub import hf_hub_download

from models.maskgit_vqgan import PretrainedTokenizer
from convert_model_torch_to_flax.convert_vqgan_torch_to_flax import convert_vqgan_state_dict
from models.rar import FlaxRAR, convert_torch_to_flax_rar, init_cache, FlaxRARConfig
from pytorch_rar.utils.train_utils import create_pretrained_tokenizer

import jax
# jax.config.update('jax_platform_name', 'cpu')
import flax
import jax.numpy as jnp
import chex
from jax.sharding import PartitionSpec as P ,NamedSharding,Mesh

from jax.experimental.shard_map import shard_map

from utils.generate_utils import create_npz_from_np


def init_model():
    # Choose one from ["rar_b_imagenet", "rar_l_imagenet", "rar_xl_imagenet", "rar_xxl_imagenet"]
    rar_model_size = ["rar_b", "rar_l", "rar_xl", "rar_xxl"][-1]
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
    # maskgit-vq as tokenizer
    tokenizer = create_pretrained_tokenizer(config)
    generator = demo_util.get_rar_generator(config)


    flax_config=FlaxRARConfig(embed_dim=config.model.generator.hidden_size,
                              depth=config.model.generator.num_hidden_layers,
                              intermediate_size=config.model.generator.intermediate_size)

    model = FlaxRAR(config=flax_config)
    model_params = convert_torch_to_flax_rar(generator.state_dict())
    tokenizer_params = convert_vqgan_state_dict(tokenizer.state_dict())
    tokenizer = PretrainedTokenizer(config=config_tokenizer)

    rar_config=RARConfig(hidden_size=config.model.generator.hidden_size,num_hidden_layers=config.model.generator.num_hidden_layers)
    return model_params,tokenizer_params,model,tokenizer,rar_config




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
            randomize_temperature=1.02
            ):
    image_seq_len = 256

    def choice(logits,key):
        logits=logits/randomize_temperature
        logits=flax.linen.softmax(logits,axis=-1)
        return jax.random.choice(key, a=jnp.arange(0, logits.shape[0]), p=logits)

    vmap_choice = jax.vmap(choice)

    origin_key=key
    key,key_prefill,key_decode=jax.random.split(key[0],3)
    condition = jax.random.randint(key, (batch_size, 1), 0, 1000)

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

    next_token = vmap_choice(logits[:, -1], jax.random.split(key_prefill, logits.shape[0]))

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

        # next_token = jnp.argmax(logits[:, -1], axis=-1)
        sample_state.token_buffer = sample_state.token_buffer.at[:, i + 1].set(next_token)
        sample_state.position_ids += 1
        sample_state.cache=cache

        return sample_state

    sample_state=jax.lax.fori_loop(0,255,loop_body,sample_state)
    token_buffer=sample_state.token_buffer

    reconstructed = tokenizer_jax.apply({'params':tokenizer_params}, token_buffer, method=PretrainedTokenizer.decode, deterministic=True)
    generated_image = jnp.array(jnp.clip(reconstructed * 255.0,0,255), dtype=jnp.uint8)
    return origin_key.at[0].set(key),generated_image









class Sampler:

    def __init__(self,model,tokenizer,tokenizer_params,rar_config:RARConfig,batch_size=128,fid_model=None,fid_model_params=None):
        self.model=model
        self.tokenizer=tokenizer
        self.tokenizer_params=tokenizer_params
        self.rng=rng = jax.random.PRNGKey(0)
        sample_rng, dropout_rng = jax.random.split(rng)
        self.sample_rng = jax.random.split(sample_rng, jax.device_count())
        physical_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
        mesh = Mesh(physical_mesh, ('dp',))
        self.batch_size=batch_size


        sample_fn = partial(sample, model=model, config=rar_config, batch_size=batch_size, tokenizer_jax=tokenizer)
        sample_fn = shard_map(sample_fn, mesh=mesh, in_specs=(
            P('dp'), P(None), P(None)
        ),
            out_specs=P('dp')
        )
        self.sample_jit = jax.jit(sample_fn,out_shardings=NamedSharding(mesh,P(None)))

        self.fid_model=fid_model
        self.fid_model_params=fid_model_params
        self.fid_eval_batch_size=1024

        def fid_apply_fn(x,params):

            return self.fid_model.apply(params,x)#.with_memory_kind(kind="pinned_host")

        self.fid_apply_fn = shard_map(fid_apply_fn, mesh=mesh, in_specs=(
            P('dp'),P(None)
        ),
             out_specs=P('dp')
            # out_specs=P(None),check_rep=False
        )

        self.fid_apply_fn_jit=jax.jit(self.fid_apply_fn,out_shardings=NamedSharding(mesh,P(None)))


    def compute_array_statistics(self,x):
        data=x
        print(data)
        images = []
        for img in tqdm.tqdm(data):
            img=PIL.Image.fromarray(img)
            img = img.resize(
                size=(299,299),
                resample=Image.BILINEAR,
            )
            img = np.array(img,dtype=np.float32) / 255.0
            # print(img)
            images.append(img)

        num_batches = int(len(images) // self.batch_size)
        act = []
        for i in tqdm.tqdm(range(num_batches)):
            x = images[i * self.batch_size: i * self.batch_size + self.batch_size]
            x = np.asarray(x)
            x = 2 * x - 1
            pred = self.fid_apply_fn_jit(jax.lax.stop_gradient(x),self.fid_model_params)
            act.append(pred.squeeze(axis=1).squeeze(axis=1))
        act = jnp.concatenate(act, axis=0)

        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma


    def computer_fid(self,x):
        mu1, sigma1 = self.compute_array_statistics(x)
        mu2, sigma2 = fid.compute_statistics("VIRTUAL_imagenet256_labeled.npz", None,None,None,None)
        fid_score = fid.compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6)
        print('Fid:', fid_score)
        return fid_score


    def sample(self,params,save_npz):
        # 构造 rngs 字典
        sample_rng=self.sample_rng
        data = []
        # iters = 100
        # iters = 51200//jax.device_count()
        iters=10
        for _ in tqdm.tqdm(range(iters)):
            sample_rng, sample_img = self.sample_jit(sample_rng, params, self.tokenizer_params)
            data.append(np.array(sample_img))

        data = np.concatenate(data, axis=0)
        if save_npz:
            create_npz_from_np('./test2', data)
            os.makedirs('assets',exist_ok=True)
            Image.fromarray(data[0]).save(f"assets/rar_generated_{1}.png")
        return data

    def sample_and_eval(self,params):
        # data=np.load('test2.npz')
        # generated_image = data['arr_0']
        # generated_image = self.sample(params, True)

        generated_image=self.sample(params,False)
        fid=self.computer_fid(generated_image)
        return fid



def main():
    # 初始化模型
    model_params,tokenizer_params,model,tokenizer_jax,rar_config = init_model()
    fid_model = inception.InceptionV3(pretrained=True)
    fid_model_params = fid_model.init(jax.random.PRNGKey(1), jnp.ones((1, 256, 256, 3)))
    sampler=Sampler(model,tokenizer_jax,tokenizer_params,rar_config,128,fid_model,fid_model_params)
    sampler.sample_and_eval(model_params)
    # sampler.sample(model_params)


if __name__ == "__main__":
    main()