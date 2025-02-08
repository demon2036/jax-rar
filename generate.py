import argparse
import math
import threading
from functools import partial
from pathlib import Path
import orbax.checkpoint as ocp
import tqdm
from flax.training import orbax_utils
from jax._src.mesh import Mesh
from jax.experimental import mesh_utils, multihost_utils

from pytorch_rar import demo_util
from huggingface_hub import hf_hub_download

from utils.generate_utils import CustomShardWriter, thread_write, send_file, get_remote_path
from models.maskgit_vqgan import PretrainedTokenizer
from generate_and_sample.sample import RARConfig, sample
from convert_model_torch_to_flax.convert_vqgan_torch_to_flax import convert_vqgan_state_dict
from models.rar import FlaxRAR, convert_torch_to_flax_rar, FlaxRARConfig
from pytorch_rar.utils.train_utils import create_pretrained_tokenizer

import jax
# jax.config.update('jax_platform_name', 'cpu')
from jax.sharding import PartitionSpec as P

from jax.experimental.shard_map import shard_map


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




def main(args):
    # 初始化模型
    model_params,tokenizer_params,model,tokenizer_jax,rar_config = init_model()

    # 构造随机输入：input_ids（整数 token）和 condition（假设为 [B, 1]）
    batch_size = args.batch_per_core
    rng = jax.random.PRNGKey(args.global_seed)
    sample_rng, dropout_rng = jax.random.split(rng)
    sample_rng=jax.random.split(sample_rng,jax.device_count())

    physical_mesh=mesh_utils.create_device_mesh((jax.device_count(),))
    mesh=Mesh(physical_mesh, ('dp',))

    sample_fn=partial(sample,model=model,config=rar_config,batch_size=batch_size,tokenizer_jax=tokenizer_jax)
    sample_fn=shard_map(sample_fn,mesh=mesh,in_specs=(
        P('dp'),P(None),P(None)
    ),
        out_specs=P('dp')
    )

    # 构造 rngs 字典
    sample_jit=jax.jit(sample_fn)


    data_per_shard = args.data_per_shard
    per_process_generate_data = args.batch_per_core * jax.local_device_count()
    assert data_per_shard % per_process_generate_data == 0
    iter_per_shard = data_per_shard // per_process_generate_data
    shard_dir_path = Path('shard_path')
    shard_dir_path.mkdir(exist_ok=True)
    shard_filename = str(shard_dir_path / 'shards-%05d.tar')


    dst = get_remote_path(args.output_dir)
    thread_writes=[]
    send_file_threads=[]
    keep_files=2
    ckpts=[]
    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

    iteration = math.ceil(args.num_samples / args.batch_per_core / jax.device_count())

    if args.resume:
        ckpt = {
            'rng': rng,
            'label': 1
        }

        try:
            ckpt = checkpointer.restore(f'{dst}/resume-0', item=ckpt)
        except Exception as e:
            ckpt = checkpointer.restore(f'{dst}/resume-1', item=ckpt)
            print(e)

        sample_rng = ckpt['rng']
        init_step = ckpt['label']
    else:
        init_step=0




    sink = CustomShardWriter(
        pattern=shard_filename,
        maxcount=data_per_shard,
        maxsize=3e10,
        start_shard=jax.process_index()+init_step*jax.process_count(),
        verbose=jax.process_index() == 0,
        progress_count=jax.process_count()
        # maxsize=shard_size,
    )


    iters=iteration
    print(init_step)
    # iters = 5
    for step in tqdm.tqdm(range(init_step, iters + 1), initial=init_step, total=iters + 1):
        for _ in range(iter_per_shard):
            sample_rng,sample_img,labels=sample_jit(sample_rng,model_params,tokenizer_params)
            for thread in thread_writes:
                thread.join()
            thread_writes=[]
            thread=threading.Thread(target=thread_write,args=(sample_img,labels,sink))
            thread.start()
            thread_writes.append(thread)


        for thread in send_file_threads:
            thread.join()

        if len(ckpts)<keep_files:
            pass
        else:
            ckpt,ckpts=ckpts[0],ckpts[1:]
            save_args = orbax_utils.save_args_from_target(ckpt)
            multihost_utils.sync_global_devices('sync device for save')
            checkpointer.save(f'{dst}/resume-{ckpt["label"]%2}', ckpt, save_args=save_args, force=True)

        ckpt,send_file_threads=send_file(keep_files,dst,sample_rng,step)
        ckpts.append(ckpt)


    for thread in thread_writes:
        thread.join()

    for thread in send_file_threads:
        thread.join()
    sink.close()

    ckpt=ckpts[-1]
    save_args = orbax_utils.save_args_from_target(ckpt)
    checkpointer.wait_until_finished()
    checkpointer.save(f'{dst}/resume.json', ckpt, save_args=save_args, force=True)
    checkpointer.wait_until_finished()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="shard_path2")
    # parser.add_argument("--output-dir", default="gs://shadow-center-2b/imagenet-generated-100steps-cfg1.75")
    # parser.add_argument("--output-dir", default="gs://musk-center-2b/imagenet-generated-sit-250steps-50m")
    # parser.add_argument("--seed", type=int, default=7)
    # parser.add_argument("--sample-seed", type=int, default=24)
    # parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--data-per-shard", type=int, default=8192)  #2048
    # parser.add_argument("--per-process-shards", type=int, default=400)
    # parser.add_argument("--per-device-batch", type=int, default=128)  #128
    parser.add_argument("--resume",  action="store_true", default=False)

    parser.add_argument("--global-seed", type=int, default=20360724)
    parser.add_argument("--batch-per-core", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=50000000)
    # jax.distributed.initialize()
    main(parser.parse_args())
    # main(parser.parse_args())