# ------------------------------
# PyTorch -> Flax 权重转换函数
# ------------------------------
import flax


def convert_torch_to_flax_rar(state_dict):
    params = {}
    # 顶层参数
    params['cls_token'] = state_dict['cls_token']
    params['pos_embed'] = state_dict['pos_embed']
    params['target_aware_pos_embed'] = state_dict['target_aware_pos_embed']
    params['timesteps_embeddings'] = state_dict['timesteps_embeddings']

    # embeddings: torch key "embeddings.weight" → flax key "embeddings.embedding"
    params['embeddings.embedding'] = state_dict['embeddings.weight']

    # lm_head: 对 Dense 层的 weight 做转置
    params['lm_head.kernel'] = state_dict['lm_head.weight'].transpose(1, 0)
    params['lm_head.bias'] = state_dict['lm_head.bias']

    # 处理 blocks 模块：循环处理 blocks.0, blocks.1, ... 直到没有对应键
    layer_idx = 0
    while f"blocks.{layer_idx}.attn.qkv.bias" in state_dict:
        prefix = f"blocks.{layer_idx}"
        # LayerNorm norm1 → flax: norm1.scale / norm1.bias
        params[f'blocks_{layer_idx}.norm1.scale'] = state_dict[f'{prefix}.norm1.weight']
        params[f'blocks_{layer_idx}.norm1.bias'] = state_dict[f'{prefix}.norm1.bias']

        # attn.qkv: Dense 层，需要转置 weight
        params[f'blocks_{layer_idx}.attn.qkv.kernel'] = state_dict[f'{prefix}.attn.qkv.weight'].transpose(1, 0)
        params[f'blocks_{layer_idx}.attn.qkv.bias'] = state_dict[f'{prefix}.attn.qkv.bias']

        # attn.q_norm (LayerNorm)
        params[f'blocks_{layer_idx}.attn.q_norm.scale'] = state_dict[f'{prefix}.attn.q_norm.weight']
        params[f'blocks_{layer_idx}.attn.q_norm.bias'] = state_dict[f'{prefix}.attn.q_norm.bias']

        # attn.k_norm (LayerNorm)
        params[f'blocks_{layer_idx}.attn.k_norm.scale'] = state_dict[f'{prefix}.attn.k_norm.weight']
        params[f'blocks_{layer_idx}.attn.k_norm.bias'] = state_dict[f'{prefix}.attn.k_norm.bias']

        # attn.proj: Dense 层
        params[f'blocks_{layer_idx}.attn.proj.kernel'] = state_dict[f'{prefix}.attn.proj.weight'].transpose(1, 0)
        params[f'blocks_{layer_idx}.attn.proj.bias'] = state_dict[f'{prefix}.attn.proj.bias']

        # LayerNorm norm2
        params[f'blocks_{layer_idx}.norm2.scale'] = state_dict[f'{prefix}.norm2.weight']
        params[f'blocks_{layer_idx}.norm2.bias'] = state_dict[f'{prefix}.norm2.bias']

        # mlp.fc1: Dense 层
        params[f'blocks_{layer_idx}.mlp_fc1.kernel'] = state_dict[f'{prefix}.mlp.fc1.weight'].transpose(1, 0)
        params[f'blocks_{layer_idx}.mlp_fc1.bias'] = state_dict[f'{prefix}.mlp.fc1.bias']

        # mlp.fc2: Dense 层
        params[f'blocks_{layer_idx}.mlp_fc2.kernel'] = state_dict[f'{prefix}.mlp.fc2.weight'].transpose(1, 0)
        params[f'blocks_{layer_idx}.mlp_fc2.bias'] = state_dict[f'{prefix}.mlp.fc2.bias']

        # adaLN_modulation: Dense 层（这里命名为 layers_1）
        params[f'blocks_{layer_idx}.adaln_fc.kernel'] = state_dict[f'{prefix}.adaLN_modulation.1.weight'].transpose(1, 0)
        params[f'blocks_{layer_idx}.adaln_fc.bias'] = state_dict[f'{prefix}.adaLN_modulation.1.bias']

        layer_idx += 1

    # 处理 adaln_before_head 模块
    params['adaln_before_head.adaln_fc.kernel'] = state_dict['adaln_before_head.adaLN_modulation.1.weight'].transpose(1, 0)
    params['adaln_before_head.adaln_fc.bias'] = state_dict['adaln_before_head.adaLN_modulation.1.bias']

    # 将所有 torch.Tensor 转为 numpy 数组（若你使用的是 torch.Tensor 类型）
    params = {k: v.numpy() for k, v in params.items()}
    params = flax.traverse_util.unflatten_dict(params,sep='.')
    return params