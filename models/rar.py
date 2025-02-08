"""
jax_rar.py

本文件给出基于 Flax 的 RAR 模型实现，同时包含从 PyTorch 权重转换的辅助代码和测试代码。
"""
import dataclasses
import math, numpy as np, jax, jax.numpy as jnp
from functools import partial
from typing import Any, Optional, Callable, Tuple
import flax.linen as nn
from flax.core import freeze, unfreeze
from flax.training import train_state
import optax
import flax


# dtype=jnp.bfloat16
dtype=jnp.float32

# ------------------------------
# 辅助函数
# ------------------------------

def modulate(x, shift, scale):
    """等价于 torch 版中 modulate(x, shift, scale)"""
    return x * (1 + scale) + shift

def build_causal_mask(seq_length: int) -> jnp.ndarray:
    # 构造上三角形 mask，注意这里“负无穷”在 softmax 前使用
    mask = -jnp.inf * jnp.ones((seq_length, seq_length))
    mask = jnp.triu(mask, k=1)
    return mask

# ------------------------------
# Flax 版本的 Attention 模块
# ------------------------------

class AttentionRAR(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    norm_layer: Callable[..., nn.Module] = nn.LayerNorm
    kv_cache: bool = False  # 是否使用 KV cache

    def setup(self):
        assert self.dim % self.num_heads == 0, "dim 必须能被 num_heads 整除"
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Dense(3 * self.dim, use_bias=self.qkv_bias, name="qkv",dtype=dtype)
        if self.qk_norm:
            # 分别对 q 和 k 做 layer norm
            self.q_norm = self.norm_layer(name="q_norm", epsilon=1e-6)
            self.k_norm = self.norm_layer(name="k_norm", epsilon=1e-6)
        else:
            self.q_norm = lambda x: x
            self.k_norm = lambda x: x

        self.proj = nn.Dense(self.dim, name="proj",dtype=dtype)
        self.proj_dropout = nn.Dropout(rate=self.proj_drop)

    def __call__(self, x: jnp.ndarray, attn_mask: Optional[jnp.ndarray] = None,
                 deterministic: bool = True,cache=None):
        """
        x: [B, N, C]
        attn_mask: [N, N] 或者 None
        """
        B, N, C = x.shape
        # [B, N, 3*dim] -> reshape 成 [B, N, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # 转置后得到 (3, B, num_heads, N, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q).astype(q.dtype)
        k = self.k_norm(k).astype(q.dtype)

        # 如果开启 KV cache，可以将过去的 k、v 拼接（这里只是简单示例，实际使用中需使用 mutable state）
        if cache is not None:
            end_index = cache['end_index'][0]

            slice_indices = (0, 0, end_index , 0)
            v = jax.lax.dynamic_update_slice(
                cache['v'],
                v,
                slice_indices,
            )
            k = jax.lax.dynamic_update_slice(
                cache['k'], k, slice_indices
            )
            new_cache = {
                'v': v,
                'k': k,
                'end_index': cache['end_index'] + N,
            }
        else:
            new_cache = None


        attn_weights = (q @ k.swapaxes(2, 3)) / math.sqrt(self.head_dim)

        if attn_mask is not None:  # no matter the length, we just slice it
            # causal_mask = attn_mask[:, :, :, : key_states.shape[-2]]
            # jax.debug.print("hello {bar}", bar=attn_mask,)
            causal_mask = attn_mask
            # print(causal_mask)
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.softmax(attn_weights.astype(jnp.float32), axis=-1, ).astype(attn_weights.dtype)
        attn = attn_weights @ v

        # attn shape: [B, num_heads, N, head_dim] → 转换回 [B, N, C]
        attn = jnp.transpose(attn, (0, 2, 1, 3)).reshape(B, N, C)
        x = self.proj(attn)
        x = self.proj_dropout(x, deterministic=deterministic)
        return x,new_cache

    @classmethod
    def init_cache(
            cls,
            cache_size: int,
            num_heads: int,
            head_dim: int,
            batch_size: int,
            dtype: jnp.dtype = jnp.float32,
    ):
        del cls  # not used
        return {
            'v': jnp.zeros(
                (batch_size, num_heads, cache_size, head_dim), dtype=dtype
            ),
            'k': jnp.zeros(
                (batch_size, num_heads, cache_size, head_dim), dtype=dtype
            ),
            'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
        }

# ------------------------------
# Flax 版本的 FinalLayer 模块
# ------------------------------

class FinalLayerRAR(nn.Module):
    dim: int
    norm_layer: Callable[..., nn.Module] = nn.LayerNorm

    def setup(self):
        # 注意 norm_final 不带 elementwise affine（与 torch 中 elementwise_affine=False 相对应）
        self.norm_final = self.norm_layer(epsilon=1e-6, use_scale=False, use_bias=False, name="norm_final")
        self.adaLN_modulation = nn.Sequential([
            nn.silu,
            nn.Dense(2 * self.dim, name="adaln_fc",dtype=dtype)
        ])

    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        # c 的 shape 为 [B, D] 或 [B, 1, D]，这里要求最后一维为 dim
        modulation = self.adaLN_modulation(c)
        # 与 torch 不同，此处直接 chunk 为 (scale, shift)
        scale, shift = jnp.split(modulation, 2, axis=-1)
        x = self.norm_final(x)
        x = modulate(x, shift, scale)
        return x

# ------------------------------
# Flax 版本的 Block 模块
# ------------------------------

class BlockRAR(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_norm: bool = True
    proj_drop: float = 0.0
    attn_drop: float = 0.0
    act_layer: Callable = nn.gelu
    norm_layer: Callable[..., nn.Module] = nn.LayerNorm

    def setup(self):
        self.norm1 = self.norm_layer(epsilon=1e-6, name="norm1")
        self.attn = AttentionRAR(
            dim=self.dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qk_norm=self.qk_norm,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            norm_layer=self.norm_layer,
            name="attn",
        )
        self.norm2 = self.norm_layer(epsilon=1e-6, name="norm2")
        hidden_features = int(self.dim * self.mlp_ratio)
        self.mlp = nn.Sequential([
            nn.Dense(hidden_features, name="mlp_fc1",dtype=dtype),
            self.act_layer,
            nn.Dense(self.dim, name="mlp_fc2",dtype=dtype),
        ])
        # 生成 6*dim 的 modulation 参数
        self.adaLN_modulation = nn.Sequential([
            nn.silu,
            nn.Dense(6 * self.dim, name="adaln_fc"),
        ])

    def __call__(self, x: jnp.ndarray, attn_mask: Optional[jnp.ndarray],
                 c: jnp.ndarray, deterministic: bool = True,layer_cache=None) :
        modulation = self.adaLN_modulation(c)
        # modulation 分成 6 份：shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        splits = jnp.split(modulation, 6, axis=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = splits

        # 注意：这里需要扩展 modulation参数的维度，使其与 x 匹配
        # x shape: [B, N, dim]，而 modulation shape: [B, dim] 或 [B,1,dim]
        def expand_mod(m):
            if m.ndim == 2:
                return m[:, None, :]
            return m
        shift_msa = expand_mod(shift_msa)
        scale_msa = expand_mod(scale_msa)
        gate_msa  = expand_mod(gate_msa)
        shift_mlp = expand_mod(shift_mlp)
        scale_mlp = expand_mod(scale_mlp)
        gate_mlp  = expand_mod(gate_mlp)

        # 第一部分：Attention
        x_norm1 = self.norm1(x)
        x_modulated1 = modulate(x_norm1, shift_msa, scale_msa)
        attn_out,layer_cache = self.attn(x_modulated1, attn_mask=attn_mask, deterministic=deterministic,cache=layer_cache)
        x = x + gate_msa * attn_out

        # 第二部分：MLP
        x_norm2 = self.norm2(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_modulated2)
        x = x + gate_mlp * mlp_out

        return x,layer_cache

# ------------------------------
# Flax 版本的 RAR 模型
# ------------------------------





"""



vq_model:
    codebook_size: 1024
    token_size: 256
    num_latent_tokens: 256
    finetune_decoder: False
    pretrained_tokenizer_weight: "maskgit-vqgan-imagenet-f16-256.bin"

generator:
        hidden_size: 768
        num_hidden_layers: 24
        num_attention_heads: 16
        intermediate_size: 3072
        dropout: 0.1
        attn_drop: 0.1
        class_label_dropout: 0.1
        image_seq_len: 256
        condition_num_classes: 1000



embed_dim = self.config["models"]["generator"]["hidden_size"]
        depth = self.config["models"]["generator"]["num_hidden_layers"]
        num_heads = self.config["models"]["generator"]["num_attention_heads"]
        intermediate_size = self.config["models"]["generator"]["intermediate_size"]
        mlp_ratio = intermediate_size / embed_dim

        image_seq_len = self.config["models"]["generator"]["image_seq_len"]
        target_codebook_size = self.config["models"]["vq_model"]["codebook_size"]
        condition_num_classes = self.config["models"]["generator"]["condition_num_classes"]

        dropout_rate = self.config["models"]["generator"]["dropout"]
        attn_dropout_rate = self.config["models"]["generator"]["attn_drop"]
  self.random_ratio = self.config["models"]["generator"].get("random_ratio", 0.0)



"""



@dataclasses.dataclass
class FlaxRARConfig:
    def __init__(self,
                 embed_dim,
                 depth,
                 intermediate_size,
                 num_heads=16,
                 image_seq_len=256,
                 target_codebook_size=1024,
                 condition_num_classes=1000,
                 dropout_rate=0.0,
                 attn_dropout_rate=0.0,
                 random_ratio=0.0,
    ):
        self.embed_dim=embed_dim
        self.depth=depth
        self.num_heads=num_heads
        self.intermediate_size=intermediate_size
        self.mlp_ratio=intermediate_size / embed_dim
        self.image_seq_len=image_seq_len
        self.target_codebook_size=target_codebook_size
        self.condition_num_classes=condition_num_classes
        self.dropout_rate=dropout_rate
        self.attn_dropout_rate=attn_dropout_rate
        self.random_ratio=random_ratio





class FlaxRAR(nn.Module):
    # config: Any  # 配置字典
    config: FlaxRARConfig  # 配置字典
    dtype:Any = dtype

    # 为简化起见，下面将一些参数展开为属性
    def setup(self):
        # 从 config 中解析参数
        embed_dim = self.config.embed_dim
        depth = self.config.depth
        num_heads = self.config.num_heads
        intermediate_size = self.config.intermediate_size
        mlp_ratio = intermediate_size / embed_dim

        image_seq_len = self.config.image_seq_len
        target_codebook_size = self.config.target_codebook_size
        condition_num_classes = self.config.condition_num_classes

        dropout_rate = self.config.dropout_rate
        attn_dropout_rate = self.config.attn_dropout_rate

        self.embed_dim = embed_dim
        self.image_seq_len = image_seq_len
        self.target_codebook_size = target_codebook_size
        self.condition_num_classes = condition_num_classes
        self.none_condition_id = condition_num_classes + target_codebook_size + 1

        # cls token：shape [1, 1, embed_dim]
        self.cls_token = self.param("cls_token", nn.initializers.zeros, (1, 1, embed_dim))
        # 构造 blocks，使用 nn.scan 或直接列表（这里直接用列表）
        self.blocks = [BlockRAR(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            qk_norm=True,
            proj_drop=dropout_rate,
            attn_drop=attn_dropout_rate,
            norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
            name=f"blocks_{i}",
        ) for i in range(depth)]
        # embeddings：token 数 embedding（target_codebook_size + 1 + condition_num_classes + 1）
        vocab_size = target_codebook_size + 1 + condition_num_classes + 1
        self.embeddings = nn.Embed(num_embeddings=vocab_size, features=embed_dim, name="embeddings",dtype=dtype)

        # 位置编码：假设 shape 为 [1, image_seq_len + 1024, embed_dim]
        pos_embed_shape = (1, image_seq_len + 1024, embed_dim)
        self.pos_embed = self.param("pos_embed",
                                    lambda key, shape: jax.random.truncated_normal(key, -2, 2, shape) * 0.02,
                                    pos_embed_shape)
        self.target_aware_pos_embed = self.param("target_aware_pos_embed",
                                                 lambda key, shape: jax.random.truncated_normal(key, -2, 2, shape) * 0.02,
                                                 pos_embed_shape)
        # timesteps embeddings: shape [1, image_seq_len + 100, embed_dim]
        timesteps_shape = (1, image_seq_len + 100, embed_dim)
        self.timesteps_embeddings = self.param("timesteps_embeddings",
                                               lambda key, shape: jax.random.truncated_normal(key, -2, 2, shape) * 0.02,
                                               timesteps_shape)
        # adaln_before_head
        self.adaln_before_head = FinalLayerRAR(dim=embed_dim, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), name="adaln_before_head")
        # lm_head: 输出 target_codebook_size 个类别
        self.lm_head = nn.Dense(target_codebook_size, name="lm_head")
        # attn mask （静态）：形状 [seq_len, seq_len]
        total_seq_len = image_seq_len + 1024
        self.attn_mask = build_causal_mask(total_seq_len)
        # 随机采样比例
        # self.random_ratio = self.config["models"]["generator"].get("random_ratio", 0.0)

        self.random_ratio = self.config.random_ratio

        # use_checkpoint 在 JAX 中可以使用 remat 包裹 Block，但这里暂略

    def sample_orders(self, x: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
        """
        对于每个 batch，根据 random_ratio 决定是否随机排列
        x: [B, ...]
        返回 orders: [B, image_seq_len]
        """
        B = x.shape[0]
        def sample_order(key):
            # 如果随机，则返回随机排列，否则返回 raster 顺序
            r = jax.random.uniform(key, ())
            order = jnp.arange(self.image_seq_len)
            order = jax.lax.cond(r < self.random_ratio,
                                 lambda _: jax.random.permutation(key, order),
                                 lambda _: order,
                                 operand=None)
            return order
        keys = jax.random.split(rng, B)
        orders = jax.vmap(sample_order)(keys)
        return orders

    def shuffle(self, x: jnp.ndarray, orders: jnp.ndarray) -> jnp.ndarray:
        """
        x: [B, seq_len, ...]，orders: [B, seq_len]
        返回按 orders 重新排列后的 x
        """

        batch_size, seq_len = x.shape[:2]
        batch_indices = jnp.arange(0,batch_size)[...,None]
        shuffled_x = x[batch_indices, orders]


        return shuffled_x

    def unshuffle(self, shuffled_x: jnp.ndarray, orders: jnp.ndarray) -> jnp.ndarray:
        """
        将 shuffle 后的结果还原
        """
        B, L = orders.shape
        # 构造逆序 index
        def _unshuffle(x_i, order_i):
            inv = jnp.argsort(order_i)
            return x_i[inv]
        return jax.vmap(_unshuffle)(shuffled_x, orders)

    def preprocess_condition(self, condition: jnp.ndarray, rng: jax.random.PRNGKey, cond_drop_prob: float = 0.0) -> jnp.ndarray:
        """
        condition: [B, ...]，假设 dtype 为 int32
        将 condition 加上偏移，并以概率 cond_drop_prob 将其置为 none_condition_id
        """
        # 使用 jax.random.uniform 生成 mask
        B = condition.shape[0]
        drop = jax.random.uniform(rng, shape=condition.shape) < cond_drop_prob
        condition = condition + self.target_codebook_size + 1
        condition = jnp.where(drop, self.none_condition_id, condition)
        return condition

    def get_none_condition(self, condition: jnp.ndarray) -> jnp.ndarray:
        return jnp.full_like(condition, self.none_condition_id)

    def __call__(self, input_ids: jnp.ndarray, condition: jnp.ndarray,
                 rngs: dict,  # 要求传入 {"dropout": dropout_key, "sample": sample_key}
                 return_labels: bool = False,
                 orders: Optional[jnp.ndarray] = None,
                 is_sampling: bool = True) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        类似于 torch 版的 forward_fn
        input_ids: [B, image_seq_len]，整数 token id
        condition: [B, ?]（这里假设是 [B, 1] 或 [B, num_condition]）
        rngs 中至少包含 "dropout" 和 "sample"
        """
        # 如果 orders 未给出，则生成 raster 顺序
        B = input_ids.shape[0]
        if orders is None:
            orders = jnp.tile(jnp.arange(self.image_seq_len)[None, :], (B, 1))

        # 复制一份 labels
        labels = input_ids.copy()

        # 将 condition 展开为 [B, condition_tokens] 并拼接到 input_ids 前
        # 此处假设 condition 已经是合适 shape
        input_ids = jnp.concatenate([condition.reshape(B, -1), input_ids.reshape(B, -1)], axis=1)

        # 获取 token embedding
        embeddings = self.embeddings(input_ids)  # [B, seq_len, embed_dim]
        # condition_token 就取 embeddings 的第一 token（与 torch 版一致）
        condition_token = embeddings[:, 0]  # [B, embed_dim]

        # 位置编码：先复制 pos_embed 到 batch 上
        pos_embed = jnp.tile(self.pos_embed, (B, 1, 1))
        # 前 prefix 个 token 不 shuffle，假设 prefix = 2
        prefix = 2
        pos_embed_prefix = pos_embed[:, :prefix]
        # 对后面的 image_seq_len 部分根据 orders 做 shuffle
        pos_embed_postfix = self.shuffle(pos_embed[:, prefix: prefix + self.image_seq_len], orders)
        pos_embed_combined = jnp.concatenate([pos_embed_prefix, pos_embed_postfix], axis=1)
        # 同理 target aware pos embed
        target_aware_pos_embed_postfix = self.shuffle(self.target_aware_pos_embed[:, prefix: prefix + self.image_seq_len], orders)

        # 如果非采样阶段，则也对 embeddings 做 shuffle（除了第一个 token）
        if not is_sampling:
            emb_prefix = embeddings[:, :1]
            emb_postfix = self.shuffle(embeddings[:, 1:], orders)
            embeddings = jnp.concatenate([emb_prefix, emb_postfix], axis=1)

        # 拼接 cls token：将 cls_token 展开到 batch 上，然后拼接到 embeddings 前
        cls_tokens = jnp.tile(self.cls_token, (B, 1, 1))
        x = jnp.concatenate([cls_tokens, embeddings], axis=1)

        # 加上位置编码（取前 x.shape[1] 个 token 的 pos_embed）
        x = x + pos_embed_combined[:, :x.shape[1], :]

        # 加上 target-aware pos embed
        # 这里构造一个 target aware pos embed：前 prefix-1 全零，后面 target aware 部分，再后面零补
        zeros_prefix = jnp.zeros_like(x[:, :prefix-1, :])
        zeros_suffix = jnp.zeros_like(x[:, -1:, :])
        target_aware = jnp.concatenate([zeros_prefix, target_aware_pos_embed_postfix, zeros_suffix], axis=1)
        x = x + target_aware[:, :x.shape[1], :]

        # 生成 attn_mask（注意：在 Flax 中，一般要求 bias shape 为 [1, 1, seq_len, seq_len]）
        attn_mask = self.attn_mask[:x.shape[1], :x.shape[1]]
        attn_mask = attn_mask[None, None, :, :]  # broadcast 到 [B, num_heads, seq_len, seq_len]

        # 对 condition_token 加上 timesteps_embeddings（取前 x.shape[1] 个）
        condition_token = condition_token[:, None, :] + self.timesteps_embeddings[:, :x.shape[1], :]

        # 如果开启 KV cache（例如在采样时），可能只处理最后一个 token（这里略）
        # 依次通过 blocks
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, c=condition_token, deterministic=True#not ("dropout" in rngs)
                      )
        # 如果没有 KV cache，则去除 cls token，假设 prefix-1
        x = x[:, prefix - 1:, :]
        condition_token = condition_token[:, prefix - 1:, :]

        # adaln_before_head
        x = self.adaln_before_head(x, condition_token)
        logits = self.lm_head(x)
        if return_labels:
            return logits, labels
        return logits #, None











    def train_dpo(self, input_ids: jnp.ndarray, condition: jnp.ndarray,
                 rngs: dict=None,  # 要求传入 {"dropout": dropout_key, "sample": sample_key}
                 return_labels: bool = False,
                 orders: Optional[jnp.ndarray] = None,
                 is_sampling: bool = True) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        类似于 torch 版的 forward_fn
        input_ids: [B, image_seq_len]，整数 token id
        condition: [B, ?]（这里假设是 [B, 1] 或 [B, num_condition]）
        rngs 中至少包含 "dropout" 和 "sample"
        """
        # 如果 orders 未给出，则生成 raster 顺序
        B = input_ids.shape[0]
        if orders is None:
            orders = jnp.tile(jnp.arange(self.image_seq_len)[None, :], (B, 1))

        # 复制一份 labels
        labels = input_ids.copy()




        # 将 condition 展开为 [B, condition_tokens] 并拼接到 input_ids 前
        # 此处假设 condition 已经是合适 shape
        input_ids = jnp.concatenate([condition.reshape(B, -1), input_ids.reshape(B, -1)], axis=1)




        # 获取 token embedding
        embeddings = self.embeddings(input_ids)  # [B, seq_len, embed_dim]
        # condition_token 就取 embeddings 的第一 token（与 torch 版一致）
        condition_token = embeddings[:, 0]  # [B, embed_dim]

        # 位置编码：先复制 pos_embed 到 batch 上
        pos_embed = jnp.tile(self.pos_embed, (B, 1, 1))
        # 前 prefix 个 token 不 shuffle，假设 prefix = 2
        prefix = 2
        pos_embed_prefix = pos_embed[:, :prefix]
        # 对后面的 image_seq_len 部分根据 orders 做 shuffle
        pos_embed_postfix = self.shuffle(pos_embed[:, prefix: prefix + self.image_seq_len], orders)
        pos_embed_combined = jnp.concatenate([pos_embed_prefix, pos_embed_postfix], axis=1)
        # 同理 target aware pos embed
        target_aware_pos_embed_postfix = self.shuffle(self.target_aware_pos_embed[:, prefix: prefix + self.image_seq_len], orders)

        # 如果非采样阶段，则也对 embeddings 做 shuffle（除了第一个 token）
        if not is_sampling:
            emb_prefix = embeddings[:, :1]
            emb_postfix = self.shuffle(embeddings[:, 1:], orders)
            embeddings = jnp.concatenate([emb_prefix, emb_postfix], axis=1)

        # 拼接 cls token：将 cls_token 展开到 batch 上，然后拼接到 embeddings 前
        cls_tokens = jnp.tile(self.cls_token, (B, 1, 1))
        x = jnp.concatenate([cls_tokens, embeddings], axis=1)

        # 加上位置编码（取前 x.shape[1] 个 token 的 pos_embed）


        x = x + pos_embed_combined[:, :x.shape[1], :]

        # 加上 target-aware pos embed
        # 这里构造一个 target aware pos embed：前 prefix-1 全零，后面 target aware 部分，再后面零补
        zeros_prefix = jnp.zeros_like(x[:, :prefix-1, :])
        zeros_suffix = jnp.zeros_like(x[:, -1:, :])
        target_aware = jnp.concatenate([zeros_prefix, target_aware_pos_embed_postfix, zeros_suffix], axis=1)
        x = x + target_aware[:, :x.shape[1], :]

        # 生成 attn_mask（注意：在 Flax 中，一般要求 bias shape 为 [1, 1, seq_len, seq_len]）
        attn_mask = self.attn_mask[:x.shape[1], :x.shape[1]]
        attn_mask = attn_mask[None, None, :, :]  # broadcast 到 [B, num_heads, seq_len, seq_len]

        # 对 condition_token 加上 timesteps_embeddings（取前 x.shape[1] 个）
        condition_token = condition_token[:, None, :] + self.timesteps_embeddings[:, :x.shape[1], :]

        # 如果开启 KV cache（例如在采样时），可能只处理最后一个 token（这里略）
        # 依次通过 blocks
        for block in self.blocks:
            x,layer_cache = block(x, attn_mask=attn_mask, c=condition_token, deterministic=True#not ("dropout" in rngs)
                      )
        # 如果没有 KV cache，则去除 cls token，假设 prefix-1
        x = x[:, prefix - 1:, :]
        condition_token = condition_token[:, prefix - 1:, :]

        # adaln_before_head
        x = self.adaln_before_head(x, condition_token)
        logits = self.lm_head(x)
        if return_labels:
            return logits, labels
        return logits #, None










    def prefill(self, condition: jnp.ndarray,cache=None,attn_mask=None) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        类似于 torch 版的 forward_fn
        input_ids: [B, image_seq_len]，整数 token id
        condition: [B, ?]（这里假设是 [B, 1] 或 [B, num_condition]）
        rngs 中至少包含 "dropout" 和 "sample"
        """
        B,*_=condition.shape
        # 将 condition 展开为 [B, condition_tokens] 并拼接到 input_ids 前
        # 此处假设 condition 已经是合适 shape
        # 如果 orders 未给出，则生成 raster 顺序
        orders = jnp.tile(jnp.arange(self.image_seq_len)[None, :], (B, 1))

        input_ids=condition

        # 获取 token embedding
        embeddings = self.embeddings(input_ids)  # [B, seq_len, embed_dim]
        # condition_token 就取 embeddings 的第一 token（与 torch 版一致）
        condition_token = embeddings[:, 0]  # [B, embed_dim]

        # 位置编码：先复制 pos_embed 到 batch 上
        pos_embed = jnp.tile(self.pos_embed, (B, 1, 1))
        # 前 prefix 个 token 不 shuffle，假设 prefix = 2
        prefix = 2
        # pos_embed_prefix = pos_embed[:, :prefix]
        # # 对后面的 image_seq_len 部分根据 orders 做 shuffle
        # pos_embed_postfix = self.shuffle(pos_embed[:, prefix: prefix + self.image_seq_len], orders)
        # pos_embed_combined = jnp.concatenate([pos_embed_prefix, pos_embed_postfix], axis=1)

        pos_embed_combined=pos_embed

        # 同理 target aware pos embed
        target_aware_pos_embed_postfix = self.shuffle(
            self.target_aware_pos_embed[:, prefix: prefix + self.image_seq_len], orders)

        # target_aware_pos_embed_postfix=self.target_aware_pos_embed[:, prefix: prefix + self.image_seq_len]

        # 如果非采样阶段，则也对 embeddings 做 shuffle（除了第一个 token）
        # if not is_sampling:
        #     emb_prefix = embeddings[:, :1]
        #     emb_postfix = self.shuffle(embeddings[:, 1:], orders)
        #     embeddings = jnp.concatenate([emb_prefix, emb_postfix], axis=1)

        # 拼接 cls token：将 cls_token 展开到 batch 上，然后拼接到 embeddings 前
        cls_tokens = jnp.tile(self.cls_token, (B, 1, 1))
        x = jnp.concatenate([cls_tokens, embeddings], axis=1)

        # 加上位置编码（取前 x.shape[1] 个 token 的 pos_embed）
        x = x + pos_embed_combined[:, :x.shape[1], :]

        # 加上 target-aware pos embed
        # 这里构造一个 target aware pos embed：前 prefix-1 全零，后面 target aware 部分，再后面零补
        zeros_prefix = jnp.zeros_like(x[:, :prefix - 1, :])
        target_aware = jnp.concatenate([zeros_prefix, target_aware_pos_embed_postfix], axis=1)

        x = x + target_aware[:, :x.shape[1], :]
        b, n, d = x.shape
        if n > 1:
            attn_mask = jnp.full(
                (n, attn_mask.shape[1]), float("-inf")
            )
            attn_mask = jnp.triu(attn_mask, 1)[None, None, ...]
        else:
            raise NotImplementedError()
            # attention_mask = jnp.where(attention_mask, 0, float("-inf"))

        # 对 condition_token 加上 timesteps_embeddings（取前 x.shape[1] 个）
        condition_token = condition_token[:, None, :] + self.timesteps_embeddings[:, :x.shape[1], :]


        # 如果开启 KV cache（例如在采样时），可能只处理最后一个 token（这里略）
        # 依次通过 blocks
        for i,block in enumerate(self.blocks):
            layer_name = f'layer_{i}'
            layer_cache = cache[layer_name] if cache else None

            x,layer_cache = block(x, attn_mask=attn_mask, c=condition_token,
                      deterministic=True,layer_cache=layer_cache
                      )
            if cache is not None:
                cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

        # 如果没有 KV cache，则去除 cls token，假设 prefix-1
        x = x[:, prefix - 1:, :]
        condition_token = condition_token[:, prefix - 1:, :]

        # adaln_before_head
        x = self.adaln_before_head(x, condition_token)
        logits = self.lm_head(x)

        return logits , cache



    def decode(self, input_ids: jnp.ndarray,condition, position_ids,cache,attn_mask=None) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
               类似于 torch 版的 forward_fn
               input_ids: [B, image_seq_len]，整数 token id
               condition: [B, ?]（这里假设是 [B, 1] 或 [B, num_condition]）
               rngs 中至少包含 "dropout" 和 "sample"
               """

        attn_mask = jnp.where(attn_mask, 0, float("-inf"))

        B, *_ = input_ids.shape
        # 将 condition 展开为 [B, condition_tokens] 并拼接到 input_ids 前
        # 此处假设 condition 已经是合适 shape
        # 如果 orders 未给出，则生成 raster 顺序

        # 获取 token embedding
        embeddings = self.embeddings(input_ids)  # [B, seq_len, embed_dim]
        # condition_token 就取 embeddings 的第一 token（与 torch 版一致）
        condition_token = self.embeddings(condition)  # [B, embed_dim]

        # 位置编码：先复制 pos_embed 到 batch 上
        pos_embed = jnp.tile(self.pos_embed, (B, 1, 1)) # [B, seq_len, embed_dim]
        # 前 prefix 个 token 不 shuffle，假设 prefix = 2
        prefix = 2

        pos_embed_combined = jnp.take_along_axis(pos_embed,prefix+position_ids[...,None],axis=1)
        # target_aware_pos_embed_postfix = self.target_aware_pos_embed[:, prefix: prefix + self.image_seq_len]
        target_aware_pos_embed_postfix =jnp.take_along_axis(self.target_aware_pos_embed,prefix-1+position_ids[...,None],axis=1)

        x=embeddings

        # 加上位置编码（取前 x.shape[1] 个 token 的 pos_embed）
        x = x + pos_embed_combined[:, :x.shape[1], :]
        x=x+jnp.take_along_axis(self.target_aware_pos_embed,prefix+1+position_ids[...,None],axis=1)
        condition_token = condition_token + jnp.take_along_axis(self.timesteps_embeddings,prefix+position_ids[...,None],axis=1)

        # 如果开启 KV cache（例如在采样时），可能只处理最后一个 token（这里略）
        # 依次通过 blocks
        for i,block in enumerate(self.blocks):
            layer_name = f'layer_{i}'
            layer_cache = cache[layer_name] if cache else None

            x,layer_cache = block(x, attn_mask=attn_mask, c=condition_token,
                      deterministic=True,layer_cache=layer_cache
                      )

            cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch
        # 如果没有 KV cache，则去除 cls token，假设 prefix-1
        # condition_token = condition_token[:, prefix - 1:, :]

        # print(condition_token.shape)
        # while True:
        #     pass

        # adaln_before_head
        x = self.adaln_before_head(x, condition_token)
        logits = self.lm_head(x)

        return logits   , cache







    # generate() 方法可按照 torch 版逻辑写，但为了简化这里只给出 forward 部分

# ------------------------------
# PyTorch -> Flax 权重转换函数
# ------------------------------

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






def init_cache(
        config,
        batch_size: int,
        max_cache_length:int=258,
        dtype: jnp.dtype = jnp.bfloat16,
):

    """Initializes a new Transformer cache."""
    config.head_dim = config.hidden_size // config.num_attention_heads

    print(batch_size, config.num_key_value_heads, max_cache_length, config.head_dim)

    cache = {
        f'layer_{i}': AttentionRAR.init_cache(
            max_cache_length,
            config.num_key_value_heads,
            config.head_dim,
            batch_size,
            dtype,
        )
        for i in range(config.num_hidden_layers)
    }
    return cache






# ------------------------------
# 测试代码
# ------------------------------

def main():
    # 构造一个简单的 config 字典
    config = {
        "models": {
            "generator": {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "image_seq_len": 196,
                "condition_num_classes": 1000,
                "dropout": 0.1,
                "attn_drop": 0.1,
                "random_ratio": 0.0,
            },
            "vq_model": {
                "codebook_size": 1024,
            },
        }
    }

    # 初始化模型
    model = FlaxRAR(config=config)

    # 构造随机输入：input_ids（整数 token）和 condition（假设为 [B, 1]）
    batch_size = 2
    image_seq_len = config["models"]["generator"]["image_seq_len"]
    rng = jax.random.PRNGKey(0)
    sample_rng, dropout_rng = jax.random.split(rng)
    input_ids = jax.random.randint(sample_rng, (batch_size, 0), 0, config["models"]["vq_model"]["codebook_size"])
    condition = jax.random.randint(sample_rng, (batch_size, 1), 0, config["models"]["generator"]["condition_num_classes"])

    # 构造 rngs 字典
    rngs = {"dropout": dropout_rng, "sample": sample_rng,'params':sample_rng}

    # 初始化参数
    variables = model.init(rngs, input_ids, condition, rngs)
    params = variables["params"]

    # 前向一次
    logits, labels = jax.jit(model.apply)(variables, input_ids, condition, rngs)
    print("Logits shape:", logits.shape)
    if labels is not None:
        print("Labels shape:", labels.shape)

if __name__ == "__main__":
    main()
