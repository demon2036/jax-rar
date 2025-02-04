import functools
import math
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

# =============================================================================
# 辅助层：Conv2dSame
# =============================================================================
# Flax 中 nn.Conv 可直接指定 padding="SAME"
# class Conv2dSame(nn.Conv):
#     def __init__(
#         self,
#         features: int,
#         kernel_size: Sequence[int],
#         strides: Sequence[int] = (1, 1),
#         dilation: Sequence[int] = (1, 1),
#         use_bias: bool = False,
#         **kwargs,
#     ):
#         super().__init__(
#             features=features,
#             kernel_size=kernel_size,
#             strides=strides,
#             padding="SAME",
#             dilation=dilation,
#             use_bias=use_bias,
#             **kwargs,
#         )


Conv2dSame=functools.partial(nn.Conv,use_bias=False)

# =============================================================================
# 残差块，与 PyTorch 版本 ResnetBlock 对应
# =============================================================================
class ResnetBlock(nn.Module):
    in_channels: int
    out_channels: Optional[int] = None  # 若为 None，则与 in_channels 相同
    dropout: float = 0.0
    norm_num_groups: int = 32
    activation: Callable = nn.silu  # 使用 silu 激活，与 PyTorch F.silu 对应

    @nn.compact
    def __call__(self, hidden_states, *, deterministic: bool = True):
        out_channels = self.in_channels if self.out_channels is None else self.out_channels


        # 第一子层：GroupNorm -> silu -> 3x3 Conv
        h = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-6,name='norm1')(hidden_states)
        h = self.activation(h)
        h = Conv2dSame(features=out_channels, kernel_size=(3, 3), use_bias=False,name='conv1')(h)

        # 第二子层：GroupNorm -> silu -> Dropout -> 3x3 Conv
        h = nn.GroupNorm(num_groups=self.norm_num_groups, epsilon=1e-6,name='norm2')(h)
        h = self.activation(h)
        h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
        h = Conv2dSame(features=out_channels, kernel_size=(3, 3), use_bias=False,name='conv2')(h)


        # 如果输入输出通道不匹配，采用 1x1 Conv 调整（对应 PyTorch 中的 nin_shortcut）
        if self.in_channels != out_channels:
            # print(f'{hidden_states.shape=} {h.shape=}')
            # hidden_states = Conv2dSame(features=out_channels, kernel_size=(1, 1), use_bias=False,name='nin_shortcut')(hidden_states)
            hidden_states = Conv2dSame(features=out_channels, kernel_size=(1, 1), use_bias=False, name='nin_shortcut')(
                h)
        return h + hidden_states


def avg_pool2d_nhwc(x, kernel_size=2, stride=2, padding="SAME"):
    # 输入 x shape 为 (B, H, W, C)
    window_shape = (1, kernel_size, kernel_size, 1)
    strides = (1, stride, stride, 1)
    pooled = jax.lax.reduce_window(x, 0.0, jax.lax.add, window_shape, strides, padding)
    window_size = kernel_size * kernel_size
    return pooled / window_size

def tensorflow_style_avg_pooling(x, window_shape, strides, padding: str):
  """Avg pooling as done by TF (Flax layer gives different results).

  To be specific, Flax includes padding cells when taking the average,
  while TF does not.

  Args:
    x: Input tensor
    window_shape: Shape of pooling window; if 1-dim tuple is just 1d pooling, if
      2-dim tuple one gets 2d pooling.
    strides: Must have the same dimension as the window_shape.
    padding: Either 'SAME' or 'VALID' to indicate pooling method.

  Returns:
    pooled: Tensor after applying pooling.
  """
  pool_sum = jax.lax.reduce_window(x, 0.0, jax.lax.add,
                                   (1,) + window_shape + (1,),
                                   (1,) + strides + (1,), padding)
  pool_denom = jax.lax.reduce_window(
      jnp.ones_like(x), 0.0, jax.lax.add, (1,) + window_shape + (1,),
      (1,) + strides + (1,), padding)
  return pool_sum / pool_denom

# =============================================================================
# 下采样块，与 PyTorch 版本 DownsamplingBlock 对应
# =============================================================================
class DownsamplingBlock(nn.Module):
    config: Any  # 配置对象，应包含字段：channel_mult, hidden_channels, num_res_blocks, num_resolutions
    block_idx: int

    @nn.compact
    def __call__(self, hidden_states, *, deterministic: bool = True):
        # in_channel_mult = (1,) + config.channel_mult
        in_channel_mult = [1] + list(self.config.channel_mult)
        block_in = self.config.hidden_channels * in_channel_mult[self.block_idx]
        block_out = self.config.hidden_channels * self.config.channel_mult[self.block_idx]

        # 残差块
        for _ in range(self.config.num_res_blocks):
            hidden_states = ResnetBlock(
                in_channels=block_in,
                out_channels=block_out,
                dropout=self.config.dropout,
                norm_num_groups=32,
                activation=nn.silu,
            )(hidden_states, deterministic=deterministic)
            block_in = block_out  # 后续残差块输入通道已更新

        # 非最后一层进行下采样（平均池化 kernel=2, stride=2）
        if self.block_idx != self.config.num_resolutions - 1:
            # print(hidden_states.shape)
            # Flax 的 nn.avg_pool 要求输入为 NHWC，此处假定输入为 NCHW，因此先转置后池化，再转回
            # hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 1))
            # hidden_states = nn.avg_pool(hidden_states, window_shape=(2, 2), strides=(2, 2), padding="SAME")
            hidden_states=avg_pool2d_nhwc(hidden_states,2,2,'same')

            # hidden_states = jnp.transpose(hidden_states, (0, 3, 1, 2))
        return hidden_states

# =============================================================================
# 上采样块，与 PyTorch 版本 UpsamplingBlock 对应
# =============================================================================
class UpsamplingBlock(nn.Module):
    config: Any  # 配置对象
    block_idx: int

    @nn.compact
    def __call__(self, hidden_states, *, deterministic: bool = True):
        num_resolutions = self.config.num_resolutions
        if self.block_idx == num_resolutions - 1:
            block_in = self.config.hidden_channels * self.config.channel_mult[-1]
        else:
            block_in = self.config.hidden_channels * self.config.channel_mult[self.block_idx + 1]
        block_out = self.config.hidden_channels * self.config.channel_mult[self.block_idx]

        # 残差块
        for _ in range(self.config.num_res_blocks):
            hidden_states = ResnetBlock(
                in_channels=block_in,
                out_channels=block_out,
                dropout=self.config.dropout,
                norm_num_groups=32,
                activation=nn.silu,
            )(hidden_states, deterministic=deterministic)
            block_in = block_out

        # 非第一层进行上采样
        if self.block_idx != 0:
        # if self.block_idx != self.config.num_resolutions - 1:
            # 转为 NHWC 后上采样，再转回 NCHW

            # hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 1))
            n, h, w, c = hidden_states.shape
            hidden_states = jax.image.resize(hidden_states, shape=(n, h * 2, w * 2, c), method="nearest")
            # hidden_states = jnp.transpose(hidden_states, (0, 3, 1, 2))
            hidden_states = Conv2dSame(features=block_out, kernel_size=(3, 3), use_bias=True,name='upsample_conv')(hidden_states)
        return hidden_states

# =============================================================================
# Encoder，与 PyTorch 版本 Encoder 对应
# =============================================================================
class Encoder(nn.Module):
    config: Any  # 配置对象，应包含：num_channels, hidden_channels, channel_mult, num_res_blocks, num_resolutions, dropout, resolution, z_channels

    @nn.compact
    def __call__(self, pixel_values, *, deterministic: bool = True):
        # 下采样：输入 (B, C, H, W)
        hidden_states = Conv2dSame(features=self.config.hidden_channels, kernel_size=(3, 3), use_bias=False)(pixel_values)

        # print(hidden_states.shape,self.config.num_resolutions)

        for i_level in range(self.config.num_resolutions):
            # print('hi')

            hidden_states = DownsamplingBlock(config=self.config, block_idx=i_level)(hidden_states, deterministic=deterministic)

        # 中间层：若干残差块（最低分辨率）
        mid_channels = self.config.hidden_channels * self.config.channel_mult[-1]
        for _ in range(self.config.num_res_blocks):
            hidden_states = ResnetBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                dropout=self.config.dropout,
                norm_num_groups=32,
                activation=nn.silu,
            )(hidden_states, deterministic=deterministic)

        # 结束部分：GroupNorm -> silu -> 1x1 Conv 得到 z_channels
        hidden_states = nn.GroupNorm(num_groups=32, epsilon=1e-6)(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = Conv2dSame(features=self.config.z_channels, kernel_size=(1, 1),use_bias=True)(hidden_states)
        return hidden_states

# =============================================================================
# Decoder，与 PyTorch 版本 Decoder 对应
# =============================================================================
class Decoder(nn.Module):
    config: Any  # 配置对象

    @nn.compact
    def __call__(self, hidden_states, *, deterministic: bool = True):
        # 最低分辨率对应的空间尺寸（仅供参考，可用于调试）
        curr_res = self.config.resolution // (2 ** (self.config.num_resolutions - 1))
        # z -> block_in
        block_in = self.config.hidden_channels * self.config.channel_mult[self.config.num_resolutions - 1]
        hidden_states = Conv2dSame(features=block_in, kernel_size=(3, 3), use_bias=True)(hidden_states)

        # 中间层：残差块
        for _ in range(self.config.num_res_blocks):
            hidden_states = ResnetBlock(
                in_channels=block_in,
                out_channels=block_in,
                dropout=self.config.dropout,
                norm_num_groups=32,
                activation=nn.silu,
            )(hidden_states, deterministic=deterministic)




        # 上采样部分：逆序执行各层
        for i_level in reversed(range(self.config.num_resolutions)):

            # print(f'{hidden_states.shape=}')
            hidden_states = UpsamplingBlock(config=self.config, block_idx=i_level,name=f'UpsamplingBlock_{i_level}')(hidden_states, deterministic=deterministic)

        # up_samples_blocks=[UpsamplingBlock
        #                    (config=self.config, block_idx=i_level)(hidden_states, deterministic=deterministic) for i_level in range(self.config.num_resolutions)]
        #
        #
        # for block in up_samples_blocks:
        #     hidden_states=block(hidden_states, deterministic=deterministic)



        # 结束部分：GroupNorm -> silu -> 3x3 Conv 得到输出通道数（如 3）
        hidden_states = nn.GroupNorm(num_groups=32, epsilon=1e-6)(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = Conv2dSame(features=self.config.num_channels, kernel_size=(3, 3),use_bias=True)(hidden_states)
        return hidden_states

# =============================================================================
# VectorQuantizer，与 PyTorch 版本 VectorQuantizer 对应
# =============================================================================
class VectorQuantizer(nn.Module):
    num_embeddings: int
    embedding_dim: int
    commitment_cost: float

    def setup(self):
        init_bound = 1.0 / self.num_embeddings
        self.embedding = self.param(
            "embedding",
            lambda key, shape: jax.random.uniform(key, shape, minval=-init_bound, maxval=init_bound),
            (self.num_embeddings, self.embedding_dim)
        )

    def __call__(self, hidden_states, *, return_loss: bool = False):
        """
        hidden_states: Tensor，形状 (B, C, H, W)（与 PyTorch 版本一致）
        返回：z_q (Tensor，同样形状), min_encoding_indices (Tensor, shape (B, num_tokens)), loss（若 return_loss=True）
        """
        # 将 hidden_states 从 (B, C, H, W) 转为 (B, H, W, C)
        print(f'{hidden_states.shape=}')
        # hidden_states_perm = jnp.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states_perm=hidden_states
        flat_hidden = hidden_states_perm.reshape(-1, self.embedding_dim)

        # 计算欧氏距离：||z - e||^2 = z^2 + e^2 - 2 z·e
        z_sq = jnp.sum(flat_hidden ** 2, axis=1, keepdims=True)  # (N, 1)
        e_sq = jnp.sum(self.embedding ** 2, axis=1)  # (num_embeddings,)
        distances = z_sq + e_sq - 2 * jnp.dot(flat_hidden, self.embedding.T)  # (N, num_embeddings)

        # 找到每个向量最近的 embedding 索引
        encoding_indices = jnp.argmin(distances, axis=1)  # (N,)
        encodings = jax.nn.one_hot(encoding_indices, self.num_embeddings, dtype=hidden_states.dtype)  # (N, num_embeddings)
        quantized_flat = jnp.dot(encodings, self.embedding)  # (N, embedding_dim)
        quantized = quantized_flat.reshape(hidden_states_perm.shape)
        # print(f'{quantized.shape=}')
        z_q=quantized
        # z_q = jnp.transpose(quantized, (0, 3, 1, 2))

        loss = None
        if return_loss:
            loss = jnp.mean((jax.lax.stop_gradient(z_q) - hidden_states) ** 2) + \
                   self.commitment_cost * jnp.mean((z_q - jax.lax.stop_gradient(hidden_states)) ** 2)
            # 梯度穿透
            z_q = hidden_states + jax.lax.stop_gradient(z_q - hidden_states)

        B = hidden_states.shape[0]
        num_tokens = flat_hidden.shape[0] // B
        min_encoding_indices = encoding_indices.reshape(B, num_tokens)
        return z_q, min_encoding_indices, loss

    def get_codebook_entry(self, indices):
        """
        根据给定 indices（形状 (B, num_tokens) 或 (B, H, W)）获得对应的量化向量，
        并 reshape 为 (B, C, H, W)（若 indices 为 2D，则假定 H=W=sqrt(num_tokens)）
        """
        print(f'{indices.shape=}')
        if indices.ndim == 2:
            B, num_tokens = indices.shape
            side = int(math.sqrt(num_tokens))
            z_q = jnp.take(self.embedding, indices, axis=0)
            z_q = z_q.reshape(B, side, side, -1)
            # z_q = jnp.transpose(z_q, (0, 3, 1, 2))
        elif indices.ndim == 3:
            B, H, W = indices.shape
            z_q = jnp.take(self.embedding, indices, axis=0)
            z_q = z_q.reshape(B, H, W, -1)
            # z_q = jnp.transpose(z_q, (0, 3, 1, 2))
        else:
            raise NotImplementedError(f"indices shape {indices.shape} not supported.")
        return z_q

# =============================================================================
# PretrainedTokenizer，与 PyTorch 版本对应
#
# 注意：
# 1. Flax 模型通常不在模块内部加载预训练权重，而是在外部加载 checkpoint 并传入模型参数。
#    此处仅给出一个 load_pretrained() 的示例提示。
# 2. Flax 中没有 eval() 模式，只需在 forward 时传入 deterministic=True 。
# =============================================================================
class PretrainedTokenizer(nn.Module):
    config: Any  # 配置对象，应包含：channel_mult, num_resolutions, dropout, hidden_channels,
                 # num_channels, num_res_blocks, resolution, z_channels

    def setup(self):
        self.encoder = Encoder(config=self.config)
        self.decoder = Decoder(config=self.config)
        self.quantize = VectorQuantizer(
            num_embeddings=1024,
            embedding_dim=self.config.z_channels,
            commitment_cost=0.25
        )

    def encode(self, x, *, deterministic: bool = True):
        """
        x: 输入图像，形状 (B, C, H, W)
        返回：codebook_indices，形状 (B, num_tokens)
        """
        hidden_states = self.encoder(x, deterministic=deterministic)
        _, codebook_indices, _ = self.quantize(hidden_states, return_loss=False)
        return codebook_indices

    def decode(self, codes, *, deterministic: bool = True):
        """
        codes: codebook 索引，形状 (B, num_tokens) 或 (B, H, W)
        返回：重构图像，形状 (B, C, H, W)，数值在 [0, 1] 内的法
        """
        quantized_states = self.quantize.get_codebook_entry(codes)
        rec_images = self.decoder(quantized_states, deterministic=deterministic)
        rec_images = jnp.clip(rec_images, 0.0, 1.0)
        return rec_images

    def decode_tokens(self, codes, *, deterministic: bool = True):
        return self.decode(codes, deterministic=deterministic)

    @classmethod
    def load_pretrained(cls, pretrained_path: str, config: Any, rng, dummy_input):
        """
        示例函数：加载预训练权重（checkpoint）并初始化模型参数
        注意：实际实现需要根据具体 checkpoint 格式编写加载代码
        """
        model = cls(config=config)
        variables = model.init(rng, dummy_input)
        # 例如：使用 flax.training.checkpoints.restore_checkpoint 加载预训练参数
        # pretrained_variables = checkpoints.restore_checkpoint(pretrained_path, target=None)
        # 这里假定 pretrained_variables 与 variables 结构一致
        # variables = pretrained_variables
        return variables



    def __call__(self, x):
        return self.decode(self.encode(x))


# =============================================================================
# 使用示例
# =============================================================================
if __name__ == "__main__":
    # 示例配置（可根据需要修改或使用 ml_collections.ConfigDict）
    class Config:
        channel_mult = [1, 1, 2, 2, 4]
        num_resolutions = 5
        dropout = 0.0
        hidden_channels = 128
        num_channels = 3
        num_res_blocks = 2
        # resolution = 256
        resolution = 32
        z_channels = 256

    config = Config()

    # 构造 PretrainedTokenizer 模型
    model = PretrainedTokenizer(config=config)

    # 初始化模型参数（假设输入形状为 (B, C, H, W)，例如 B=1）
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1,  config.resolution, config.resolution,config.num_channels,), jnp.float32)
    variables = model.init(rng, dummy_input)
    params=variables['params']

    print(params['encoder'].keys())
    print(params['decoder'].keys())
    print(params['quantize'].keys())


    # 若有预训练权重，可以通过 load_pretrained() 加载，例如：
    # variables = PretrainedTokenizer.load_pretrained("pretrained_checkpoint_path", config, rng, dummy_input)

    # 使用 encode() 和 decode() 方法（注意：Flax 的 apply() 需要传入参数字典）
    codebook_indices = model.apply(variables, dummy_input, method=PretrainedTokenizer.encode, deterministic=True)
    reconstructed = model.apply(variables, codebook_indices, method=PretrainedTokenizer.decode, deterministic=True)

    print("codebook_indices shape:", codebook_indices.shape)
    print("reconstructed image shape:", reconstructed.shape)
