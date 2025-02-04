import re
import numpy as np


def conv_weight_transpose(pt_array):
    """
    将 PyTorch 卷积权重从 [out, in, kh, kw] 转置为 Flax 的 [kh, kw, in, out].
    """
    if pt_array.ndim == 4:
        return np.transpose(pt_array, (2, 3, 1, 0))
    return pt_array


def convert_param_name(pt_param_name):
    """
    根据模块所属，转换参数名称：
      - 对于归一化层：将 "weight" 映射为 "scale"。
      - 对于卷积层等：将 "weight" 映射为 "kernel"。
      - 其他保持不变。
    此函数仅用于子模块内的名称转换，不负责路径重组。
    """
    if pt_param_name == "weight":
        return "weight"  # 默认先返回 weight；后续在上层决定是否转为 kernel 或 scale
    return pt_param_name


def update_dict(tree, keys, value):
    """
    将 value 写入嵌套字典 tree 中，路径为 keys 列表。
    """
    key = keys[0]
    if len(keys) == 1:
        tree[key] = value
    else:
        if key not in tree:
            tree[key] = {}
        update_dict(tree[key], keys[1:], value)


def convert_vqgan_state_dict(pt_state_dict):
    """
    将 PyTorch 的 state_dict 转换为符合 Flax 模型参数结构的字典。
    假设 pt_state_dict 的 key 格式为：
      encoder.conv_in.weight
      encoder.down.<i>.block.<j>.<submodule>.<param>
      encoder.mid.<j>.<submodule>.<param>
      encoder.norm_out.<param>
      encoder.conv_out.<param>
      decoder.conv_in.<param>
      decoder.mid.<j>.<submodule>.<param>
      decoder.up.<i>.block.<j>.<submodule>.<param>
      decoder.up.<i>.upsample_conv.<param>
      decoder.norm_out.<param>
      decoder.conv_out.<param>
      quantize.embedding.weight
    根据下面的映射规则重新组织到嵌套字典中。
    """
    flax_params = {}

    for pt_key, pt_val in pt_state_dict.items():
        # 若 pt_val 为 torch.Tensor，则转换为 numpy 数组
        if not isinstance(pt_val, np.ndarray):
            pt_val = pt_val.cpu().numpy() if hasattr(pt_val, "cpu") else np.array(pt_val)

        # 用正则匹配不同模块
        if pt_key.startswith("encoder.conv_in."):
            # encoder.conv_in.weight / bias
            m = re.match(r"encoder\.conv_in\.(\w+)", pt_key)
            if m:
                param = m.group(1)
                # 映射到 encoder -> "Conv_0"
                target_key = ["encoder", "Conv_0"]
                if param == "weight":
                    target_param = "kernel"
                    pt_val = conv_weight_transpose(pt_val)
                else:
                    target_param = param
                update_dict(flax_params, target_key + [target_param], pt_val)
        elif pt_key.startswith("encoder.down."):
            # 匹配 encoder.down.<i>.block.<j>.<sub>.<param>
            m = re.match(r"encoder\.down\.(\d+)\.block\.(\d+)\.([^\.]+)\.(\w+)", pt_key)
            if m:
                i, j, subname, param = m.groups()
                # 映射到 encoder -> DownsamplingBlock_{i} -> ResnetBlock_{j} -> <subname_converted>
                target_key = ["encoder", f"DownsamplingBlock_{i}", f"ResnetBlock_{j}"]
                # 对于 norm 层：weight -> scale
                if "norm" in subname:
                    if param == "weight":
                        target_param = "scale"
                    else:
                        target_param = param
                # 对于卷积相关层（conv, nin_shortcut）及 upsample_conv（此处不适用）：
                elif ("conv" in subname) or ("nin_shortcut" in subname):
                    if subname=="nin_shortcut":
                        print(subname)
                        print(pt_val.shape,pt_key)

                    if param == "weight":
                        target_param = "kernel"
                        pt_val = conv_weight_transpose(pt_val)
                    else:
                        target_param = param
                else:
                    target_param = param
                # 有时 subname 本身也需要保留，如 "nin_shortcut"（保持名称不变）
                # 最终我们在该层下以 subname 作为一级字典
                update_dict(flax_params, target_key + [subname, target_param], pt_val)
            else:
                # 若不匹配上述格式，可打印提示
                print("Unmatched encoder.down key:", pt_key)
        elif pt_key.startswith("encoder.mid."):
            # 匹配 encoder.mid.<j>.<sub>.<param>
            m = re.match(r"encoder\.mid\.(\d+)\.([^\.]+)\.(\w+)", pt_key)
            if m:
                j, subname, param = m.groups()
                # 映射到 encoder -> ResnetBlock_{j} -> <subname>
                target_key = ["encoder", f"ResnetBlock_{j}"]
                if "norm" in subname:
                    if param == "weight":
                        target_param = "scale"
                    else:
                        target_param = param
                elif ("conv" in subname):
                    if param == "weight":
                        target_param = "kernel"
                        pt_val = conv_weight_transpose(pt_val)
                    else:
                        target_param = param
                else:
                    target_param = param
                update_dict(flax_params, target_key + [subname, target_param], pt_val)
            else:
                print("Unmatched encoder.mid key:", pt_key)
        elif pt_key.startswith("encoder.norm_out."):
            # 映射到 encoder -> GroupNorm_0
            m = re.match(r"encoder\.norm_out\.(\w+)", pt_key)
            if m:
                param = m.group(1)
                target_key = ["encoder", "GroupNorm_0"]
                # 对于 norm 层：weight -> scale
                if param == "weight":
                    target_param = "scale"
                else:
                    target_param = param
                update_dict(flax_params, target_key + [target_param], pt_val)
        elif pt_key.startswith("encoder.conv_out."):
            m = re.match(r"encoder\.conv_out\.(\w+)", pt_key)
            if m:
                param = m.group(1)
                target_key = ["encoder", "Conv_1"]
                if param == "weight":
                    target_param = "kernel"
                    pt_val = conv_weight_transpose(pt_val)
                else:
                    target_param = param
                update_dict(flax_params, target_key + [target_param], pt_val)
        elif pt_key.startswith("decoder.conv_in."):
            m = re.match(r"decoder\.conv_in\.(\w+)", pt_key)
            if m:
                param = m.group(1)
                target_key = ["decoder", "Conv_0"]
                if param == "weight":
                    target_param = "kernel"
                    pt_val = conv_weight_transpose(pt_val)
                else:
                    target_param = param
                update_dict(flax_params, target_key + [target_param], pt_val)
        elif pt_key.startswith("decoder.mid."):
            m = re.match(r"decoder\.mid\.(\d+)\.([^\.]+)\.(\w+)", pt_key)
            if m:
                j, subname, param = m.groups()
                target_key = ["decoder", f"ResnetBlock_{j}"]
                if "norm" in subname:
                    if param == "weight":
                        target_param = "scale"
                    else:
                        target_param = param
                elif "conv" in subname:
                    if param == "weight":
                        target_param = "kernel"
                        pt_val = conv_weight_transpose(pt_val)
                    else:
                        target_param = param
                else:
                    target_param = param
                update_dict(flax_params, target_key + [subname, target_param], pt_val)
            else:
                print("Unmatched decoder.mid key:", pt_key)
        elif pt_key.startswith("decoder.up."):
            # 分为两种情况：up.<i>.block.<j>.<sub>.<param> 或 up.<i>.upsample_conv.<param>
            m1 = re.match(r"decoder\.up\.(\d+)\.block\.(\d+)\.([^\.]+)\.(\w+)", pt_key)
            if m1:
                i, j, subname, param = m1.groups()
                target_key = ["decoder", f"UpsamplingBlock_{i}", f"ResnetBlock_{j}"]
                if "norm" in subname:
                    if param == "weight":
                        target_param = "scale"
                    else:
                        target_param = param
                elif ("conv" in subname) or ("nin_shortcut" in subname):
                    if param == "weight":
                        target_param = "kernel"
                        pt_val = conv_weight_transpose(pt_val)
                    else:
                        target_param = param
                else:
                    target_param = param
                update_dict(flax_params, target_key + [subname, target_param], pt_val)
            else:
                m2 = re.match(r"decoder\.up\.(\d+)\.upsample_conv\.(\w+)", pt_key)
                if m2:
                    i, param = m2.groups()
                    target_key = ["decoder", f"UpsamplingBlock_{i}", "upsample_conv"]
                    if param == "weight":
                        target_param = "kernel"
                        pt_val = conv_weight_transpose(pt_val)
                    else:
                        target_param = param
                    update_dict(flax_params, target_key + [target_param], pt_val)
                else:
                    print("Unmatched decoder.up key:", pt_key)
        elif pt_key.startswith("decoder.norm_out."):
            m = re.match(r"decoder\.norm_out\.(\w+)", pt_key)
            if m:
                param = m.group(1)
                target_key = ["decoder", "GroupNorm_0"]
                if param == "weight":
                    target_param = "scale"
                else:
                    target_param = param
                update_dict(flax_params, target_key + [target_param], pt_val)
        elif pt_key.startswith("decoder.conv_out."):
            m = re.match(r"decoder\.conv_out\.(\w+)", pt_key)
            if m:
                param = m.group(1)
                target_key = ["decoder", "Conv_1"]
                if param == "weight":
                    target_param = "kernel"
                    pt_val = conv_weight_transpose(pt_val)
                else:
                    target_param = param
                update_dict(flax_params, target_key + [target_param], pt_val)
        elif pt_key.startswith("quantize.embedding."):
            m = re.match(r"quantize\.embedding\.(\w+)", pt_key)
            if m:
                param = m.group(1)
                target_key = ["quantize", "embedding"]
                # 对于嵌入层，仅考虑 weight
                update_dict(flax_params, target_key, pt_val)
        else:
            print("Unmatched key:", pt_key)

    return flax_params


# =============================================================================
# 示例：假设我们有部分 PyTorch state_dict（实际使用时请加载完整 state_dict）
# =============================================================================
if __name__ == "__main__":
    # 示例 PyTorch state_dict（这里只给出少量示例键）
    pt_state_dict = {
        "encoder.conv_in.weight": np.random.randn(128, 3, 3, 3).astype(np.float32),
        "encoder.down.0.block.0.norm1.weight": np.random.randn(128).astype(np.float32),
        "encoder.down.0.block.0.norm1.bias": np.random.randn(128).astype(np.float32),
        "encoder.down.0.block.0.conv1.weight": np.random.randn(128, 128, 3, 3).astype(np.float32),
        "encoder.mid.0.norm1.weight": np.random.randn(256).astype(np.float32),
        "encoder.mid.0.norm1.bias": np.random.randn(256).astype(np.float32),
        "encoder.norm_out.weight": np.random.randn(256).astype(np.float32),
        "encoder.norm_out.bias": np.random.randn(256).astype(np.float32),
        "encoder.conv_out.weight": np.random.randn(256, 256, 1, 1).astype(np.float32),
        "encoder.conv_out.bias": np.random.randn(256).astype(np.float32),
        "decoder.conv_in.weight": np.random.randn(512, 256, 3, 3).astype(np.float32),
        "decoder.conv_in.bias": np.random.randn(512).astype(np.float32),
        "decoder.up.0.block.0.norm1.weight": np.random.randn(512).astype(np.float32),
        "decoder.up.0.block.0.norm1.bias": np.random.randn(512).astype(np.float32),
        "decoder.up.0.block.0.conv1.weight": np.random.randn(512, 512, 3, 3).astype(np.float32),
        "decoder.norm_out.weight": np.random.randn(128).astype(np.float32),
        "decoder.norm_out.bias": np.random.randn(128).astype(np.float32),
        "decoder.conv_out.weight": np.random.randn(3, 128, 3, 3).astype(np.float32),
        "decoder.conv_out.bias": np.random.randn(3).astype(np.float32),
        "quantize.embedding.weight": np.random.randn(1024, 256).astype(np.float32),
    }

    flax_params = convert_vqgan_state_dict(pt_state_dict)

    # 检查转换后的结构
    import pprint

    print("Encoder keys:")
    pprint.pprint(flax_params.get("encoder", {}).keys())
    print("Decoder keys:")
    pprint.pprint(flax_params.get("decoder", {}).keys())
    print("Quantize keys:")
    pprint.pprint(flax_params.get("quantize", {}).keys())
