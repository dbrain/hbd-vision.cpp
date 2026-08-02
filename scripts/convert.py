#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "torchvision",
#   "timm",
#   "pytest",
#   "opencv-python",
#   "ruff",
#   "einops>=0.8.1",
#   "spandrel>=0.4.1",
#   "gguf>=0.17.1",
# ]
# ///
import argparse
import itertools
import torch
import safetensors
import numpy as np

from enum import Enum
from pathlib import Path
from gguf import GGUFWriter, Metadata, GGML_QUANT_VERSION
from torch import Tensor

#
# Common


class TensorLayout(Enum):
    unknown = "unknown"
    nchw = "whcn"
    nhwc = "cwhn"

    @staticmethod
    def parse(s: str):
        if s == "whcn" or s == "nchw":
            return TensorLayout.nchw
        if s == "cwhn" or s == "nhwc":
            return TensorLayout.nhwc
        return TensorLayout.unknown


class Writer(GGUFWriter):
    def __init__(self, path: Path, arch_name: str, float_type: str, verbose: bool):
        super().__init__(path, arch_name)
        self.arch = arch_name
        self.float_type = float_type
        self.tensor_layout = TensorLayout.unknown
        self.verbose = verbose
        self.conv2d_weights: list[int] = []
        self._index = 0

    def add_tensor(self, name: str, tensor: Tensor, float_type: str | None = None):
        if len(name) >= 64:
            print("Warning: name too long", len(name), name)

        float_type = float_type or self.float_type
        if float_type == "f16" and tensor.dtype == torch.float32:
            tensor = tensor.to(torch.float16)

        tensor_data = tensor.numpy()
        if self.verbose:
            print(name, tensor.shape, tensor_data.dtype)
        super().add_tensor(name, tensor_data)
        self._index += 1

    def convert_tensor_2d(self, tensor: Tensor):
        # assume tensor is NCHW layout (PyTorch default)
        if self.tensor_layout is TensorLayout.nhwc:
            return conv_2d_to_nhwc(tensor)
        else:
            # add tensor index to list to optionally convert layout on the fly later
            self.conv2d_weights.append(self._index)
            return tensor

    def add_int32(self, key: str, val: int):
        print("*", key, "=", val)
        super().add_int32(key, val)

    def set_tensor_layout(self, layout: TensorLayout):
        print("*", f"{self.arch}.tensor_data_layout", "=", layout.value)
        self.tensor_layout = layout
        self.add_tensor_data_layout(layout.value)

    def set_tensor_layout_default(self, layout: TensorLayout):
        if self.tensor_layout is TensorLayout.unknown:
            self.set_tensor_layout(layout)

    def add_conv2d_weight_indices(self):
        if self.conv2d_weights:
            self.add_array(f"{self.arch}.conv2d_weights", self.conv2d_weights)


def load_model(path: Path) -> dict[str, Tensor]:
    if path.suffix in [".safetensors", ".safetensor"]:
        weights = safetensors.safe_open(path, "pt")
        return {k: weights.get_tensor(k) for k in weights.keys()}
    else:
        return torch.load(path, map_location="cpu", weights_only=True)


batch_norm_eps = 1e-5


def is_conv_2d(name: str, tensor: Tensor):
    return (
        tensor.ndim == 4
        and tensor.shape[2] == tensor.shape[3]
        and tensor.shape[2] in (1, 3, 4, 7, 14)
        and name.endswith("weight")
    )


def conv_2d_to_nhwc(kernel: Tensor):
    c_in = kernel.shape[1]
    if c_in == 1:  # depthwise
        return kernel.permute(2, 3, 1, 0)  # H W 1 C_out
    else:
        return kernel.permute(0, 2, 3, 1)  # C_out H W C_in


def conv_transpose_2d_to_nhwc(kernel: Tensor):
    # C_in C_out H W -> C_out H W C_in
    return kernel.permute(1, 2, 3, 0)


def fuse_batch_norm(model: dict[str, Tensor], key: str, key_bn: str):
    suffix_weight = f"{key_bn}.weight"
    suffix_bias = f"{key_bn}.bias"

    if key.endswith(suffix_weight):
        base = key.removesuffix(suffix_weight)
        weight = model[key]
        var = model[f"{base}{key_bn}.running_var"]
        return weight / torch.sqrt(var + batch_norm_eps)

    elif key.endswith(suffix_bias):
        base = key.removesuffix(suffix_bias)
        bias = model[key]
        weight = model[f"{base}{key_bn}.weight"]
        mean = model[f"{base}{key_bn}.running_mean"]
        var = model[f"{base}{key_bn}.running_var"]
        return bias - mean * weight / torch.sqrt(var + batch_norm_eps)

    elif key.endswith(f"{key_bn}.running_mean") or key.endswith(f"{key_bn}.running_var"):
        return None

    return model[key]


def fuse_conv_2d_batch_norm(
    model: dict[str, Tensor],
    key: str,
    name: str,
    key_module: str,
    key_conv: str,
    key_norm: str,
    writer: Writer,
):
    suffix_conv = f"{key_module}{key_conv}.weight"
    suffix_bias = f"{key_module}{key_conv}.bias"
    suffix_norm = f"{key_module}{key_norm}."

    if key.endswith(suffix_conv):
        conv_weight = model[key]
        base = key.removesuffix(suffix_conv)
        bn_weight = model.get(f"{base}{suffix_norm}weight")
        if bn_weight is None:
            return False
        bn_bias = model[f"{base}{suffix_norm}bias"]
        bn_mean = model[f"{base}{suffix_norm}running_mean"]
        bn_var = model[f"{base}{suffix_norm}running_var"]
        conv_bias = model.get(f"{base}{suffix_bias}", torch.zeros_like(bn_bias))

        bn_weight = bn_weight / torch.sqrt(bn_var + batch_norm_eps)
        fused_weight = conv_weight * bn_weight[:, None, None, None]
        fused_bias = (conv_bias - bn_mean) * bn_weight + bn_bias

        fused_weight = writer.convert_tensor_2d(fused_weight)
        writer.add_tensor(name, fused_weight)
        writer.add_tensor(name.replace("weight", "bias"), fused_bias)
        return True

    elif key.endswith(suffix_bias):
        base = key.removesuffix(suffix_bias)
        return f"{base}{suffix_norm}weight" in model

    elif suffix_norm in key:
        return True  # batch norm was fused above

    return False  # tensor is not part of conv2d+batch-norm


#
# MobileSAM


def convert_sam(input_filepath: Path, writer: Writer):
    writer.add_license("apache-2.0")
    writer.set_tensor_layout_default(TensorLayout.nchw)

    model = load_model(input_filepath)

    for key, tensor in model.items():
        name = key
        name = name.replace("image_encoder.", "enc.")
        name = name.replace("mask_decoder.", "dec.")
        name = name.replace("_image_to_token.", "_i2t.")
        name = name.replace("_token_to_image.", "_t2i.")

        if name.endswith("attention_biases"):
            num_heads = tensor.shape[0]
            resolution = {4: 7, 5: 14, 10: 7}[num_heads]
            attention_bias_idxs = build_attention_bias_indices(resolution)
            name = name + "_indexed"
            tensor = tensor[:, attention_bias_idxs]

        if "local_conv" in key:  # always convert to nhwc
            original_tensor_layout = writer.tensor_layout
            writer.tensor_layout = TensorLayout.nhwc
            fuse_conv_2d_batch_norm(model, key, name, "", "c", "bn", writer)
            writer.tensor_layout = original_tensor_layout
            continue

        if fuse_conv_2d_batch_norm(model, key, name, "", "c", "bn", writer):
            continue

        if name.endswith("neck.0.weight") or name.endswith("neck.2.weight"):
            assert tensor.shape[2] == tensor.shape[3] and tensor.shape[2] <= 3
            tensor = writer.convert_tensor_2d(tensor)

        # Precompute dense positional embeddings from random matrix stored in the model
        if name == "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix":
            pe = build_dense_positional_embeddings(tensor)
            writer.add_tensor("dec.dense_positional_embedding", pe, "f32")

        if name in ["dec.iou_token.weight", "dec.mask_tokens.weight"]:
            writer.add_tensor(name, tensor, "f32")
            continue

        writer.add_tensor(name, tensor)


def build_attention_bias_indices(resolution: int):
    points = list(itertools.product(range(resolution), range(resolution)))
    N = len(points)
    attention_offsets = {}
    idxs = []
    for p1 in points:
        for p2 in points:
            offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
            if offset not in attention_offsets:
                attention_offsets[offset] = len(attention_offsets)
            idxs.append(attention_offsets[offset])

    return torch.LongTensor(idxs).view(N, N)


def build_dense_positional_embeddings(
    positional_encoding_gaussian_matrix: torch.Tensor, image_embedding_size=64
):
    # from sam/modeling/prompt_encoder.py - PositionEmbeddingRandom
    h, w = image_embedding_size, image_embedding_size
    grid = torch.ones((h, w), dtype=torch.float32)
    y_embed = grid.cumsum(dim=0) - 0.5
    x_embed = grid.cumsum(dim=1) - 0.5
    y_embed = y_embed / h
    x_embed = x_embed / w

    coords = torch.stack((x_embed, y_embed), dim=-1)
    coords = 2 * coords - 1
    coords = coords @ positional_encoding_gaussian_matrix
    coords = 2 * np.pi * coords
    # outputs d_1 x ... x d_n x C shape
    pe = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    return pe


#
# BirefNet


def convert_birefnet(input_filepath: Path, writer: Writer):
    writer.add_license("mit")
    writer.set_tensor_layout_default(TensorLayout.nchw)

    model = load_model(input_filepath)

    x = model["bb.layers.0.blocks.0.attn.proj.bias"]
    if x.shape[0] == 96:
        writer.add_string("swin.config", "tiny")
        writer.add_int32("swin.embed_dim", 96)
    elif x.shape[0] == 192:
        writer.add_string("swin.config", "large")
        writer.add_int32("swin.embed_dim", 192)
    else:
        raise ValueError(f"Unsupported Swin Transformer embed dim: {x.shape[0]}")

    image_size = 1024
    if "HR" in input_filepath.name or "2K" in input_filepath.name:
        image_size = 2048  # actually 2K should rather be 2560x1440
    elif "dynamic" in input_filepath.name:
        image_size = -1
    writer.add_int32("birefnet.image_size", image_size)
    writer.add_int32("birefnet.image_multiple", 128)

    for key, tensor in model.items():
        # Shorten some names to fit into 64 chars
        name = key
        name = name.replace("decoder_block", "block")
        name = name.replace("atrous_conv", "conv")
        name = name.replace("modulator_conv", "modulator")
        name = name.replace("offset_conv", "offset")
        name = name.replace("regular_conv", "conv")

        if name.endswith("relative_position_index"):
            continue  # precomputed in c++ code

        # Fuse all regular conv + batch norm pairs into a single conv with bias
        if fuse_conv_2d_batch_norm(model, key, name, "global_avg_pool.", "1", "2", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "dec_att.", "conv1", "bn1", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "", "conv_in", "bn_in", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "", "conv_out", "bn_out", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "gdt_convs_4.", "0", "1", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "gdt_convs_3.", "0", "1", writer):
            continue
        if fuse_conv_2d_batch_norm(model, key, name, "gdt_convs_2.", "0", "1", writer):
            continue

        # Fuse batch norm into multiply+add for deformable conv
        tensor = fuse_batch_norm(model, key, "bn")
        if tensor is None:
            continue  # batch norm was fused

        if is_conv_2d(name, tensor):
            if "patch_embed" in name:  # part of SWIN, always store as NHWC
                tensor = conv_2d_to_nhwc(tensor)
            else:  # store rest in requested tensor layout
                tensor = writer.convert_tensor_2d(tensor)

        writer.add_tensor(name, tensor)


#
# Depth-Anything


def convert_depth_anything(input_filepath: Path, writer: Writer):
    if "small" in input_filepath.name.lower():
        writer.add_license("apache-2.0")
    else:
        writer.add_license("cc-by-nc-4.0")
    writer.set_tensor_layout_default(TensorLayout.nchw)

    model = load_model(input_filepath)

    if "pretrained.cls_token" in model:
        print("The converter is written for the transformers (.safetensors) version of the model.")
        print("The original weights (.pth) are currently not supported.")
        raise ValueError("Weights not supported")

    shape = model["backbone.embeddings.patch_embeddings.projection.weight"].shape
    writer.add_int32("dino.patch_size", shape[2])
    writer.add_int32("dino.embed_dim", shape[0])
    writer.add_int32("depthanything.image_size", 518)
    match shape[0]:
        case 384:  # Small
            writer.add_int32("dino.n_heads", 6)
            writer.add_int32("dino.n_layers", 12)
            writer.add_array("depthanything.feature_layers", [2, 5, 8, 11])
        case 768:  # Base
            writer.add_int32("dino.n_heads", 12)
            writer.add_int32("dino.n_layers", 12)
            writer.add_array("depthanything.feature_layers", [2, 5, 8, 11])
        case 1024:  # Large
            writer.add_int32("dino.n_heads", 16)
            writer.add_int32("dino.n_layers", 24)
            writer.add_array("depthanything.feature_layers", [4, 11, 17, 23])

    for key, tensor in model.items():
        name = key

        if is_conv_2d(name, tensor):
            if "patch_embeddings" in name or ("projection" in name and "fusion" not in name):
                tensor = conv_2d_to_nhwc(tensor)
            elif "0.resize" in name or "1.resize" in name:
                pass  # ConvTranspose2D, don't change layout
            else:
                tensor = writer.convert_tensor_2d(tensor)

        if "position_embeddings" in name or "cls_token" in name:
            writer.add_tensor(name, tensor, "f32")
            continue

        writer.add_tensor(name, tensor)


#
# Depth-Anything-3
#
# depth-anything/DA3-* (ByteDance-Seed/depth-anything-3, Apache-2.0). Verified against
# DA3-SMALL: 440 stored tensors (446 state_dict entries; safetensors de-duplicates 6 tied
# LayerNorm params), 0.08B params, config.json `net=vits out_layers=[5,7,9,11] alt_start=4
# qknorm_start=4 rope_start=4 cat_token=true`, `head=DualDPT dim_in=768 features=64
# out_channels=[48,96,192,384] output_dim=2`.
#
# The checkpoint uses ORIGINAL DINOv2 naming (`patch_embed.proj`, `blocks.N.attn.qkv`,
# `ls1.gamma`), not the transformers naming convert_depth_anything() requires. The encoder
# is renamed here to the transformers/HF layout that src/visp/arch/dino.cpp already reads,
# and the fused qkv is split into three linears, so the DINOv2 graph is reusable as-is for
# the parts DA3 did not change:
#
#   model.backbone.pretrained.patch_embed.proj.W  -> backbone.embeddings.patch_embeddings.projection.W
#   model.backbone.pretrained.cls_token           -> backbone.embeddings.cls_token
#   model.backbone.pretrained.pos_embed           -> backbone.embeddings.position_embeddings
#   model.backbone.pretrained.camera_token        -> backbone.embeddings.camera_token   (DA3 only)
#   model.backbone.pretrained.blocks.N.norm1/2    -> backbone.encoder.layer.N.norm1/2
#   model.backbone.pretrained.blocks.N.attn.qkv   -> backbone.encoder.layer.N.attention.attention.{query,key,value}
#   model.backbone.pretrained.blocks.N.attn.proj  -> backbone.encoder.layer.N.attention.output.dense
#   model.backbone.pretrained.blocks.N.attn.q_norm-> backbone.encoder.layer.N.attention.attention.q_norm  (DA3 only)
#   model.backbone.pretrained.blocks.N.attn.k_norm-> backbone.encoder.layer.N.attention.attention.k_norm  (DA3 only)
#   model.backbone.pretrained.blocks.N.ls1.gamma  -> backbone.encoder.layer.N.layer_scale1.lambda1
#   model.backbone.pretrained.blocks.N.ls2.gamma  -> backbone.encoder.layer.N.layer_scale2.lambda1
#   model.backbone.pretrained.blocks.N.mlp.fc1/2  -> backbone.encoder.layer.N.mlp.fc1/2
#   model.backbone.pretrained.norm                -> backbone.layernorm
#
# The DualDPT head, camera decoder and camera encoder keep their upstream names with the
# `model.` prefix stripped (`head.*`, `cam_dec.*`, `cam_enc.*`). cam_enc.trunk.N gets the
# same encoder rename so a single ggml transformer-block builder covers both.
#
# What the head tensors are for (the next port needs this map):
#   head.norm                          LayerNorm(768) over the concatenated [local|global] tokens
#   head.projects.i                    1x1 conv 768 -> out_channels[i], per pyramid stage (NHWC)
#   head.resize_layers.0/.1            ConvTranspose2d x4 / x2 (stage 0/1 upsample; NCHW kernels)
#   head.resize_layers.3               conv 3x3 stride 2 (stage 3 downsample); stage 2 is Identity
#   head.scratch.layerK_rn             3x3 conv out_channels[K-1] -> features, bias-free
#   head.scratch.refinenetK            DPT top-down fusion, DEPTH branch (K=4 top .. K=1 finest)
#   head.scratch.refinenetK_aux        same chain, independent weights, RAY branch
#     .resConfUnit1                    lateral residual unit (absent on refinenet4/4_aux)
#     .resConfUnit2                    main residual unit; then bilinear upsample, then
#     .out_conv                        1x1 conv 64->64
#   head.scratch.output_conv1          3x3 conv 64 -> 32, depth neck (applied inside _fuse)
#   head.scratch.output_conv2.0/.2     3x3 conv 32->32, then 1x1 conv 32 -> 2 depth logits
#   head.scratch.output_conv1_aux.L.*  5-conv ray neck per pyramid level L (64->32->64->32->64->32)
#   head.scratch.output_conv2_aux.L.0  3x3 conv 32->32 \
#   head.scratch.output_conv2_aux.L.2  LayerNorm(32)    | ray head, level L
#   head.scratch.output_conv2_aux.L.5  1x1 conv 32 -> 7 /
#   cam_dec.backbone.0/.2              768->768 Linear + ReLU x2 over the camera token
#   cam_dec.fc_t / fc_qvec / fc_fov.0  translation(3) / quaternion xyzw(4) / fov h,w(2)
#
# Outputs: depth = exp(output_conv2[...,0]), depth_conf = 1+exp(output_conv2[...,1]);
# ray = output_conv2_aux[-1][...,0:6] (linear), ray_conf = 1+exp(...[...,6]). Camera
# extrinsics/intrinsics come from cam_dec on the layer-11 camera token (token 0), or from
# the ray field when use_ray_pose is set. Only aux level 3 is consumed at inference; levels
# 0-2 of output_conv{1,2}_aux are dead weight kept for fidelity.
#
# Patch embedding is a plain 4D Conv2d [384,3,14,14] -> no ggml 4-dimension problem.


da3_encoder_renames = [
    ("patch_embed.proj", "embeddings.patch_embeddings.projection"),
    ("cls_token", "embeddings.cls_token"),
    ("camera_token", "embeddings.camera_token"),
    ("pos_embed", "embeddings.position_embeddings"),
    ("blocks.", "encoder.layer."),
    ("attn.proj", "attention.output.dense"),
    ("attn.q_norm", "attention.attention.q_norm"),
    ("attn.k_norm", "attention.attention.k_norm"),
    ("ls1.gamma", "layer_scale1.lambda1"),
    ("ls2.gamma", "layer_scale2.lambda1"),
    ("norm.weight", "layernorm.weight"),
    ("norm.bias", "layernorm.bias"),
]


def da3_rename_encoder(name: str, prefix: str) -> str:
    name = name.removeprefix(prefix)
    for old, new in da3_encoder_renames:
        # norm.weight/norm.bias only ever match the trailing final layer norm: the per-block
        # norm1/norm2 and q_norm/k_norm were already rewritten or carry a distinguishing prefix
        if name.startswith(old) or f".{old}" in name:
            name = name.replace(old, new)
    return name


def da3_split_qkv(name: str, tensor: Tensor, writer: Writer):
    q, k, v = tensor.chunk(3, dim=0)
    base = name.replace("attn.qkv", "attention.attention")
    for part, t in (("query", q), ("key", k), ("value", v)):
        writer.add_tensor(base.replace("attention.attention", f"attention.attention.{part}"), t)


def da3_block_index(key: str) -> int | None:
    for tag in ("pretrained.blocks.", "cam_enc.trunk."):
        if tag in key:
            return int(key.split(tag, 1)[1].split(".", 1)[0])
    return None


def convert_depth_anything_3(input_filepath: Path, writer: Writer, max_blocks: int = -1):
    import json

    writer.add_license("apache-2.0")
    writer.set_tensor_layout_default(TensorLayout.nchw)

    config_path = input_filepath.parent / "config.json"
    if not config_path.exists():
        raise ValueError(f"config.json not found next to the weights ({config_path})")
    config = json.loads(config_path.read_text())["config"]
    net_cfg = config["net"]

    model = load_model(input_filepath)
    if "model.backbone.pretrained.patch_embed.proj.weight" not in model:
        raise ValueError("Not a Depth-Anything-3 checkpoint (expected depth_anything_3 naming)")

    # safetensors de-duplicates tied parameters; the ray head shares one LayerNorm across all
    # pyramid levels, so re-materialize the copies rather than push the aliasing into the loader
    aux_ln = "model.head.scratch.output_conv2_aux"
    n_aux_levels = 1 + max(int(k.split(".")[-3]) for k in model if k.startswith(f"{aux_ln}."))
    for level in range(n_aux_levels):
        for suffix in ("weight", "bias"):
            model.setdefault(f"{aux_ln}.{level}.2.{suffix}", model[f"{aux_ln}.0.2.{suffix}"])

    patch = model["model.backbone.pretrained.patch_embed.proj.weight"]
    embed_dim, _, patch_size, _ = patch.shape
    n_layers = 1 + max(
        int(k.split(".")[4]) for k in model if k.startswith("model.backbone.pretrained.blocks.")
    )
    q_norm = next(k for k in model if k.endswith("attn.q_norm.weight"))
    head_dim = model[q_norm].shape[0]
    n_pos = model["model.backbone.pretrained.pos_embed"].shape[1] - 1
    pos_grid = round(n_pos**0.5)

    writer.add_int32("dino.patch_size", patch_size)
    writer.add_int32("dino.embed_dim", embed_dim)
    writer.add_int32("dino.n_heads", embed_dim // head_dim)
    writer.add_int32("dino.n_layers", n_layers)

    arch = "depthanything3"
    writer.add_int32(f"{arch}.image_size", 504)  # api.py inference(process_res=504)
    writer.add_int32(f"{arch}.image_multiple", patch_size)
    writer.add_string(f"{arch}.resize_mode", "upper_bound")  # bound the LONGEST side
    writer.add_int32(f"{arch}.pos_embed_grid", pos_grid)
    writer.add_array(f"{arch}.feature_layers", net_cfg["out_layers"])
    writer.add_int32(f"{arch}.alt_start", net_cfg["alt_start"])
    writer.add_int32(f"{arch}.qknorm_start", net_cfg["qknorm_start"])
    writer.add_int32(f"{arch}.rope_start", net_cfg["rope_start"])
    writer.add_float32(f"{arch}.rope_frequency", 100.0)
    writer.add_bool(f"{arch}.cat_token", net_cfg["cat_token"])
    writer.add_int32(f"{arch}.patch_start_idx", 1)

    head_cfg = config["head"]
    out_channels = [model[f"model.head.projects.{i}.weight"].shape[0] for i in range(4)]
    writer.add_int32(f"{arch}.head_dim_in", model["model.head.norm.weight"].shape[0])
    writer.add_int32(f"{arch}.head_features", model["model.head.scratch.layer1_rn.weight"].shape[0])
    writer.add_array(f"{arch}.head_out_channels", out_channels)
    writer.add_int32(f"{arch}.head_output_dim", head_cfg["output_dim"])
    writer.add_int32(f"{arch}.head_aux_output_dim", model[f"{aux_ln}.0.5.weight"].shape[0])
    writer.add_int32(f"{arch}.head_aux_levels", n_aux_levels)
    writer.add_float32(f"{arch}.head_pos_embed_ratio", 0.1)
    writer.add_float32(f"{arch}.head_pos_embed_omega", 100.0)
    writer.add_string(f"{arch}.head_activation", "exp")
    writer.add_string(f"{arch}.head_conf_activation", "expp1")

    writer.add_int32(f"{arch}.cam_dec_dim", model["model.cam_dec.fc_t.weight"].shape[1])
    writer.add_int32(f"{arch}.cam_enc_dim", model["model.cam_enc.token_norm.weight"].shape[0])
    writer.add_int32(
        f"{arch}.cam_enc_layers",
        1 + max(int(k.split(".")[3]) for k in model if k.startswith("model.cam_enc.trunk.")),
    )
    writer.add_int32(f"{arch}.cam_enc_n_heads", 16)  # CameraEnc default, not derivable from weights

    backbone_prefix = "model.backbone.pretrained."
    for key in sorted(model.keys()):
        block = da3_block_index(key)
        if max_blocks >= 0 and block is not None and block >= max_blocks:
            continue

        tensor = model[key]
        if key.startswith(backbone_prefix):
            name = "backbone." + da3_rename_encoder(key, backbone_prefix)
        elif key.startswith("model.cam_enc.trunk."):
            name = "cam_enc." + da3_rename_encoder(key, "model.cam_enc.")
        else:
            name = key.removeprefix("model.")

        if "attn.qkv" in name:
            da3_split_qkv(name, tensor, writer)
            continue

        if is_conv_2d(name, tensor):
            if "patch_embeddings" in name or "head.projects" in name:
                tensor = conv_2d_to_nhwc(tensor)  # 1x1 / patch conv, kept CWHN for mul_mat
            elif "resize_layers.0" in name or "resize_layers.1" in name:
                pass  # ConvTranspose2D, don't change layout
            else:
                tensor = writer.convert_tensor_2d(tensor)

        if "position_embeddings" in name or "cls_token" in name or "camera_token" in name:
            writer.add_tensor(name, tensor, "f32")
            continue

        writer.add_tensor(name, tensor)


#
# MI-GAN


def convert_migan(input_filepath: Path, writer: Writer):
    writer.add_license("mit")
    writer.set_tensor_layout_default(TensorLayout.nchw)

    model = load_model(input_filepath)

    if "encoder.b512.fromrgb.weight" in model:
        writer.add_int32("migan.image_size", 512)
    elif "encoder.b256.fromrgb.weight" in model:
        writer.add_int32("migan.image_size", 256)

    for name, tensor in model.items():
        if is_conv_2d(name, tensor):
            tensor = writer.convert_tensor_2d(tensor)

        writer.add_tensor(name, tensor)


#
# ESRGAN


def convert_esrgan(input_filepath: Path, writer: Writer):
    from spandrel import ModelLoader

    # Load the model using spandrel
    # - it converts the various versions of ESRGAN checkpoints to a common format
    model = ModelLoader().load_from_file(input_filepath)

    if model.model.shuffle_factor is not None:
        raise ValueError("RealESRGAN models with pixel shuffle are not supported yet.")
    if getattr(model.model, "plus", False):
        raise ValueError("RealESRGAN+ (plus) models are not supported yet.")

    writer.set_tensor_layout_default(TensorLayout.nchw)
    writer.add_int32("esrgan.scale", model.scale)
    for tag in model.tags:
        if tag.endswith("nb"):
            writer.add_int32("esrgan.block_count", int(tag[:-2]))
        if tag.endswith("nf"):
            writer.add_int32("esrgan.filter_count", int(tag[:-2]))

    for name, tensor in model.model.state_dict().items():
        if is_conv_2d(name, tensor):
            tensor = writer.convert_tensor_2d(tensor)
        writer.add_tensor(name, tensor)


#
# Main
#######

arch_names = {
    "sam": "mobile-sam",
    "birefnet": "birefnet",
    "depth-anything": "depthanything",
    "depth-anything-3": "depthanything3",
    "migan": "migan",
    "esrgan": "esrgan",
}

# archs that are only ever built and shipped as F16
f16_only_archs = {"depth-anything-3"}

file_types = {None: 0, "f32": 0, "f16": 1}

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Convert model weights (.pt/.pth/.safetensors) to GGUF format.")
    parser.add_argument("arch", choices=list(arch_names.keys()), help="Model architecture")
    parser.add_argument("input", type=str, help="Path to the input model file")
    parser.add_argument("--output", "-o", type=str, default="models", help="Path to the output directory or file")
    parser.add_argument("--quantize", "-q", choices=["f16"], default=None, help="Convert float weights to the specified data type")
    parser.add_argument("--layout", "-l", choices=["whcn", "cwhn"], default=None, help="Tensor data layout for 2D operations like convolution")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--model-name", type=str, default=None, help="Name of the model for metadata")
    parser.add_argument("--metadata", type=Path, help="Specify the path for an authorship metadata override file")
    parser.add_argument("--max-blocks", type=int, default=-1, help="Debug: only convert transformer blocks [0,N); -1=all")
    # fmt: on
    args = parser.parse_args()

    if args.quantize is None and args.arch in f16_only_archs:
        args.quantize = "f16"

    input_path = Path(args.input)
    output_path = Path(args.output)
    quant_suffix = f"-{args.quantize.upper()}" if args.quantize else ""
    layout_suffix = f"-{args.layout.upper()}" if args.layout else ""
    if output_path.is_dir() or output_path.suffix != ".gguf":
        output_path = output_path / f"{input_path.stem}{quant_suffix}{layout_suffix}.gguf"

    print(f"Converting {args.arch}")
    print("* input: ", input_path)
    print("* output:", output_path)

    try:
        writer = Writer(
            output_path,
            arch_names.get(args.arch, args.arch),
            args.quantize,
            args.verbose,
        )
        metadata = Metadata.load(args.metadata, input_path.with_suffix(""), args.model_name)

        if args.layout is not None:
            writer.set_tensor_layout(TensorLayout.parse(args.layout))

        match args.arch:
            case "sam":
                convert_sam(input_path, writer)
            case "birefnet":
                convert_birefnet(input_path, writer)
            case "depthany" | "depth-anything":
                convert_depth_anything(input_path, writer)
            case "depth-anything-3":
                convert_depth_anything_3(input_path, writer, args.max_blocks)
            case "migan":
                convert_migan(input_path, writer)
            case "esrgan":
                convert_esrgan(input_path, writer)
            case _:
                raise ValueError(f"Unknown architecture: {args.arch}")

        metadata.set_gguf_meta_model(writer)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        writer.add_file_type(file_types[args.quantize])
        writer.add_conv2d_weight_indices()
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
    except ValueError as e:
        print("\033[31mError:\033[0m", e)
        exit(1)
    except Exception as e:
        print("\033[31mError:\033[0m", e)
        exit(-1)

    print("")
