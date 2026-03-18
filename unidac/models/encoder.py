from functools import partial
from enum import Enum
from typing import Any, Union

import torch
from timm.models.vision_transformer import _cfg
from torch import nn

from unidac.models.backbones import (Bottleneck, EfficientNet, ResNet,
                                    SwinTransformer, _resnet, _make_dinov2_model, _make_dinov3_vit, Weights)


def swin_tiny(pretrained: bool = True, **kwargs):
    if pretrained:
        pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
    model = SwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dims=[96 * (2**i) for i in range(4)],
        num_heads=[3, 6, 12, 24],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 6, 2],
        drop_path_rate=0.2,
        pretrained=pretrained,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


def swin_small(pretrained: bool = True, **kwargs):
    if pretrained:
        pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth"
    model = SwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dims=[96 * (2**i) for i in range(4)],
        num_heads=[3, 6, 12, 24],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 18, 2],
        drop_path_rate=0.3,
        pretrained=pretrained,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


def swin_base(pretrained: bool = True, **kwargs):
    if pretrained:
        pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth"
    model = SwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dims=[128 * (2**i) for i in range(4)],
        num_heads=[4, 8, 16, 32],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 18, 2],
        drop_path_rate=0.5,
        pretrained=pretrained,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


def swin_large_22k(pretrained: bool = True, **kwargs):
    if pretrained:
        pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth"
    model = SwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dims=[192 * (2**i) for i in range(4)],
        num_heads=[6, 12, 24, 48],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 18, 2],
        drop_path_rate=0.2,
        pretrained=pretrained,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

# def swinv2_large_22k(pretrained: bool = True, **kwargs):
#     if pretrained:
#         pretrained = "https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth"
#     model = SwinTransformerV2(
#         window_size=12,
#         embed_dims=[192 * (2**i) for i in range(4)],
#         num_heads=[ 6, 12, 24, 48 ],
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=[2, 2, 18, 2],
#         drop_path_rate=0.2,
#         pretrained=pretrained,
#         **kwargs
#     )
#     model.default_cfg = _cfg()
#     return model


def resnet50(pretrained: bool = True, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = True, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def efficientnet_b5(pretrained: bool = True, **kwargs):
    basemodel_name = "tf_efficientnet_b5_ap"
    basemodel = torch.hub.load(
        "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=pretrained
    )
    basemodel.global_pool = nn.Identity()
    basemodel.classifier = nn.Identity()
    return EfficientNet(basemodel, [5, 6, 8, 15])  # 11->15


def dinov2_vits14(config, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    vit = _make_dinov2_model(
        arch_name="vit_small",
        pretrained=config["pretrained"],
        img_size=tuple(config.get("img_size", 518)),
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit


def dinov2_vitb14(config, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-B/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    vit = _make_dinov2_model(
        arch_name="vit_base",
        pretrained=config["pretrained"],
        img_size=tuple(config.get("img_size", 518)),
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit


def dinov2_vitl14(config, pretrained: str = "", **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    vit = _make_dinov2_model(
        arch_name="vit_large",
        pretrained=config["pretrained"],
        img_size=tuple(config.get("img_size", 518)),
        output_idx=config.get("output_idx", [5, 12, 18, 24]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit

Weights = Enum("Weights", ["LVD1689M", "SAT493M"])
def dinov3_vits16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "08c60483"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vits",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vits16plus(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "4057cbaa"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=6,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitsplus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitb16(
    config,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "73cec8be"
    kwargs["version"] = None
    return _make_dinov3_vit(
        pretrained=config["pretrained"],
        img_size=tuple(config.get("img_size", 512)),
        pos_embed_rope_rescale_coords=2,
        drop_path_rate=config.get("drop_path", 0.0),
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        n_storage_tokens=config.get("num_storage_tokens", 4),
        mask_k_bias=True,
        weights=config.get("weights", weights),
        compact_arch_name="vitb",
        check_hash=check_hash,
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        use_norm=config.get("use_norm", False),
        **kwargs,
    )


def dinov3_vitl16(
    config,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    untie_global_and_local_cls_norm = False
    if weights == Weights.LVD1689M:
        if "hash" not in kwargs:
            kwargs["hash"] = "8aa4cbdd"
    elif weights == Weights.SAT493M:
        if "hash" not in kwargs:
            kwargs["hash"] = "eadcf0ff"
        untie_global_and_local_cls_norm = True
    elif type(weights) is str:
        import re

        pattern = r"-(.{8}).pth"
        matches = re.findall(pattern, weights)
        if len(matches) != 1:
            raise ValueError(f"Unexpected weights specification for the ViT-L backbone: {weights}")
        hash = matches[0]
        if hash == "eadcf0ff":
            untie_global_and_local_cls_norm = True
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=tuple(config.get("img_size", 512)),
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=config.get("drop_path", 0.0),
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=config.get("num_storage_tokens", 4),
        mask_k_bias=True,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
        pretrained=config["pretrained"],
        weights=config.get("weights", weights),
        compact_arch_name="vitl",
        check_hash=check_hash,
        output_idx=config.get("output_idx", [6, 12, 18, 24]),
        use_norm=config.get("use_norm", False),
        **kwargs,
    )


def dinov3_vitl16plus(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "46503df0"

    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitlplus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vith16plus(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "7c1da9a5"

    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vithplus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vit7b16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if weights == Weights.LVD1689M:
        if "hash" not in kwargs:
            kwargs["hash"] = "a955f4ea"
    elif weights == Weights.SAT493M:
        if "hash" not in kwargs:
            kwargs["hash"] = "a6675841"
    kwargs["version"] = None
    untie_global_and_local_cls_norm = True
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        qkv_bias=False,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu64",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vit7b",
        check_hash=check_hash,
        **kwargs,
    )