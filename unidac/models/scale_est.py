import torch
import torch.nn as nn
import torch.nn.functional as F
from dac.models.backbones.metadinov3 import SelfAttentionBlock, Mlp, SwiGLUFFN
from dac.models.backbones.dinov3 import ffn_layer_dict, norm_layer_dict

def _assert_finite(name, x):
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).nonzero(as_tuple=False)
        raise RuntimeError(f"{name} has non-finite entries at {bad[:5]}")

class ScaleMapEstimation(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        num_heads: int,
        rope_embed: None,
        ffn_ratio=4,
        ffn_layer="mlp",
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path_rate: int = 0.0,
        layerscale_init: float = 1.0e-05,
        mask_k_bias: bool = False,
        norm_layer: str = "layernormbf16",
        dino_state_dict: dict = None,
        img_H: int = 224,
        img_W: int = 224,
        eps: float=1e-6,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.rope_embed = rope_embed
        self.num_heads = num_heads
        self.img_H = img_H
        self.img_W = img_W

        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        norm_layer_cls = norm_layer_dict[norm_layer]
        self.self_attn = SelfAttentionBlock(
                        dim=self.embed_dim,
                        num_heads=self.num_heads,
                        ffn_ratio=ffn_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        drop_path=drop_path_rate,
                        norm_layer=norm_layer_cls,
                        act_layer=nn.GELU,
                        ffn_layer=ffn_layer_cls,
                        init_values=layerscale_init,
                        mask_k_bias=mask_k_bias
                    )
        if dino_state_dict is not None:
            self.self_attn.load_state_dict(dino_state_dict, strict=True)
        self.mlp = self.mlp = nn.Sequential(nn.LayerNorm(self.embed_dim),
                        nn.Linear(self.embed_dim, self.embed_dim, bias=True),
                        nn.GELU(),
                        nn.Linear(self.embed_dim, self.embed_dim, bias=True),
                        nn.GELU(),
                        nn.Linear(self.embed_dim, 1, bias=True)
                    )

        self.register_buffer(
            "nb",
            torch.tensor([[0,0],[0,1],[1,0],[1,1],[0,-1],[-1,0],[-1,-1],[1,-1],[-1,1]],
                        dtype=torch.long),
            persistent=False
        )
        self.temp = 0.7
        self.nb_cache = {}

    @torch.no_grad()
    def get_nb_coords(self, img_H, img_W, device):
        h,w = img_H//self.patch_size, img_W//self.patch_size

        xx, yy = torch.meshgrid(torch.arange(img_H, device=device), torch.arange(img_W, device=device), indexing='ij')
        xy = torch.stack([xx, yy], dim=-1)//self.patch_size
        xy_nb = xy.unsqueeze(2)+self.nb[None,None,...]

        valid = (xy_nb[...,0] >= 0) & (xy_nb[...,0] < h) & (xy_nb[...,1] >= 0) & (xy_nb[...,1] < w)

        return xy_nb, valid


    def get_low_res_scale_map(self, feat_g, lat_grid=None):
        B, C, feat_h, feat_w = feat_g.shape

        if lat_grid is not None:
            lat_patch = torch.nn.functional.avg_pool2d(lat_grid, kernel_size=self.patch_size, stride=self.patch_size)
        else:
            lat_patch = None

        if self.rope_embed is not None:
            rope_sincos = self.rope_embed(H=feat_h, W=feat_w, lat_patch=lat_patch)
        else:
            rope_sincos = None
        
        x = feat_g.view(B,C,-1).permute(0,2,1).contiguous()
        out = self.self_attn(x, rope_sincos)
        out = self.mlp(out).clamp(-10,10).exp()

        scale_map_low_res = out.permute(0,2,1).view(B, 1, feat_h, feat_w)

        return scale_map_low_res

    @torch.no_grad()
    def median_pooling(self, x, kernel_size=16, stride=16, min_count=32):
        assert x.ndim == 4
        assert kernel_size == stride
        B,C,H,W = x.shape

        ks = kernel_size
        h, w = H//ks, W//ks
        x_split = x.view(B,C,h,ks,w,ks).permute(0,1,2,4,3,5).contiguous()
        x_patches = x_split.view(B,C,h,w,ks*ks)

        count_valid = torch.sum(~torch.isnan(x_patches), dim=-1)  # (B,C,h,w)
        valid_token_mask = (count_valid >= min_count).bool()
        
        median_patches = torch.nanmedian(x_patches, dim=-1).values
        median_patches = median_patches.nan_to_num(0)#.view(B,C,h,w)

        return median_patches, valid_token_mask
    
    @torch.no_grad()
    def get_dist_map(self, rel_depth, rel_depth_low_res, xy_nb):
        h,w = rel_depth_low_res.shape[-2:]
        nb_values = rel_depth_low_res[...,xy_nb[...,0].clamp(0,h-1), xy_nb[...,1].clamp(0,w-1)]
        
        z = torch.log(rel_depth[...,None].clamp(min=1e-6))
        z_nb = torch.log(nb_values.clamp(min=1e-6))
        dist_map = torch.abs(z - z_nb)

        return dist_map

    def get_final_scale_map(self, weights, scale_map_low_res, xy_nb, valid, scale_map_upsamp):
        h,w = scale_map_low_res.shape[-2:]
        scale_map_nb = scale_map_low_res[...,xy_nb[...,0].clamp(0,h-1), xy_nb[...,1].clamp(0,w-1)]

        weights_m = weights.masked_fill(~valid, 0.0)
        scale_nb_m = scale_map_nb.masked_fill(~valid, 0.0)
        scale_map_full = scale_nb_m*weights_m
        final_scale_map = torch.where(valid.any(dim=-1), scale_map_full.sum(dim=-1), scale_map_upsamp)

        return final_scale_map

    def forward(self, feat_g, rel_depth, mask=None, lat_grid=None):
        img_H, img_W = rel_depth.shape[-2:]

        key = (img_H, img_W, self.patch_size, rel_depth.device)
        if key not in self.nb_cache:
            xy_nb, valid = self.get_nb_coords(img_H, img_W, device=rel_depth.device)
            self.nb_cache[key] = (xy_nb, valid)
        else:
            xy_nb, valid = self.nb_cache[key]


        valid = valid[None,None,...]
        scale_map_low_res = self.get_low_res_scale_map(feat_g, lat_grid=lat_grid)
        _assert_finite("scale_map_low_res", scale_map_low_res)

        # Get median pooled depth
        if mask is not None:
            rel_depth_nan = torch.where(mask.bool(), rel_depth, torch.nan)
        else:
            rel_depth_nan = rel_depth

        rel_depth_low_res, valid_token_mask = self.median_pooling(rel_depth_nan, kernel_size=self.patch_size, stride=self.patch_size, min_count=64)
        h,w = valid_token_mask.shape[-2:]
        valid_nb = valid_token_mask[..., xy_nb[...,0].clamp(0,h-1), xy_nb[...,1].clamp(0,w-1)]

        # Get distance_map and weights
        dist_map = self.get_dist_map(rel_depth.detach(), rel_depth_low_res, xy_nb)
        if mask is not None:
            valid_final = valid & valid_nb & mask.bool()[...,None]
        else:
            valid_final = valid & valid_nb
        _assert_finite("rel_depth", rel_depth)
        _assert_finite("rel_depth_low_res", rel_depth_low_res)
        _assert_finite("dist_map", dist_map)

        with torch.no_grad():
            valid_dist_map = torch.where(valid_final, dist_map, 1e9)
            logits = -valid_dist_map / self.temp
            # logits = -(dist_map / self.temp)
            # logits = torch.where(valid_final, logits, torch.full_like(logits, float('-inf')))
            logits = logits - logits.amax(dim=-1, keepdim=True)
            weights = torch.nn.functional.softmax(logits, dim=-1).detach()
        # _assert_finite("weights", weights)

        # Get scale_map_full
        scale_map = self.get_final_scale_map(weights, scale_map_low_res, xy_nb, valid_final, 0)#scale_map_upsamp)
        return scale_map