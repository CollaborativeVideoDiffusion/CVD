from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import BaseOutput
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward

from animatediff.models.resnet import InflatedGroupNorm
from typing import Dict, Any
from animatediff.models.attention_processor import PoseAdaptorAttnProcessor, EpiAttnProcessor

from einops import rearrange
import math


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


def get_epi_module(
        in_channels,
        epi_module_kwargs: dict
):
    return EpiModule(in_channels=in_channels, **epi_module_kwargs)

class EpiModule(nn.Module):
    def __init__(
            self,
            in_channels,
            num_attention_heads=8,
            num_transformer_block=2,
            attention_block_types=("Epi_Self",),
            epi_position_encoding              = True,
            epi_position_encoding_feat_max_size= 64,
            epi_position_encoding_F_mat_size   = 256,
            epi_no_attention_mask              = False,
            epi_mono_direction = False, 
            epi_fix_firstframe = False,
            epi_rand_slope_ff = False, 
            cross_attention_dim=320,
            zero_initialize=True,
            encoder_hidden_states_query=(False, False),
            attention_activation_scale=1.0,
            attention_processor_kwargs: Dict = {},
            rescale_output_factor=1.0
    ):
        super().__init__()

        self.epi_transformer = EpiTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_attention_dim=cross_attention_dim,
            epi_position_encoding=epi_position_encoding,
            epi_position_encoding_feat_max_size=epi_position_encoding_feat_max_size,
            epi_position_encoding_F_mat_size=epi_position_encoding_F_mat_size,
            epi_no_attention_mask = epi_no_attention_mask,
            epi_mono_direction=epi_mono_direction,
            epi_fix_firstframe=epi_fix_firstframe,
            epi_rand_slope_ff=epi_rand_slope_ff,
            encoder_hidden_states_query=encoder_hidden_states_query,
            attention_activation_scale=attention_activation_scale,
            attention_processor_kwargs=attention_processor_kwargs,
            rescale_output_factor=rescale_output_factor
        )

        if zero_initialize:
            self.epi_transformer.proj_out = zero_module(self.epi_transformer.proj_out)

    def forward(self, hidden_states, F_mats=None, H_mats=None, temb=None, encoder_hidden_states=None, attention_mask=None,
                cross_attention_kwargs: Dict[str, Any] = {}):
        hidden_states, aux = self.epi_transformer(hidden_states, F_mats, H_mats, encoder_hidden_states, attention_mask, cross_attention_kwargs=cross_attention_kwargs)

        output = hidden_states
        return output, aux


class EpiTransformer3DModel(nn.Module):
    def __init__(
            self,
            in_channels,
            num_attention_heads,
            attention_head_dim,
            num_layers,
            attention_block_types=("Epi_Self",),
            dropout=0.0,
            norm_num_groups=32,
            cross_attention_dim=320,
            activation_fn="geglu",
            attention_bias=False,
            upcast_attention=False,
            epi_position_encoding=False,
            epi_position_encoding_feat_max_size=32,
            epi_position_encoding_F_mat_size=256,
            epi_no_attention_mask=False,
            epi_mono_direction=False,
            epi_fix_firstframe=False,
            epi_rand_slope_ff=False,
            encoder_hidden_states_query=(False, False),
            attention_activation_scale=1.0,
            attention_processor_kwargs: Dict = {},

            rescale_output_factor=1.0
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = InflatedGroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                EpiTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    epi_position_encoding=epi_position_encoding,
                    epi_position_encoding_feat_max_size=epi_position_encoding_feat_max_size,
                    epi_position_encoding_F_mat_size=epi_position_encoding_F_mat_size,
                    encoder_hidden_states_query=encoder_hidden_states_query,
                    epi_no_attention_mask=epi_no_attention_mask,
                    epi_mono_direction=epi_mono_direction,
                    epi_fix_firstframe=epi_fix_firstframe,
                    epi_rand_slope_ff=epi_rand_slope_ff,
                    attention_activation_scale=attention_activation_scale,
                    attention_processor_kwargs=attention_processor_kwargs,
                    rescale_output_factor=rescale_output_factor,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)


    def forward(self, hidden_states, F_mats=None, H_mats=None, encoder_hidden_states=None, attention_mask=None,
                cross_attention_kwargs: Dict[str, Any] = {},):
        residual = hidden_states

        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length, height, width = hidden_states.shape[-3:]
        hidden_states = self.norm(hidden_states)
        # hidden_states = rearrange(hidden_states, "b c f h w -> (b h w) f c")
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) (h w) c")
        if F_mats is not None:
            if isinstance(F_mats, torch.Tensor):
                F_mats = rearrange(F_mats, "b f h w -> (b f) h w")
            else:
                F_mats = [rearrange(F_mats[0], "b f h w -> (b f) h w"), F_mats[1]]
        if H_mats is not None:
            H_mats = rearrange(H_mats, "b f h w -> (b f) h w")
                     
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        additional_outputs = []
        for block in self.transformer_blocks:
            hidden_states, aux = block(hidden_states, F_mats, H_mats, encoder_hidden_states=encoder_hidden_states,
                                  attention_mask=attention_mask, cross_attention_kwargs=cross_attention_kwargs)
            additional_outputs += aux
        hidden_states = self.proj_out(hidden_states)

        # hidden_states = rearrange(hidden_states, "(b h w) f c -> b c f h w", h=height, w=width)
        hidden_states = rearrange(hidden_states, "(b f) (h w) c -> b c f h w", f=video_length, h=height, w=width)

        output = hidden_states + residual
        return output, additional_outputs


class EpiTransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_attention_heads,
            attention_head_dim,
            attention_block_types=("Temporal_Self", "Temporal_Self",),
            dropout=0.0,
            norm_num_groups=32,
            cross_attention_dim=768,
            activation_fn="geglu",
            attention_bias=False,
            upcast_attention=False,
            epi_position_encoding=False,
            epi_position_encoding_feat_max_size=32,
            epi_position_encoding_F_mat_size=256,
            epi_no_attention_mask=False,
            epi_mono_direction=False,
            epi_fix_firstframe=False,
            epi_rand_slope_ff=False,
            encoder_hidden_states_query=(False, False),
            attention_activation_scale=1.0,
            attention_processor_kwargs: Dict = {},
            rescale_output_factor=1.0
    ):
        super().__init__()

        attention_blocks = []
        norms = []
        self.attention_block_types = attention_block_types

        for block_idx, block_name in enumerate(attention_block_types):
            attention_blocks.append(
                EpiSelfAttention(
                    attention_mode=block_name,
                    cross_attention_dim=None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    epi_position_encoding=epi_position_encoding,
                    epi_position_encoding_feat_max_size=epi_position_encoding_feat_max_size,
                    epi_position_encoding_F_mat_size=epi_position_encoding_F_mat_size,
                    epi_no_attention_mask=epi_no_attention_mask,
                    epi_mono_direction=epi_mono_direction,
                    epi_fix_firstframe=epi_fix_firstframe,
                    epi_rand_slope_ff=epi_rand_slope_ff,
                    rescale_output_factor=rescale_output_factor,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states, F_mats=None, H_mats=None, encoder_hidden_states=None, attention_mask=None, cross_attention_kwargs: Dict[str, Any] = {}):
        additional_outputs = []
        for attention_block, norm, attention_block_type in zip(self.attention_blocks, self.norms, self.attention_block_types):
            norm_hidden_states = norm(hidden_states)
            res, aux = attention_block(
                norm_hidden_states,
                F_mats=F_mats,
                H_mats=H_mats, 
                encoder_hidden_states=norm_hidden_states if attention_block_type == 'Temporal_Self' else encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs
            ) 
            hidden_states = hidden_states + res
            additional_outputs.append(aux)

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output, additional_outputs

class EpiEncoding(nn.Module):
    def __init__(
        self, 
        d_model, 
        dropout = 0., 
        max_feat_size = 128,
        F_mat_size = 256, 
        rand_slope_on_first_frame = False, 
    ):
        super().__init__()
        self.F_mat_size = F_mat_size
        self.rand_slope_on_first_frame = rand_slope_on_first_frame
        self.dropout = nn.Dropout(p=dropout)
        coords = torch.arange(max_feat_size)
        coords_x, coords_y = torch.meshgrid(coords, coords, indexing='xy') 
        coords = torch.stack([coords_x, coords_y, coords_x*0+1], dim=-1) # 64 x 64 x 3

        self.register_buffer('coords', coords)

    def forward(self, x, F_mats=None):
        pass 

    def get_attn_map(self, x, F_mats=None, H_mats=None, pixel_band=3, decay_alpha=3):
        feat_size = int(x.shape[1] ** 0.5)
        
        selected_coords = self.coords[:feat_size, :feat_size].reshape(-1, 3) 
        # Rescale pixel coordinates to where the F matrix is defined
        coords = ((self.F_mat_size / feat_size) * selected_coords + (self.F_mat_size / feat_size-1) / 2)[None] # 1 x feat_size^2 x 3
        coords[..., -1] = 1
    
        if H_mats is not None: 
            # Get F_coords by homography transformation
            # In case where H_mats is given, pseodo epipolar lines are generated 
            batch_size = H_mats.shape[0]
            H_coords = coords.repeat(batch_size, 1, 1) # B x feat_size^2 x 3
            H_coords[...,:2] = H_coords[...,:2] - (self.F_mat_size-1) / 2
            H_coords = torch.bmm(H_mats.float(), H_coords.permute(0, 2, 1)).permute(0, 2, 1)
            H_coords = H_coords / (H_coords[...,2:]+1e-6)
            H_coords[...,:2] = H_coords[...,:2] + (self.F_mat_size-1) / 2
            F_coords = self.get_pseudo_F_coords(H_coords, random_slope=True)
        elif F_mats is not None:
            # Get F_coords by epipolar transformation
            batch_size = F_mats.shape[0]
            F_coords = coords.repeat(batch_size, 1, 1)
            F_coords = torch.bmm(F_mats.float(), F_coords.float().permute(0, 2, 1)).permute(0, 2, 1) # B x feat_size^2 x 3
            F_coords[::16] = self.get_pseudo_F_coords(coords[::16], random_slope=self.rand_slope_on_first_frame)
        else:
            # Get F_coords by identity transformation
            batch_size = x.shape[0]
            F_coords = self.get_pseudo_F_coords(coords.repeat(batch_size, 1, 1), random_slope=True)
            
        ab_norm = (F_coords[:, :, :2] * F_coords[:, :, :2]).sum(-1).sqrt()[:, :, None] 
        cFc = torch.bmm(F_coords, coords.repeat(batch_size, 1, 1).permute(0, 2, 1)).abs()
        cFc = cFc / (ab_norm+1e-6)
        normed_pixel_band = (pixel_band / (self.F_mat_size // 2) * cFc.reshape(cFc.shape[0], -1).max(dim=-1)[0])[:, None, None]
        map_weight_decay = decay_alpha / (normed_pixel_band+1e-6)
        attn_mask = - (cFc-normed_pixel_band).clip(0) * map_weight_decay # B x feat_size^2 x feat_size^2
        # attn_mask = 1. - torch.sigmoid(50. * (cFc/256. - 0.01))
        return attn_mask.detach()

    def get_pseudo_F_coords(self, coords, random_slope=False):
        feat_size = int(coords.shape[1] ** 0.5)
        batch_size = coords.shape[0]
        if random_slope is True:
            slope = torch.rand([batch_size], device=coords.device) * math.pi
            F_coords_a = torch.cos(slope)[:, None, None].repeat(1, feat_size**2, 1)
            F_coords_b = torch.sin(slope)[:, None, None].repeat(1, feat_size**2, 1)
            F_coords_c = -(F_coords_a * coords[...,0:1] + F_coords_b * coords[...,1:2])
            F_coords = torch.cat([F_coords_a, F_coords_b, F_coords_c], dim=-1)
        else:
            F_coords_a = torch.zeros([1, feat_size**2, 1], device=coords.device).repeat(batch_size, 1, 1)
            F_coords_b = -torch.ones([1, feat_size**2, 1], device=coords.device).repeat(batch_size, 1, 1)
            F_coords_c = coords[...,1:2]

            F_coords = torch.cat([F_coords_a, F_coords_b, F_coords_c], dim=-1)
        return F_coords
    


class EpiSelfAttention(Attention):
    def __init__(
            self,
            attention_mode=None,
            epi_position_encoding=False,
            epi_position_encoding_feat_max_size=32,
            epi_position_encoding_F_mat_size=256,
            epi_no_attention_mask=False,
            epi_mono_direction=False,
            epi_fix_firstframe=False,
            epi_rand_slope_ff=False,
            rescale_output_factor=1.0,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Epi_Self"

        self.pos_encoder = EpiEncoding(
            kwargs["query_dim"],
            dropout=0., 
            max_feat_size=epi_position_encoding_feat_max_size,
            F_mat_size=epi_position_encoding_F_mat_size,
            rand_slope_on_first_frame=epi_rand_slope_ff
        ) if epi_position_encoding else None
        self.rescale_output_factor = rescale_output_factor
        self.epi_no_attention_mask = epi_no_attention_mask
        self.epi_mono_direction = epi_mono_direction
        self.epi_fix_firstframe = epi_fix_firstframe

    def set_use_memory_efficient_attention_xformers(
            self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ):
        # disable motion module efficient xformers to avoid bad results, don't know why
        # TODO: fix this bug
        pass

    def forward(self, hidden_states, F_mats=None, H_mats=None, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        # add position encoding
        if self.pos_encoder is not None and not self.epi_no_attention_mask:
            with torch.no_grad():
                # hidden_states = self.pos_encoder(hidden_states)
                attention_mask = self.pos_encoder.get_attn_map(hidden_states, 
                                                            F_mats = F_mats[0] if isinstance(F_mats, list) else F_mats,
                                                            H_mats = H_mats)
                if attention_mask.shape[0] != hidden_states.shape[0]:
                    assert attention_mask.shape[0] % hidden_states.shape[0] == 0
                    B, N, C = hidden_states.shape
                    attention_mask = attention_mask.reshape(-1, B, N, N)
                    attention_mask = attention_mask.permute(1, 2, 3, 0).reshape(B, N, -1)

            if torch.isnan(attention_mask).any():
                print("attention_mask contains NaN")

            torch.nan_to_num(attention_mask, nan=0.0, posinf=0.0, neginf=0.0, out=attention_mask)

        # if attention_mask.shape[1] == 1024:
        #     import pdb
        #     pdb.set_trace()
        #     torch.save(attention_mask.detach().cpu(), "temp_attention_mask_for_debug.pt")
        # attention_mask=None

        # if "pose_feature" in cross_attention_kwargs:
        #     pose_feature = cross_attention_kwargs["pose_feature"]
        #     if pose_feature.ndim == 5:
        #         pose_feature = rearrange(pose_feature, "b c f h w -> (b h w) f c")
        #     else:
        #         assert pose_feature.ndim == 3
        #     cross_attention_kwargs["pose_feature"] = pose_feature

        assert isinstance(self.processor, EpiAttnProcessor)
        if attention_mask is not None and (attention_mask.shape[0] > 200 or attention_mask.shape[-1] > 2048): # memorrrrrrrry......
            bs = hidden_states.shape[0]
            chunk_num = 128 * 1024 // max(1024, attention_mask.shape[-1])
            # hidden_states_0, hidden_states_1 = hidden_states.chunk(2)
            if isinstance(F_mats, list):
                kv_index=F_mats[1]
                encoder_hidden_states = hidden_states[kv_index]
                # encoder_hidden_states_0, encoder_hidden_states_1 = encoder_hidden_states.chunk(2)
            else:
                encoder_hidden_states = hidden_states
                # encoder_hidden_states_0, encoder_hidden_states_1 = hidden_states_1, hidden_states_0
            hid_list = []
            for i in range(bs // chunk_num):
                st, ed = i*chunk_num, (i+1)*chunk_num
                hid_list.append(self.processor(
                        self,
                        hidden_states[st:ed],
                        encoder_hidden_states=encoder_hidden_states[st:ed],
                        attention_mask=attention_mask[st:ed],
                        kv_index=None, # no need to input kv_index here since encoder_hidden_states is assigned 
                        mono_direction=self.epi_mono_direction,
                        fix_firstframe=self.epi_fix_firstframe,
                        **cross_attention_kwargs,
                    )[0])
            torch.cuda.empty_cache()
            hidden_states = torch.cat(hid_list, dim=0)
            aux = None # {k:torch.cat([hid_0[1][k], hid_1[1][k]], dim=0) for k in hid_0[1].keys()}
            return hidden_states, aux
        else:
            return self.processor(
                self,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                kv_index=F_mats[1] if isinstance(F_mats, list) else None,
                mono_direction=self.epi_mono_direction,
                fix_firstframe=self.epi_fix_firstframe,
                **cross_attention_kwargs,
            )
