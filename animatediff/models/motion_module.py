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
from animatediff.models.attention_processor import PoseAdaptorAttnProcessor

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


def get_motion_module(
        in_channels,
        motion_module_type: str,
        motion_module_kwargs: dict
):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(in_channels=in_channels, **motion_module_kwargs)
    elif motion_module_type == "Conv":
        return ConvTemporalModule(in_channels=in_channels, **motion_module_kwargs)
    else:
        raise ValueError


class ConvTemporalModule(nn.Module):
    def __init__(
            self,
            in_channels,
            temporal_kernel_size=3,
            num_resnet_blocks=1,
            temporal_position_encoding=True,
            temporal_position_encoding_max_len=32,
    ):
        super().__init__()

        resnets = []
        for i in range(num_resnet_blocks):
            resnets.append(
                TemporalResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temporal_kernel_size=temporal_kernel_size,
                    conv_shortcut=False,
                    temb_channels=1280,
                    groups=32,
                    non_linearity="silu",
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.pos_encoder = PositionalEncoding(
            in_channels,
            max_len=temporal_position_encoding_max_len
        ) if temporal_position_encoding else None

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None):
        # hidden_states: (b c f h w)
        batch_size, channels, video_length, height, width = hidden_states.shape

        if self.pos_encoder is not None:
            hidden_states = rearrange(hidden_states, "b c f h w -> (b h w) f c")
            hidden_states = self.pos_encoder(hidden_states)
            hidden_states = rearrange(hidden_states, "(b h w) f c -> b c f h w", b=batch_size, h=height, w=width)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        return hidden_states


class TemporalResnetBlock(nn.Module):
    def __init__(
            self,
            *,
            in_channels,
            out_channels=None,
            temporal_kernel_size=3,
            conv_shortcut=False,
            dropout=0.0,
            temb_channels=512,
            groups=32,
            groups_out=None,
            pre_norm=True,
            eps=1e-6,
            non_linearity="swish",
            time_embedding_norm="default",
            output_scale_factor=1.0,
            use_in_shortcut=None,

            zero_initialize=True,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        padding_size = (temporal_kernel_size - 1) // 2
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(temporal_kernel_size, 1, 1), stride=1,
                               padding=(padding_size, 0, 0))

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            # elif self.time_embedding_norm == "scale_shift":
            #     time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)

        conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(temporal_kernel_size, 1, 1), stride=1,
                          padding=(padding_size, 0, 0))
        if zero_initialize:
            self.conv2 = zero_module(conv2)
        else:
            self.conv2 = conv2

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = Conv3d(in_channels, out_channels, kernel_size=(temporal_kernel_size, 1, 1), stride=1,
                                        padding=(padding_size, 0, 0))

    def forward(self, input_tensor, temb):
        # input_tensor: (b c f h w)
        # temb:         (b emb_dim)

        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        # if temb is not None and self.time_embedding_norm == "scale_shift":
        #     scale, shift = torch.chunk(temb, 2, dim=1)
        #     hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class VanillaTemporalModule(nn.Module):
    def __init__(
            self,
            in_channels,
            num_attention_heads=8,
            num_transformer_block=2,
            attention_block_types=("Temporal_Self",),
            temporal_position_encoding=True,
            temporal_position_encoding_max_len=32,
            temporal_attention_dim_div=1,
            cross_attention_dim=320,
            zero_initialize=True,
            encoder_hidden_states_query=(False, False),
            attention_activation_scale=1.0,
            attention_processor_kwargs: Dict = {},
            causal_temporal_attention=False,
            causal_temporal_attention_mask_type="",
            rescale_output_factor=1.0
    ):
        super().__init__()

        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_attention_dim=cross_attention_dim,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
            encoder_hidden_states_query=encoder_hidden_states_query,
            attention_activation_scale=attention_activation_scale,
            attention_processor_kwargs=attention_processor_kwargs,
            causal_temporal_attention=causal_temporal_attention,
            causal_temporal_attention_mask_type=causal_temporal_attention_mask_type,
            rescale_output_factor=rescale_output_factor
        )

        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None,
                cross_attention_kwargs: Dict[str, Any] = {}):
        hidden_states = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask, cross_attention_kwargs=cross_attention_kwargs)

        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    def __init__(
            self,
            in_channels,
            num_attention_heads,
            attention_head_dim,
            num_layers,
            attention_block_types=("Temporal_Self", "Temporal_Self",),
            dropout=0.0,
            norm_num_groups=32,
            cross_attention_dim=320,
            activation_fn="geglu",
            attention_bias=False,
            upcast_attention=False,
            temporal_position_encoding=False,
            temporal_position_encoding_max_len=32,
            encoder_hidden_states_query=(False, False),
            attention_activation_scale=1.0,
            attention_processor_kwargs: Dict = {},

            causal_temporal_attention=None,
            causal_temporal_attention_mask_type="",
            rescale_output_factor=1.0
    ):
        super().__init__()
        assert causal_temporal_attention is not None
        self.causal_temporal_attention = causal_temporal_attention

        assert (not causal_temporal_attention) or (causal_temporal_attention_mask_type != "")
        self.causal_temporal_attention_mask_type = causal_temporal_attention_mask_type
        self.causal_temporal_attention_mask = None

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = InflatedGroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
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
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    encoder_hidden_states_query=encoder_hidden_states_query,
                    attention_activation_scale=attention_activation_scale,
                    attention_processor_kwargs=attention_processor_kwargs,
                    rescale_output_factor=rescale_output_factor,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def get_causal_temporal_attention_mask(self, hidden_states):
        batch_size, sequence_length, dim = hidden_states.shape

        if self.causal_temporal_attention_mask is None or self.causal_temporal_attention_mask.shape != (
        batch_size, sequence_length, sequence_length):
            if self.causal_temporal_attention_mask_type == "causal":
                # 1. vanilla causal mask
                mask = torch.tril(torch.ones(sequence_length, sequence_length))

            elif self.causal_temporal_attention_mask_type == "2-seq":
                # 2. 2-seq
                mask = torch.zeros(sequence_length, sequence_length)
                mask[:sequence_length // 2, :sequence_length // 2] = 1
                mask[-sequence_length // 2:, -sequence_length // 2:] = 1

            elif self.causal_temporal_attention_mask_type == "0-prev":
                # attn to the previous frame
                indices = torch.arange(sequence_length)
                indices_prev = indices - 1
                indices_prev[0] = 0
                mask = torch.zeros(sequence_length, sequence_length)
                mask[:, 0] = 1.
                mask[indices, indices_prev] = 1.

            elif self.causal_temporal_attention_mask_type == "0":
                # only attn to first frame
                mask = torch.zeros(sequence_length, sequence_length)
                mask[:, 0] = 1

            elif self.causal_temporal_attention_mask_type == "wo-self":
                indices = torch.arange(sequence_length)
                mask = torch.ones(sequence_length, sequence_length)
                mask[indices, indices] = 0

            elif self.causal_temporal_attention_mask_type == "circle":
                indices = torch.arange(sequence_length)
                indices_prev = indices - 1
                indices_prev[0] = 0

                mask = torch.eye(sequence_length)
                mask[indices, indices_prev] = 1
                mask[0, -1] = 1

            else:
                raise ValueError

            # generate attention mask fron binary values
            mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            mask = mask.unsqueeze(0)
            mask = mask.repeat(batch_size, 1, 1)

            self.causal_temporal_attention_mask = mask.to(hidden_states.device)

        return self.causal_temporal_attention_mask

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None,
                cross_attention_kwargs: Dict[str, Any] = {},):
        residual = hidden_states

        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        height, width = hidden_states.shape[-2:]

        hidden_states = self.norm(hidden_states)
        hidden_states = rearrange(hidden_states, "b c f h w -> (b h w) f c")
        hidden_states = self.proj_in(hidden_states)

        attention_mask = self.get_causal_temporal_attention_mask(
            hidden_states) if self.causal_temporal_attention else attention_mask

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states,
                                  attention_mask=attention_mask, cross_attention_kwargs=cross_attention_kwargs)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = rearrange(hidden_states, "(b h w) f c -> b c f h w", h=height, w=width)

        output = hidden_states + residual

        return output


class TemporalTransformerBlock(nn.Module):
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
            temporal_position_encoding=False,
            temporal_position_encoding_max_len=32,
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
                TemporalSelfAttention(
                    attention_mode=block_name,
                    cross_attention_dim=cross_attention_dim if block_name in ['Temporal_Cross', 'Temporal_Pose_Adaptor'] else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    rescale_output_factor=rescale_output_factor,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, cross_attention_kwargs: Dict[str, Any] = {}):
        for attention_block, norm, attention_block_type in zip(self.attention_blocks, self.norms, self.attention_block_types):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=norm_hidden_states if attention_block_type == 'Temporal_Self' else encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs
            ) + hidden_states

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model,
            dropout=0.,
            max_len=32,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalSelfAttention(Attention):
    def __init__(
            self,
            attention_mode=None,
            temporal_position_encoding=False,
            temporal_position_encoding_max_len=32,
            rescale_output_factor=1.0,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal_Self"

        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            max_len=temporal_position_encoding_max_len
        ) if temporal_position_encoding else None
        self.rescale_output_factor = rescale_output_factor

    def set_use_memory_efficient_attention_xformers(
            self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ):
        # disable motion module efficient xformers to avoid bad results, don't know why
        # TODO: fix this bug
        pass

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        # add position encoding
        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)
        if "pose_feature" in cross_attention_kwargs and cross_attention_kwargs["pose_feature"] is not None:
            pose_feature = cross_attention_kwargs["pose_feature"]
            if pose_feature.ndim == 5:
                pose_feature = rearrange(pose_feature, "b c f h w -> (b h w) f c")
            else:
                assert pose_feature.ndim == 3
            cross_attention_kwargs["pose_feature"] = pose_feature

        if isinstance(self.processor,  PoseAdaptorAttnProcessor):
            return self.processor(
                self,
                hidden_states,
                cross_attention_kwargs.pop('pose_feature'),
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
        elif hasattr(self.processor, "__call__"):
            return self.processor.__call__(
                    self,
                    hidden_states,
                    encoder_hidden_states=None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
        else:
            return self.processor(
                self,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

