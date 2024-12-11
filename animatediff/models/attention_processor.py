import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import logging
from diffusers.models.lora import LoRALinearLayer
from diffusers.models.attention import Attention
from diffusers.utils import USE_PEFT_BACKEND
from typing import Optional

from einops import rearrange
import numpy as np

logger = logging.getLogger(__name__)


class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(self, attn_type=None, spatial_extended_attention=False):
        self.attn_type = attn_type
        self.spatial_extended_attention = spatial_extended_attention

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            pose_feature=None
    ) -> torch.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        if self.spatial_extended_attention:
            if self.attn_type == "spatial" and not key.shape[1] == 77: 
                # print("extended attention")
                if key.shape[0] > 32: # TODO: replace this ugly change
                    bs = key.shape[0] // 32 
                    key = key.chunk(bs)
                    key = torch.cat(key, dim=1).repeat(bs, 1, 1)
                    value = value.chunk(bs)
                    value = torch.cat(value, dim=1).repeat(bs, 1, 1)
                else:
                    key = key.chunk(2)
                    key = torch.cat([key[0], key[1]], dim=1).repeat(2, 1, 1)
                    value = value.chunk(2)
                    value = torch.cat([value[0], value[1]], dim=1).repeat(2, 1, 1)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class LoRAAttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
            self,
            hidden_size=None,
            cross_attention_dim=None,
            rank=4,
            network_alpha=None,
            lora_scale=1.0,
            spatial_extended_attention=False,
    ):
        super().__init__()
        self.rank = rank
        self.lora_scale = lora_scale

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.spatial_extended_attention = spatial_extended_attention

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            pose_feature=None,
            scale=None
    ):
        lora_scale = self.lora_scale if scale is None else scale
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) + lora_scale * self.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + lora_scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + lora_scale * self.to_v_lora(encoder_hidden_states)

        if self.spatial_extended_attention: 
            # Automatically treat videos as pairs
            if key.shape[1] != 77:
                if key.shape[0] > 32: # TODO: replace this ugly change
                    bs = key.shape[0] // 32 
                    key = key.chunk(bs)
                    key = torch.cat(key, dim=1).repeat(bs, 1, 1)
                    value = value.chunk(bs)
                    value = torch.cat(value, dim=1).repeat(bs, 1, 1)
                else:
                    key = key.chunk(2)
                    key = torch.cat([key[0], key[1]], dim=1).repeat(2, 1, 1)
                    value = value.chunk(2)
                    value = torch.cat([value[0], value[1]], dim=1).repeat(2, 1, 1)

        # query = attn.head_to_batch_dim(query)
        # key = attn.head_to_batch_dim(key)
        # value = attn.head_to_batch_dim(value)
        # attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # hidden_states = torch.bmm(attention_probs, value)
        # hidden_states = attn.batch_to_head_dim(hidden_states)
                
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + lora_scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class PoseAdaptorAttnProcessor(nn.Module):
    def __init__(self,
                 hidden_size,  # dimension of hidden state
                 pose_feature_dim=None,  # dimension of the pose feature
                 cross_attention_dim=None,  # dimension of the text embedding
                 query_condition=False,
                 key_value_condition=False,
                 scale=1.0,
                # sync lora keywords
                 sync_lora_rank=0,
                 network_alpha=None,
                 sync_lora_scale=0,):
        super().__init__()
        self.hidden_size = hidden_size
        self.pose_feature_dim = pose_feature_dim
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.query_condition = query_condition
        self.key_value_condition = key_value_condition
        assert hidden_size == pose_feature_dim
        if self.query_condition and self.key_value_condition:
            self.qkv_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.qkv_merge.weight)
            init.zeros_(self.qkv_merge.bias)
        elif self.query_condition:
            self.q_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.q_merge.weight)
            init.zeros_(self.q_merge.bias)
        else:
            self.kv_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.kv_merge.weight)
            init.zeros_(self.kv_merge.bias)

        # sync lora
        self.sync_lora = False
        if not (sync_lora_rank == 0 or sync_lora_scale == 0):
            self.sync_lora = True
            self.sync_lora_rank = sync_lora_rank
            self.sync_lora_scale = sync_lora_scale
            self.to_q_lora_sync = LoRALinearLayer(hidden_size, hidden_size, sync_lora_rank, network_alpha)
            self.to_k_lora_sync = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, sync_lora_rank, network_alpha)
            self.to_v_lora_sync = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, sync_lora_rank, network_alpha)
            self.to_out_lora_sync = LoRALinearLayer(hidden_size, hidden_size, sync_lora_rank, network_alpha)

    def forward(self,
                attn,
                hidden_states,
                pose_feature,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
                scale=None,):
        # assert pose_feature is not None
        pose_embedding_scale = (scale or self.scale)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if hidden_states.dim == 5:
            hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) (h w) c')
        elif hidden_states.ndim == 4:
            hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c')
        else:
            assert hidden_states.ndim == 3

        if self.query_condition and self.key_value_condition:
            assert encoder_hidden_states is None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if encoder_hidden_states.ndim == 5:
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b c f h w -> (b f) (h w) c')
        elif encoder_hidden_states.ndim == 4:
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b c h w -> b (h w) c')
        else:
            assert encoder_hidden_states.ndim == 3
        if pose_feature is not None:
            if pose_feature.ndim == 5:
                pose_feature = rearrange(pose_feature, "b c f h w -> (b f) (h w) c")
            elif pose_feature.ndim == 4:
                pose_feature = rearrange(pose_feature, "b c h w -> b (h w) c")
            else:
                assert pose_feature.ndim == 3

        batch_size, ehs_sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, ehs_sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if pose_feature is not None:
            if self.query_condition and self.key_value_condition:  # only self attention
                query_hidden_state = self.qkv_merge(hidden_states + pose_feature) * pose_embedding_scale + hidden_states
                key_value_hidden_state = query_hidden_state
            elif self.query_condition:
                query_hidden_state = self.q_merge(hidden_states + pose_feature) * pose_embedding_scale + hidden_states
                key_value_hidden_state = encoder_hidden_states
            else:
                key_value_hidden_state = self.kv_merge(encoder_hidden_states + pose_feature) * pose_embedding_scale + encoder_hidden_states
                query_hidden_state = hidden_states
        else:
            query_hidden_state = hidden_states
            key_value_hidden_state = encoder_hidden_states

        # original attention
        query = attn.to_q(query_hidden_state)
        key = attn.to_k(key_value_hidden_state)
        value = attn.to_v(key_value_hidden_state)
        if self.sync_lora:
            query = query + self.sync_lora_scale * self.to_q_lora_sync(query_hidden_state)
            key = key + self.sync_lora_scale * self.to_k_lora_sync(key_value_hidden_state)
            value = value + self.sync_lora_scale * self.to_v_lora_sync(key_value_hidden_state)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # key = key.chunk(2)
        # key = torch.cat([key[0], key[1]], dim=1).repeat(2, 1, 1)
        # value = value.chunk(2)
        # value = torch.cat([value[0], value[1]], dim=1).repeat(2, 1, 1)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        if self.sync_lora:
            hidden_states = hidden_states + self.sync_lora_scale * self.to_out_lora_sync(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class LORAPoseAdaptorAttnProcessor(nn.Module):
    def __init__(self,
                 hidden_size,  # dimension of hidden state
                 pose_feature_dim=None,  # dimension of the pose feature
                 cross_attention_dim=None,  # dimension of the text embedding
                 query_condition=False,
                 key_value_condition=False,
                 scale=1.0,
                 # lora keywords
                 rank=4,
                 network_alpha=None,
                 lora_scale=1.0,
                 # sync lora keywords
                 sync_lora_rank=4,
                 sync_lora_scale=1.0,
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pose_feature_dim = pose_feature_dim
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.query_condition = query_condition
        self.key_value_condition = key_value_condition
        assert hidden_size == pose_feature_dim
        if self.query_condition and self.key_value_condition:
            self.qkv_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.qkv_merge.weight)
            init.zeros_(self.qkv_merge.bias)
        elif self.query_condition:
            self.q_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.q_merge.weight)
            init.zeros_(self.q_merge.bias)
        else:
            self.kv_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.kv_merge.weight)
            init.zeros_(self.kv_merge.bias)
        # lora
        self.rank = rank
        self.lora_scale = lora_scale
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

        # sync lora
        self.sync_lora = False
        if not (sync_lora_rank == 0 or sync_lora_scale == 0):
            self.sync_lora = True
            self.sync_lora_rank = sync_lora_rank
            self.sync_lora_scale = sync_lora_scale
            self.to_q_lora_sync = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
            self.to_k_lora_sync = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
            self.to_v_lora_sync = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
            self.to_out_lora_sync = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(self,
                 attn,
                 hidden_states,
                 encoder_hidden_states=None,
                 attention_mask=None,
                 temb=None,
                 scale=1.0,
                 pose_feature=None,
                 ):
        # assert pose_feature is not None
        lora_scale = self.lora_scale if scale is None else scale
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if hidden_states.dim == 5:
            hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) (h w) c')
        elif hidden_states.ndim == 4:
            hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c')
        else:
            assert hidden_states.ndim == 3

        if self.query_condition and self.key_value_condition:
            assert encoder_hidden_states is None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if encoder_hidden_states.ndim == 5:
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b c f h w -> (b f) (h w) c')
        elif encoder_hidden_states.ndim == 4:
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b c h w -> b (h w) c')
        else:
            assert encoder_hidden_states.ndim == 3
        if pose_feature is not None:
            if pose_feature.ndim == 5:
                pose_feature = rearrange(pose_feature, "b c f h w -> (b f) (h w) c")
            elif pose_feature.ndim == 4:
                pose_feature = rearrange(pose_feature, "b c h w -> b (h w) c")
            else:
                assert pose_feature.ndim == 3

        batch_size, ehs_sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, ehs_sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if pose_feature is not None:
            if self.query_condition and self.key_value_condition:  # only self attention
                query_hidden_state = self.qkv_merge(hidden_states + pose_feature) * self.scale + hidden_states
                key_value_hidden_state = query_hidden_state
            elif self.query_condition:
                query_hidden_state = self.q_merge(hidden_states + pose_feature) * self.scale + hidden_states
                key_value_hidden_state = encoder_hidden_states
            else:
                key_value_hidden_state = self.kv_merge(encoder_hidden_states + pose_feature) * self.scale + encoder_hidden_states
                query_hidden_state = hidden_states
        else:
            query_hidden_state = hidden_states
            key_value_hidden_state = encoder_hidden_states

        # original attention
        query = attn.to_q(query_hidden_state) + lora_scale * self.to_q_lora(query_hidden_state)
        key = attn.to_k(key_value_hidden_state) + lora_scale * self.to_k_lora(key_value_hidden_state)
        value = attn.to_v(key_value_hidden_state) + lora_scale * self.to_v_lora(key_value_hidden_state)
        if self.sync_lora:
            query = query + self.sync_lora_scale * self.to_q_lora_sync(query_hidden_state)
            key = key + self.sync_lora_scale * self.to_k_lora_sync(key_value_hidden_state)
            value = value + self.sync_lora_scale * self.to_v_lora_sync(key_value_hidden_state)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + lora_scale * self.to_out_lora(hidden_states)
        if self.sync_lora:
            hidden_states = hidden_states + self.sync_lora_scale * self.to_out_lora_sync(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class EpiAttnProcessor(nn.Module):
    def __init__(self,
                 hidden_size,  # dimension of hidden state
                #  query_condition=False,
                #  key_value_condition=False,
                 scale=1.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.scale = scale
        # self.query_condition = query_condition
        # self.key_value_condition = key_value_condition
        # assert hidden_size == pose_feature_dim
        # if self.query_condition and self.key_value_condition:
        #     self.qkv_merge = nn.Linear(hidden_size, hidden_size)
        #     init.zeros_(self.qkv_merge.weight)
        #     init.zeros_(self.qkv_merge.bias)
        # elif self.query_condition:
        #     self.q_merge = nn.Linear(hidden_size, hidden_size)
        #     init.zeros_(self.q_merge.weight)
        #     init.zeros_(self.q_merge.bias)
        # else:
        #     self.kv_merge = nn.Linear(hidden_size, hidden_size)
        #     init.zeros_(self.kv_merge.weight)
        #     init.zeros_(self.kv_merge.bias)

    def forward(self,
                attn: Attention,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                kv_index=None,
                temb=None,
                scale=None,
                mono_direction=False,
                fix_firstframe=False,
                **useless_kwargs):
        
        pose_embedding_scale = (scale or self.scale)
        residual = hidden_states
        # if attn.spatial_norm is not None:
        #     hidden_states = attn.spatial_norm(hidden_states, temb)

        assert hidden_states.ndim == 3 # BF x HW x C
        # hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) (h w) c') # BF x HW x C

        # assert encoder_hidden_states is None

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        if encoder_hidden_states is None:
            if kv_index is None:
                split_hidden_states = hidden_states.chunk(2)
                encoder_hidden_states = torch.cat([split_hidden_states[1], split_hidden_states[0]], dim=0)
            else:
                encoder_hidden_states = hidden_states[kv_index]
                if kv_index.shape[0] != hidden_states.shape[0]:
                    assert kv_index.shape[0] % hidden_states.shape[0] == 0
                    B, N, C = hidden_states.shape
                    encoder_hidden_states = encoder_hidden_states.reshape(-1, B, N, C)
                    encoder_hidden_states = encoder_hidden_states.permute(1, 2, 0, 3).reshape(B, -1, C)

        batch_size, ehs_sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, ehs_sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1]) if attention_mask is not None else None

        # original attention
        if mono_direction or fix_firstframe:
            value_self = attn.to_v(hidden_states) 
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query_origin = query
        key_origin = key

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # print(key.shape)
        # if not key.shape[1] == 16 and not key.shape[2] == 77:
        #     key = key.chunk(2)
        #     key = torch.cat([key[0], key[1]], dim=1).repeat(2, 1, 1)
        #     value = value.chunk(2)
        #     value = torch.cat([value[0], value[1]], dim=1).repeat(2, 1, 1)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if mono_direction or fix_firstframe:
            value_self = value_self.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # hidden_states = torch.bmm(attention_probs, value)
        # hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if mono_direction:
            raise ValueError("kv_index may have bug here. Not supported")
            # Only apply attention to the second video
            if kv_index is None:
                hidden_states = torch.cat([value_self[:batch_size//2], hidden_states[batch_size//2:]], dim=0)
            else:
                index = torch.arange(len(kv_index)).type_as(kv_index)
                hidden_states = torch.where(index==kv_index, value_self, hidden_states)
        if fix_firstframe:
            # Need to seperate the cond / uncond first frame
            value_reshaped = rearrange(value_self, '(b t f) n h c -> b t f n h c', t=2, f=16)
            bs = value_reshaped.shape[0]
            value_reshaped_ff = value_reshaped[:, :, 0:1].mean(dim=0, keepdim=True).repeat(bs, 1, 1, 1, 1, 1)
            value_reshaped = rearrange(value_reshaped_ff, 'b t f n h c -> (b t f) n h c')
            hidden_states[::16] = value_reshaped

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states, {"query": query_origin, "key": key_origin}
