# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
import torch

import numpy as np
import math

from typing import Callable, List, Optional, Union
from dataclasses import dataclass
from diffusers.utils import is_accelerate_available
from packaging import version
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import deprecate, logging, BaseOutput

from animatediff.models.pose_adaptor import CameraPoseEncoder, CameraPoseEncoder2D
from animatediff.models.unet import UNet3DConditionModel
from animatediff.data.dataset_epi import calc_fundamental_matrix

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class AnimationPipelineEpiControlOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]
    auxiliary: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline, LoraLoaderMixin):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,

        multidiff_total_steps: int = 1,
        multidiff_overlaps: int = 12,
        **kwargs,
    ):
        pass
 
class AnimationPipelineEpiControl(AnimationPipeline):
    _optional_components = []

    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet3DConditionModel,
                 scheduler: Union[
                     DDIMScheduler,
                     PNDMScheduler,
                     LMSDiscreteScheduler,
                     EulerDiscreteScheduler,
                     EulerAncestralDiscreteScheduler,
                     DPMSolverMultistepScheduler],
                 pose_encoder: CameraPoseEncoder):

        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)

        self.register_modules(
            pose_encoder=pose_encoder
        )

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        pose_embedding: Optional[torch.FloatTensor],
        video_length: Optional[int],
        F_mats: Optional[torch.Tensor]=None,
        H_mats: Optional[torch.Tensor]=None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        multidiff_total_steps: int = 1,
        multidiff_overlaps: int = 12,
        device = "cuda",
        aux_c2w=None,        
        aux_K_mats=None,
        multistep=1,
        n2_random_pair_num=-1,
        accumulate_step=1,
        **kwargs,
    ):
        assert multidiff_total_steps == 1
        
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)
    
        if pose_embedding is not None:
            device = pose_embedding[0].device if isinstance(pose_embedding, list) else pose_embedding.device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )           # [2bf, l, c]

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        single_model_length = video_length
        video_length = multidiff_total_steps * (video_length - multidiff_overlaps) + multidiff_overlaps
        num_channels_latents = self.unet.in_channels

        video_split_num = pose_embedding.shape[0] if pose_embedding is not None else 2
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt * video_split_num,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )                   # b c f h w
        latents_dtype = latents.dtype
        bs = latents.shape[0]
        # latents[1:,:,0] = latents[:1,:,0]

        if F_mats is not None and video_split_num == 2: # F_mats shape is b, f, 3, 3
            assert F_mats.shape[0] == latents.shape[0] 
            F_mats = F_mats.type_as(latents)
            # Order: Uncond-src, Cond-src, Uncond-tgt, Cond-tgt
            # if do_classifier_free_guidance:
            #     F_mats = F_mats.repeat_interleave(2, dim=0)       
        if H_mats is not None: # F_mats shape is b, f, 3, 3
            H_mats = H_mats.type_as(latents)

        text_embeddings = text_embeddings.repeat(video_split_num, 1, 1)

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        if pose_embedding is not None:
            if isinstance(pose_embedding, list):
                assert all([x.ndim == 5 for x in pose_embedding])
                bs = pose_embedding[0].shape[0]
                pose_embedding_features = []
                for pe in pose_embedding:
                    pose_embedding_feature = self.pose_encoder(pe)
                    pose_embedding_feature = [rearrange(x, '(b f) c h w -> b c f h w', b=bs) for x in pose_embedding_feature]
                    pose_embedding_features.append(pose_embedding_feature)
                pose_embedding_features = [[x.repeat_interleave(2, dim=0) for x in pose_embedding_feature]
                                        for pose_embedding_feature in pose_embedding_features] \
                    if do_classifier_free_guidance else pose_embedding_features
            else:
                bs = pose_embedding.shape[0]
                assert pose_embedding.ndim == 5    
                pose_embedding_features = self.pose_encoder(pose_embedding)       # bf, c, h, w
                pose_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs)
                                        for x in pose_embedding_features]
                pose_embedding_features = [x.repeat_interleave(2, dim=0) for x in pose_embedding_features] \
                    if do_classifier_free_guidance else pose_embedding_features  # [2b c f h w]

        if aux_c2w is not None:
            aux_c2w = aux_c2w.detach().cpu().numpy()[0] # bf, 4, 4
            aux_K_mats = aux_K_mats.detach().cpu().numpy()[0] # bf, 3, 3
        
        # Denoising loop        
        torch.cuda.empty_cache()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            MULTI_STEP_LIST = np.linspace(multistep**0.5, 2**0.5, len(timesteps))**2
            MULTI_STEP_LIST[-1] = 1
            for i, t in enumerate(timesteps):
                MULTI_STEP = multistep if i != len(timesteps)-1 else 1
                for mt_step in range(MULTI_STEP): # Multistep iteration during each denoising step
                    noise_pred_full = torch.zeros_like(latents).to(latents.device)
                    for acc_step in range(accumulate_step):
                        mask_full = torch.zeros_like(latents).to(latents.device)
                        noise_preds = []
                        for multidiff_step in range(multidiff_total_steps):
                            start_idx = multidiff_step * (single_model_length - multidiff_overlaps)
                            latent_partial = latents[:, :, start_idx: start_idx + single_model_length].contiguous()
                            mask_full[:, :, start_idx: start_idx + single_model_length] += 1
                            if pose_embedding is not None:
                                if isinstance(pose_embedding, list):
                                    pose_embedding_features_input = pose_embedding_features[multidiff_step]
                                else:
                                    pose_embedding_features_input = [x[:, :, start_idx: start_idx + single_model_length]
                                                                    for x in pose_embedding_features]
                            else:
                                pose_embedding_features_input = None

                            if pose_embedding_features_input is not None:
                                bf = video_split_num * single_model_length
                                video_id = np.arange(bf) # bf

                                video_perm = np.random.permutation(video_split_num).reshape(2, video_split_num//2)

                                video_id_offset = np.zeros(video_split_num, dtype=video_id.dtype)
                                video_id_offset[video_perm[0,:]] = video_perm[1,:] - video_perm[0,:]
                                video_id_offset[video_perm[1,:]] = video_perm[0,:] - video_perm[1,:]
                                video_id_offset = video_id_offset * single_model_length

                                video_id_offset = video_id_offset.repeat(single_model_length)
                                video_id = (video_id + video_id_offset + bf) % bf

                                F_mats_origin = F_mats
                                if video_split_num != 2: # calculate F_mats in the pipeline
                                    assert aux_c2w is not None
                                    
                                    tgt_c2w = aux_c2w[video_id]
                                    tgt_K_mats = aux_K_mats[video_id]
                                    F_mat_list = []
                                    for frame_id in range(bf):           
                                        s2t = np.linalg.inv(tgt_c2w[frame_id]) @ aux_c2w[frame_id]
                                        F_mat = calc_fundamental_matrix(s2t, aux_K_mats[frame_id], tgt_K_mats[frame_id])
                                        F_mat_list.append(torch.from_numpy(F_mat))
                                    F_mats = torch.cat(F_mat_list, dim=0).reshape(video_split_num, single_model_length, 3, 3)
                                    F_mats = F_mats.type_as(pose_embedding)
                                    
                                if do_classifier_free_guidance:
                                    F_mats_input = F_mats.repeat_interleave(2, dim=0)
                                    video_id_offset = video_id_offset.repeat(2, axis=0) * 2
                                    video_id_input = (np.arange(2 * bf) + video_id_offset) % (2 * bf)
                                else:
                                    F_mats_input = F_mats    
                                    video_id_input = video_id   
                                    
                                if video_split_num != 2: 
                                    F_mats_input = [F_mats_input, video_id_input]
                            else:
                                F_mats_input = None
                                
                            if H_mats is not None:
                                if do_classifier_free_guidance:
                                    H_mats_input = H_mats.repeat_interleave(2, dim=0)
                                else:
                                    H_mats_input = H_mats    
                            else:
                                H_mats_input = None

                            # expand the latents if we are doing classifier free guidance
                            # latent_model_input = torch.cat([latent_partial] * 2) if do_classifier_free_guidance else latent_partial   # [2b c f h w]
                            if do_classifier_free_guidance:
                                latent_model_input = latent_partial.repeat_interleave(2, dim=0)
                            else:
                                latent_model_input = latent_partial

                            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                            # predict the noise residual
                            output = self.unet(latent_model_input, t, F_mats=F_mats_input, H_mats=H_mats_input, encoder_hidden_states=text_embeddings,
                                                pose_embedding_features=pose_embedding_features_input)
                            noise_pred = output.sample.to(dtype=latents_dtype)
                            aux = output.auxiliary
                            # perform guidance
                            
                            if do_classifier_free_guidance:
                                noise_pred_uncond = noise_pred[0::2]
                                noise_pred_text = noise_pred[1::2]
                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                                # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                            noise_preds.append(noise_pred)
                        for pred_idx, noise_pred in enumerate(noise_preds):
                            start_idx = pred_idx * (single_model_length - multidiff_overlaps)
                            noise_pred_full[:, :, start_idx: start_idx + single_model_length] += noise_pred / mask_full[:, :, start_idx: start_idx + single_model_length]

                    # compute the previous noisy sample x_t -> x_t-1  b c f h w
                    latents = self.scheduler.step(noise_pred_full / accumulate_step, t, latents, **extra_step_kwargs).prev_sample
                    if mt_step != MULTI_STEP - 1: # Add noise again to latents
                        noise = torch.randn_like(latents)  # [b, c, f, h, w]
                        prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                        alpha_prod_t = self.scheduler.alphas_cumprod[t]
                        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep]
                        latents = latents * ((alpha_prod_t / alpha_prod_t_prev)**0.5) + ((1 - alpha_prod_t / alpha_prod_t_prev)**0.5) * noise

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineEpiControlOutput(videos=video, auxiliary=aux)
