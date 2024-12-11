# make sure you're logged in with `huggingface-cli login`
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from packaging import version as pver
from einops import rearrange
from safetensors import safe_open

from omegaconf import OmegaConf
from diffusers import (
    AutoencoderKL,
    DDIMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.utils.util import save_videos_grid
from animatediff.models.unet import UNet3DConditionModelPoseCond
from animatediff.models.pose_adaptor import CameraPoseEncoder
from animatediff.pipelines.pipeline_animation_epi import AnimationPipelineEpiControl
from animatediff.data.dataset_epi import RealEstate10KPoseFolded

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_vae_checkpoint, \
    convert_ldm_clip_checkpoint
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint
from animatediff.data.dataset import Camera
import imageio

from tools.visualize_trajectory import CameraPoseVisualizer


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def get_relative_pose(cam_params, zero_first_frame_scale):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    source_cam_c2w = abs_c2ws[0]
    if zero_first_frame_scale:
        cam_to_origin = 0
    else:
        cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def ray_condition(K, c2w, H, W, device):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


def load_civitai_base_model(pipeline, civitai_base_model):
    print(f'Load civitai base model from {civitai_base_model}')
    if civitai_base_model.endswith(".safetensors"):
        dreambooth_state_dict = {}
        with safe_open(civitai_base_model, framework="pt", device="cpu") as f:
            for key in f.keys():
                dreambooth_state_dict[key] = f.get_tensor(key)
    elif civitai_base_model.endswith(".ckpt"):
        dreambooth_state_dict = torch.load(civitai_base_model, map_location="cpu")

    # 1. vae
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, pipeline.vae.config)
    pipeline.vae.load_state_dict(converted_vae_checkpoint)
    # 2. unet
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, pipeline.unet.config)
    _, unetu = pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
    assert len(unetu) == 0
    # 3. text_model
    pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict, text_encoder=pipeline.text_encoder)
    del dreambooth_state_dict
    return pipeline


def get_pipeline(ori_model_path, unet_subfolder, image_lora_rank, image_lora_ckpt, unet_additional_kwargs,
                 unet_mm_ckpt, unet_epi_ckpt, pose_encoder_kwargs, attention_processor_kwargs,
                 noise_scheduler_kwargs, pose_adaptor_ckpt, civitai_lora_ckpt, civitai_base_model, gpu_id,
                 spatial_extended_attention=False):
    vae = AutoencoderKL.from_pretrained(ori_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(ori_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(ori_model_path, subfolder="text_encoder")
    unet = UNet3DConditionModelPoseCond.from_pretrained_2d(ori_model_path, subfolder=unet_subfolder,
                                                           unet_additional_kwargs=unet_additional_kwargs)
    pose_encoder = CameraPoseEncoder(**pose_encoder_kwargs)
    print(f"Setting the attention processors")
    unet.set_all_attn_processor(add_spatial_lora=image_lora_ckpt is not None,
                                add_motion_lora=False,
                                lora_kwargs={"lora_rank": image_lora_rank, "lora_scale": 1.0},
                                motion_lora_kwargs={"lora_rank": -1, "lora_scale": 1.0},
                                sync_lora_kwargs={"sync_lora_rank": 0, "sync_lora_scale": 0},
                                spatial_extended_attention=spatial_extended_attention,
                                **attention_processor_kwargs)

    if image_lora_ckpt is not None:
        print(f"Loading the lora checkpoint from {image_lora_ckpt}")
        lora_checkpoints = torch.load(image_lora_ckpt, map_location=unet.device)
        if 'lora_state_dict' in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints['lora_state_dict']
        _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        assert len(lora_u) == 0
        print(f'Loading done')

    if unet_mm_ckpt is not None:
        print(f"Loading the motion module checkpoint from {unet_mm_ckpt}")
        mm_checkpoints = torch.load(unet_mm_ckpt, map_location=unet.device)
        _, mm_u = unet.load_state_dict(mm_checkpoints, strict=False)
        assert len(mm_u) == 0
        print("Loading done")

    if unet_epi_ckpt is not None:
        print(f"Loading the epi module checkpoint from {unet_epi_ckpt}")
        ckpt = torch.load(unet_epi_ckpt, map_location=unet.device)
        unet_trainable_dict = ckpt['unet_trainable_dict']
        _, epi_u = unet.load_state_dict(unet_trainable_dict, strict=False)
        assert len(epi_u) == 0
        print("Loading done")
        
    print(f"Loading pose adaptor")
    pose_adaptor_checkpoint = torch.load(pose_adaptor_ckpt, map_location='cpu')
    pose_encoder_state_dict = pose_adaptor_checkpoint['pose_encoder_state_dict']
    pose_encoder_m, pose_encoder_u = pose_encoder.load_state_dict(pose_encoder_state_dict)
    assert len(pose_encoder_u) == 0 and len(pose_encoder_m) == 0
    attention_processor_state_dict = pose_adaptor_checkpoint['attention_processor_state_dict']
    _, attn_proc_u = unet.load_state_dict(attention_processor_state_dict, strict=False)
    assert len(attn_proc_u) == 0
    print(f"Loading done")

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))
    vae.to(gpu_id)
    text_encoder.to(gpu_id)
    unet.to(gpu_id)
    pose_encoder.to(gpu_id)
    pipe = AnimationPipelineEpiControl(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        pose_encoder=pose_encoder)
    assert not (civitai_base_model and civitai_lora_ckpt)
    if civitai_lora_ckpt is not None:
        pipe.load_lora_weights(civitai_lora_ckpt)
    if civitai_base_model is not None:
        load_civitai_base_model(pipeline=pipe, civitai_base_model=civitai_base_model)
    pipe.enable_vae_slicing()
    pipe = pipe.to(gpu_id)

    return pipe


def load_pair_poses(pose_file_0, pose_file_1):
    pose_file_0 = os.path.join(self.pose_file_0)
    with open(pose_file_0, 'r') as f:
        poses_0 = f.readlines()
    pose_file_1 = os.path.join(self.pose_file_1)
    with open(pose_file_1, 'r') as f:
        poses_1 = f.readlines()
    poses_0 = [pose_0.strip().split(' ') for pose in poses_0[1:]]
    cam_params_0 = [[float(x) for x in pose] for pose in poses_0]
    cam_params_0 = [Camera(cam_param) for cam_param in cam_params_0]
    poses_1 = [pose_0.strip().split(' ') for pose in poses_1[1:]]
    cam_params_1 = [[float(x) for x in pose] for pose in poses_1]
    cam_params_1 = [Camera(cam_param) for cam_param in cam_params_1]
    return cam_params_0, cam_params_1


def main(args):
    os.makedirs(args.out_root, exist_ok=True)
    rank = args.local_rank
    setup_for_distributed(rank == 0)
    gpu_id = rank % torch.cuda.device_count()
    model_configs = OmegaConf.load(args.model_config)
    unet_additional_kwargs = model_configs[
        'unet_additional_kwargs'] if 'unet_additional_kwargs' in model_configs else None
    noise_scheduler_kwargs = model_configs['noise_scheduler_kwargs']
    pose_encoder_kwargs = model_configs['pose_encoder_kwargs']
    attention_processor_kwargs = model_configs['attention_processor_kwargs']
    validation_configs =  model_configs[
        'validation_data'] if 'validation_data' in model_configs else None
    unet_additional_kwargs['epi_module_kwargs']['epi_position_encoding_F_mat_size'] = args.image_height
    
    # overwritten
    attention_processor_kwargs["scale"] = args.pose_adaptor_scale

    print(f'Constructing pipeline')
    pipeline = get_pipeline(args.ori_model_path, args.unet_subfolder, args.image_lora_rank, args.image_lora_ckpt,
                            unet_additional_kwargs, args.motion_module_ckpt, args.epi_module_ckpt, 
                            pose_encoder_kwargs, attention_processor_kwargs,
                            noise_scheduler_kwargs, args.pose_adaptor_ckpt, args.civitai_lora_ckpt,
                            args.civitai_base_model, f"cuda:{gpu_id}",
                            spatial_extended_attention=args.spatial_extended_attention)
    device = torch.device(f"cuda:{gpu_id}")
    print('Done')

    print(f'Loading Validation Dataset')

    # with open(args.validation_prompts_file, "r") as f:
    #     validation_prompts = [x.replace("\n", "") for x in f.readlines()]
    if args.visualization_captions.endswith('.json'):
        json_file = json.load(open(args.visualization_captions, 'r'))
        captions = json_file['captions'] if 'captions' in json_file else json_file['prompts']
        if args.use_negative_prompt:
            negative_prompts = json_file['negative_prompts']
        else:
            negative_prompts = None
        if isinstance(captions[0], dict):
            captions = [cap['caption'] for cap in captions]
        if args.use_specific_seeds:
            specific_seeds = json_file['seeds']
        else:
            specific_seeds = None
    elif args.visualization_captions.endswith('.txt'):
        with open(args.visualization_captions, 'r') as f:
            captions = f.readlines()
        captions = [cap.strip() for cap in captions]
        negative_prompts = None
        specific_seeds = None

    if args.num_videos is not None:
        captions = captions * args.num_videos
        negative_prompts = negative_prompts * args.num_videos
    validation_configs["validation_prompts"] = captions
    validation_configs["validation_negative_prompts"] = negative_prompts
    validation_configs["sample_size"] = args.image_height
    if args.pose_file_0 is not None and args.pose_file_1 is not None:
        validation_configs["pose_file_0"] = args.pose_file_0
        validation_configs["pose_file_1"] = args.pose_file_1
    validation_dataset = RealEstate10KPoseFolded(**validation_configs)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    print(f'Done')
  
    generator = torch.Generator(device=device)
    generator.manual_seed(args.global_seed)

    validation_data_iter = iter(validation_dataloader)
    sample_all = []

    if args.no_lora_validation:
        pipeline.unet.set_image_layer_lora_scale(0.0)

    for idx, validation_batch in enumerate(validation_data_iter):
        if specific_seeds is not None:
            specific_seed = specific_seeds[idx]
            generator.manual_seed(specific_seed)
        pixel_values = validation_batch['pixel_values']

        F_mats = validation_batch['F_mats'].to(device=pipeline.unet.device)
        F_mats = torch.cat(F_mats.chunk(2, dim=1), dim=0)
        plucker_embedding = validation_batch['plucker_embedding'].to(device=pipeline.unet.device)
        plucker_embedding = torch.cat(plucker_embedding.chunk(2, dim=1), dim=0)
        plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")
        
        print(validation_batch['validation_prompt'])
        print(validation_batch['validation_negative_prompt'])
        output = pipeline(
            F_mats=F_mats,
            prompt=validation_batch['validation_prompt'],
            negative_prompt=validation_batch['validation_negative_prompt'],
            pose_embedding=plucker_embedding,
            video_length=args.video_length,
            height=args.image_height,
            width=args.image_width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            first_frame_guidance_scale=args.first_frame_guidance_scale,
            generator=generator,
        )  # [2 3 f h w]
        sample = output.videos  # [2 3 f h w]
        # save images
        cur_out_root = os.path.join(args.out_root, str(idx))
        img_out_root = os.path.join(cur_out_root, 'imgs')
        os.makedirs(img_out_root, exist_ok=True)
        for frame_i in range(sample.shape[2]):
            imageio.imwrite(f"{img_out_root}/{frame_i}-{0}.png", 
                (sample[0, :, frame_i, ] * 255.0).clamp(0, 255).permute(1, 2, 0).detach().numpy().astype(np.uint8))
            imageio.imwrite(f"{img_out_root}/{frame_i}-{1}.png", 
                (sample[1, :, frame_i, ] * 255.0).clamp(0, 255).permute(1, 2, 0).detach().numpy().astype(np.uint8))
        # save individual videos
        vid_out_root = os.path.join(cur_out_root, 'vids')
        for video_i in range(sample.shape[0]):
            save_path = f"{vid_out_root}/{video_i}.mp4"
            save_videos_grid(sample[video_i].unsqueeze(0), save_path)
        # save combined videos
        save_path = f"{vid_out_root}/horizontal.mp4"
        save_videos_grid(rearrange(sample, "b c f h w -> c f h (b w)").unsqueeze(0), save_path)
        save_path = f"{vid_out_root}/vertical.mp4"
        save_videos_grid(rearrange(sample, "b c f h w -> c f (b h) w").unsqueeze(0), save_path)
        # save trajectories
        ret_c2w = validation_batch['ret_c2w'].squeeze()
        ret_c2w_list = ret_c2w.chunk(2)
        pose_out_root = os.path.join(cur_out_root, "poses")
        os.makedirs(pose_out_root, exist_ok=True)
        for video_idx, ret_c2w in enumerate(ret_c2w_list):
            ret_c2w = ret_c2w.detach().cpu().numpy()
            visualizer = CameraPoseVisualizer([-1, 1], [-1, 1], [-1, 1])
            transform_matrix = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).reshape(4, 4)
            for frame_idx, c2w in enumerate(ret_c2w):
                visualizer.extrinsic2pyramid(c2w @ transform_matrix, frame_idx / ret_c2w.shape[0], hw_ratio=args.image_width / args.image_height, base_xval=0.035, zval=0.04)
            visualizer.colorbar(16)
            pose_img_dir = os.path.join(pose_out_root, f"pose_img_{video_idx}.png")
            visualizer.show(pose_img_dir)
            ret_c2w_dir = os.path.join(pose_out_root, f"ret_c2w_{video_idx}.png")
            np.save(ret_c2w_dir, ret_c2w)

        sample = rearrange(sample, "b c f h w -> c f h (b w)")
        sample_all.append(sample)
    
    if args.no_lora_validation:
        pipeline.unet.set_image_layer_lora_scale(1.0)

    vid_out_root = os.path.join(args.out_root, 'vids')
    for video_i in range(len(sample_all)):
        save_path = f"{vid_out_root}/{video_i}.mp4"
        save_videos_grid(sample_all[video_i].unsqueeze(0), save_path)

    sample_all = torch.stack(sample_all, dim=0) # n x 3 x f x 2h x w
    save_path = f"{args.out_root}/results.gif"
    save_videos_grid(sample_all, save_path)
    print(f"Saved samples to {save_path}")
    save_path = f"{args.out_root}/results.mp4"
    save_videos_grid(sample_all, save_path)
    print(f"Saved samples to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str)
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--image_width", type=int, default=384)
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--ori_model_path", type=str, help='path to the sd model folder')
    parser.add_argument("--unet_subfolder", type=str, help='subfolder name of unet ckpt')
    parser.add_argument("--image_lora_rank", type=int, default=2)
    parser.add_argument("--image_lora_ckpt", default=None)
    parser.add_argument("--civitai_lora_ckpt", default=None)
    parser.add_argument("--civitai_base_model", default=None)
    parser.add_argument("--pose_adaptor_ckpt", default=None, help='path to the camera control model ckpt')
    parser.add_argument("--motion_module_ckpt", type=str, help='path to the animatediff motion module ckpt')
    parser.add_argument("--epi_module_ckpt", type=str, help='path to the epi module ckpt')
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=15.0)
    parser.add_argument("--first_frame_guidance_scale", type=float, default=30.0)
    parser.add_argument("--visualization_captions", required=True, help='prompts path, json or txt')
    parser.add_argument("--use_negative_prompt", action='store_true', help='whether to use negative prompts')
    parser.add_argument("--use_specific_seeds", action='store_true', help='whether to use specific seeds for each prompt')
    parser.add_argument("--zero_first_frame_scale", action='store_true')
    parser.add_argument("--index_result_name", action='store_true', help='save generated videos using index name')
    parser.add_argument("--original_pose_width", type=int, default=1280, help='the width of the video used to extract camera trajectory')
    parser.add_argument("--original_pose_height", type=int, default=720, help='the height of the video used to extract camera trajectory')
    parser.add_argument("--n_procs", type=int, default=8)
    parser.add_argument("--global_seed", type=int, default=1024)

    parser.add_argument("--spatial_extended_attention", action='store_true')
    parser.add_argument("--pose_adaptor_scale", type=float, default=1.0)

    parser.add_argument("--pose_file_0", default=None)
    parser.add_argument("--pose_file_1", default=None)
    parser.add_argument("--num_videos", type=int, default=None)

    # validation dataset configs
    parser.add_argument("--no_lora_validation", action='store_true')
    
    # DDP args
    parser.add_argument("--world_size", default=1, type=int,
                        help="number of the distributed processes.")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Replica rank on the current node. This field is required '
                             'by `torch.distributed.launch`.')
    args = parser.parse_args()

    main(args)