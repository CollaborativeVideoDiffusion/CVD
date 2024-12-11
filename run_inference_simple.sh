#!/bin/bash -l

GPU=$1
SEED=$2
RANDOM_PORT=$((25100 + GPU))

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --nproc_per_node=1 --master_port=${RANDOM_PORT} inference_epi_guidance.py \
--out_root /media/data2/shengqu/epictrl/results_supp_ours/${GPU}/ \
--ori_model_path /data/zhengfei/EpiDiff/models/StableDiffusion --unet_subfolder unet_webvidlora_v3 \
--pose_adaptor_ckpt /home/zhengfei/data2/CameraCtrl_internal/models/CameraCtrl/CameraCtrl.ckpt \
--motion_module_ckpt /data/zhengfei/EpiDiff/models/Motion_Module/v3_sd15_mm.ckpt \
--epi_module_ckpt /media/data2/zhengfei/remote_checkpoints/hybrid_homo_stronger.ckpt \
--model_config configs/train_epictrl/adv3_256_256_epictrl_relora_stronger.yaml \
--visualization_captions configs/prompts/cameractrl_prompts.json \
--zero_first_frame_scale \
--image_height 256 \
--image_width 256 \
--n_procs 1 \
--no_lora_validation \
--guidance_scale 14 \
--first_frame_guidance_scale 0 \
--pose_adaptor_scale 1.0 \
--global_seed ${SEED} \
--use_negative_prompt \
--num_videos 8 \
--civitai_base_model /data/zhengfei/EpiDiff/models/Dreambooth_LoRA/realisticVisionV60B1_v51VAE.safetensors \
--pose_file_0 /home/shengqu/repos/Epi_CameraCtrl/assets/pose_files/2f25826f0d0ef09a.txt \
--pose_file_1 /home/shengqu/repos/Epi_CameraCtrl/assets/pose_files/2c80f9eb0d3b2bb4.txt
# --civitai_base_model /media/data2/shengqu/epictrl/Dreambooth_LoRA/majicmixRealistic_v7.safetensors \
# --pose_file_0 /home/shengqu/repos/Epi_CameraCtrl/assets/pose_files/0bf152ef84195293.txt \
# --pose_file_1 /home/shengqu/repos/Epi_CameraCtrl/assets/pose_files/0c9b371cc6225682.txt \
# --civitai_lora_ckpt /media/data2/shengqu/epictrl/Dreambooth_LoRA/TUSUN.safetensors
# --civitai_base_model /data/zhengfei/EpiDiff/models/Dreambooth_LoRA/lyriel_v16.safetensors \
# --pose_file_0 /home/shengqu/repos/Epi_CameraCtrl/assets/pose_files/0bf152ef84195293.txt \
# --pose_file_1 /home/shengqu/repos/Epi_CameraCtrl/assets/pose_files/0bf152ef84195293.txt \
# --civitai_base_model /data/zhengfei/EpiDiff/models/Dreambooth_LoRA/realisticVisionV60B1_v51VAE.safetensors \
# --pose_file_0 /home/shengqu/repos/Epi_CameraCtrl/assets/pose_files/0bf152ef84195293.txt \
# --pose_file_1 /home/shengqu/repos/Epi_CameraCtrl/assets/pose_files/0c9b371cc6225682.txt \
# --spatial_extended_attention
# --use_specific_seeds
# --trajectory_file assets/pose_files/0f47577ab3441480.txt \