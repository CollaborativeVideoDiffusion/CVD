GPU=$1
MASTER_PORT=$(expr 27000 + $GPU)

CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=1 --master_port=$MASTER_PORT inference_epi_advanced.py \
--out_root results/for_3D_randslope_GPU$GPU \
--ori_model_path /data/zhengfei/EpiDiff/models/StableDiffusion --unet_subfolder unet_webvidlora_v3 \
--pose_adaptor_ckpt /home/zhengfei/data2/CameraCtrl_internal/models/CameraCtrl/CameraCtrl.ckpt \
--motion_module_ckpt /data/zhengfei/EpiDiff/models/Motion_Module/v3_sd15_mm.ckpt \
--model_config configs/train_epictrl/adv3_256_256_epictrl_relora_stronger_randslope.yaml \
--epi_module_ckpt /media/data2/zhengfei/remote_checkpoints/hybrid_homo_stronger.ckpt \
--civitai_base_model /data/zhengfei/EpiDiff/models/Dreambooth_LoRA/realisticVisionV60B1_v51VAE.safetensors \
--visualization_captions assets/cameractrl_prompts_for_3D_3.json \
--use_specific_seeds --zero_first_frame_scale \
--image_height 256 \
--image_width 256 \
--n_procs 1 \
--num_inference_steps 25 \
--self_defined_camera \
--use_caption_from_file \
--validation_video_split 6 \
--multistep 6 \
--multiseed 3 \
--accumulate_step 2 \
--cam_pattern "interpolate" \ 
# supported cam patterns: interpolate, circle, upper_hemi
# --use_n2_pipeline \
