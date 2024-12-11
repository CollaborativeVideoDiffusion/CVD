import os
import random
import json
import torch

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np

from decord import VideoReader
from torch.utils.data.dataset import Dataset
from packaging import version as pver
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import glob
import imageio
import cv2
# from animatediff.data.dinov2matcher import Dinov2Matcher

# a -> [ax]
# [0, -a3, a2]
# [a3, 0, -a1]
# [-a2, a1, 0]
# input: a vector (shape: 3) or vector array (shape: N x 3)
# output: a matrix  (shape: 3 x 3) of array of matrix (shape: N x 3 x 3)
def calc_cross_product_matrix(vec):
    is_array = False if len(vec.shape) == 1 else True
    if not is_array:
        vec = vec[np.newaxis, :] # 1 x 3
    ret_mat = np.zeros(list(vec.shape)+[3])
    ret_mat[:, 0, 1] = -vec[:, 2]
    ret_mat[:, 0, 2] = vec[:, 1]
    ret_mat[:, 1, 2] = -vec[:, 0]
    ret_mat -= ret_mat.transpose((0, 2, 1))
    if not is_array:
        ret_mat = ret_mat[0]
    return ret_mat

# T_mat shape is 4 x 4
# x2 = T_mat * x1 = R_mat*x1 + t = R_mat * ( x1 - (-R_mat^T * t) )
# let R_ess = R_mat, t_ess = -R_mat^T*t
# then E_mat = R_ess*[t_ess x]
def calc_essential_matrix(T_mat): 
    R_mat = T_mat[:3, :3]
    t = T_mat[:3, 3] # t 
    t_ess = -np.matmul(R_mat.transpose(), t)
    E_mat = np.matmul(R_mat, calc_cross_product_matrix(t_ess))
    return E_mat

# T_mat: from camera 1 to camera 2
# x2 = T_mat * x1 
# because in essential matrix we have x2^t E x1 = 0, 
# and x_{1,2} = K_{1,2}^-1 * coord_{1,2}, 
# we can get F = K2^-T * E * K1^-1
def calc_fundamental_matrix(T_mat, K_mat1, K_mat2):
    E_mat = calc_essential_matrix(T_mat)
    
    K2_invT = np.linalg.inv(K_mat2).transpose() 
    K1_inv = np.linalg.inv(K_mat1)
    F_mat = np.matmul(np.matmul(K2_invT, E_mat), K1_inv)

    return F_mat

# Assume cx=H/2, cy=W/2
def K_mat_from_fov(fov_deg, H, W):
    fx = (W/2) / math.tan(fov_deg/2) 
    fy = (H/2) / math.tan(fov_deg/2) 
    K_mat = np.array(
        [
            [fx, 0, W/2],
            [0, fy, H/2],
            [0, 0, 1]
        ]
    )
    return K_mat

def find_fundamental_matrix(src_w2c, dst_w2c, H=256, W=256, src_fov=45, dst_fov=45):
    if isinstance(src_w2c, torch.Tensor):
        src_w2c = src_w2c.cpu().detach().numpy()
        dst_w2c = dst_w2c.cpu().detach().numpy()
        
    if src_w2c.shape[0] == 3:
        b_vec = np.zeros_like(src_w2c[0:1,:])
        b_vec[:, -1] = 1 
        src_w2c = np.concatenate([src_w2c, b_vec], axis=0) # 4x4
        dst_w2c = np.concatenate([dst_w2c, b_vec], axis=0) # 4x4  
    T_mat = np.linalg.inv(src_w2c) @ dst_w2c
    K_mat_src = K_mat_from_fov(src_fov, H, W)
    K_mat_dst = K_mat_from_fov(dst_fov, H, W)
    return calc_fundamental_matrix(T_mat, K_mat_src, K_mat_dst)
    
def check_fundamental(image_1, image_2, F_mat):
    if image_1.shape[0] == 3:
        image_1 = ((image_1 + 1) / 2 * 255.0).clamp(0, 255).permute(1, 2, 0).detach().numpy().astype(np.uint8)
        image_2 = ((image_2 + 1) / 2 * 255.0).clamp(0, 255).permute(1, 2, 0).detach().numpy().astype(np.uint8)
        image_1 = image_1.copy()
        image_2 = image_2.copy()
        
    H, W, _ = image_1.shape

    # sample point
    for _ in range(10):
        color = tuple([int(c) for c in np.random.randint(0, 256, (3), dtype=np.int32)])
        coord1 = np.array([random.randrange(0, W), random.randrange(0, H), 1], dtype=np.float32) # 3
        a, b, c = np.matmul(F_mat, coord1) # epipolar line in image 2, means ax+by+c=0 
        if F_mat.abs().max() >= 1e-3:
            if abs(b) < 1e-5:
                l1 = (int(-a/c), 0)
                l2 = (int(-a/c), H)
            else:
                l1 = (0, int(-c/b))
                l2 = (W, int(-(c+a*W)/b))
            image_2 = cv2.line(image_2, l1, l2, color=color, thickness=2)
        circ_cord = tuple([int(c) for c in coord1[:2]])
        image_1 = cv2.circle(image_1, circ_cord, radius=5, color=color, thickness=3)

    return np.concatenate([image_1, image_2], axis=1)

class RandomHorizontalFlipWithPose(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipWithPose, self).__init__()
        self.p = p

    def get_flip_flag(self, n_image):
        return torch.rand(n_image) < self.p

    def forward(self, image, flip_flag=None):
        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)

        ret_images = []
        for fflag, img in zip(flip_flag, image):
            if fflag:
                ret_images.append(F.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)


class Camera(object):
    def __init__(self, entry):
        self.cid = entry[0]
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

class RealEstate10KPoseFolded(Dataset):
    def __init__(
            self,
            root_path,
            sample_stride=2,
            minimum_sample_stride=1,
            sample_n_frames=16,
            relative_pose=False,
            sample_size=256,
            return_clip_name=False,
            validation_prompts=None,
            validation_negative_prompts=None,
            validation_video_split=None,
            mode="train",
            pose_file_0=None,
            pose_file_1=None
    ):
        self.root_path = root_path
        self.relative_pose = relative_pose
        self.sample_stride = sample_stride
        self.minimum_sample_stride = minimum_sample_stride
        self.sample_n_frames = sample_n_frames
        self.return_clip_name = return_clip_name
        self.validation_prompts = validation_prompts
        self.validation_negative_prompts = validation_negative_prompts
        self.validation_video_split = validation_video_split
        self.mode = mode
        self.pose_file_0 = pose_file_0
        self.pose_file_1 = pose_file_1

        txt_dir = os.path.join(root_path, "RealEstate10K/train")
        video_dir = os.path.join(root_path, "dataset/train")
        txt_file_list = glob.glob(os.path.join(txt_dir, "*.txt"))
        caption_dict = json.load(open(os.path.join(root_path, "annotation_json", "train_captions.json"), 'r'))
        caption_dict.update(json.load(open(os.path.join(root_path, "annotation_json", "test_captions.json"), 'r')))
        self.dataset = []
        for pose_file_path in txt_file_list:
            clip_name = os.path.basename(pose_file_path).replace(".txt", "")
            origin_video_name = clip_name + ".mp4"
            if origin_video_name not in caption_dict.keys():
                continue
            caption = caption_dict[origin_video_name][0]
            clip_path = os.path.join(video_dir, clip_name)
            self.dataset.append({"clip_name": clip_name, "clip_path": clip_path,
                                "pose_file": pose_file_path, "caption": caption})

        self.length = len(self.dataset)
        print(f"Dataset: Loaded {self.length} video clips.")

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size
        # if use_flip:
        #     pixel_transforms = [transforms.Resize(sample_size),
        #                         transforms.CenterCrop(self.sample_size),
        #                         RandomHorizontalFlipWithPose(),
        #                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        # else:
        pixel_transforms = transforms.Compose([transforms.Resize(sample_size[0]),
                            transforms.CenterCrop(self.sample_size),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])

        self.pixel_transforms = pixel_transforms
    
    def get_relative_pose(self, c2w_list, tar_idx=0):
        abs2rel = np.linalg.inv(c2w_list[tar_idx])
        ret_poses = [abs2rel @ c2w for c2w in c2w_list]
        return np.array(ret_poses, dtype=np.float32)

    def load_video_reader(self, idx):
        video_dict = self.dataset[idx]
        return video_dict['clip_name'], video_dict['clip_path'], video_dict['caption']

    def load_cameras(self, idx):
        video_dict = self.dataset[idx]
        pose_file = os.path.join(self.root_path, video_dict['pose_file'])
        with open(pose_file, 'r') as f:
            poses = f.readlines()
        poses = [pose.strip().split(' ') for pose in poses[1:]]
        cam_params = [[float(x) for x in pose] for pose in poses]
        cam_params = [Camera(cam_param) for cam_param in cam_params]
        return cam_params

    def load_cameras_specific(self):
        pose_file_0 = os.path.join(self.pose_file_0)
        with open(pose_file_0, 'r') as f:
            poses_0 = f.readlines()
        pose_file_1 = os.path.join(self.pose_file_1)
        with open(pose_file_1, 'r') as f:
            poses_1 = f.readlines()
        poses_0 = [pose.strip().split(' ') for pose in poses_0[1:]]
        cam_params_0 = [[float(x) for x in pose] for pose in poses_0]
        cam_params_0 = [Camera(cam_param) for cam_param in cam_params_0]
        poses_1 = [pose.strip().split(' ') for pose in poses_1[1:]]
        cam_params_1 = [[float(x) for x in pose] for pose in poses_1]
        cam_params_1 = [Camera(cam_param) for cam_param in cam_params_1]
        cam_params_1.reverse()

        c2w_pose_list_0 = []
        K_mat_list_0 = []
        intrinsic_list_0 = []
        for frame_idx in range(len(cam_params_0)):
            cam = cam_params_0[frame_idx]
            H, W = 1280, 720

            crop_size = min(H, W)
            rescale = self.sample_size[0] / crop_size 
            dH, dW = (H-crop_size)/2, (W-crop_size)/2
            K_mat = np.array([[W*rescale*cam.fx, 0, (W*cam.cx-dW)*rescale], [0, H*rescale*cam.fy, (H*cam.cy-dH)*rescale], [0, 0, 1]])
            intrinsics = [K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2]]
            # While the statement in realestate10K states that the extrinsics are w2c,
            # Seems they are c2w instead
            c2w_pose_list_0.append(cam.c2w_mat)
            K_mat_list_0.append(K_mat)
            intrinsic_list_0.append(intrinsics)
        c2w_pose_list_1 = []
        K_mat_list_1 = []
        intrinsic_list_1 = []
        for frame_idx in range(len(cam_params_1)):
            cam = cam_params_1[frame_idx]
            H, W = 1280, 720

            crop_size = min(H, W)
            rescale = self.sample_size[0] / crop_size 
            dH, dW = (H-crop_size)/2, (W-crop_size)/2
            K_mat = np.array([[W*rescale*cam.fx, 0, (W*cam.cx-dW)*rescale], [0, H*rescale*cam.fy, (H*cam.cy-dH)*rescale], [0, 0, 1]])
            intrinsics = [K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2]]
            # While the statement in realestate10K states that the extrinsics are w2c,
            # Seems they are c2w instead
            c2w_pose_list_1.append(cam.c2w_mat)
            K_mat_list_1.append(K_mat)
            intrinsic_list_1.append(intrinsics)
        c2w_pose_list_0 = self.get_relative_pose(c2w_pose_list_0, tar_idx = 0)
        c2w_pose_list_1 = self.get_relative_pose(c2w_pose_list_1, tar_idx = 0)
        c2w_pose_list = np.concatenate([c2w_pose_list_0[1:][::-1], c2w_pose_list_1], axis=0)
        # force k mat to be the same
        K_mat_list = np.concatenate([K_mat_list_0[1:][::-1], K_mat_list_0], axis=0)
        intrinsic_list = np.concatenate([intrinsic_list_0[1:][::-1], intrinsic_list_1], axis=0)
        return c2w_pose_list, K_mat_list, intrinsic_list

    def interpolate_poses(self, src_poses, tgt_poses, split_num):
        frame_num = len(src_poses)
        ret_poses = np.tile(src_poses, (split_num, 1, 1))

        # interpolate translation
        for i in range(split_num):
            alpha = i / (split_num-1)
            ret_poses[i*frame_num:(i+1)*frame_num, :3, 3] = src_poses[:, :3, 3] * (1-alpha) + tgt_poses[:, :3, 3] * alpha # blend translate

        # interpolate rotation
        for frame_id in range(frame_num):
            src_quat = R.from_matrix(src_poses[frame_id, :3, :3])
            tgt_quat = R.from_matrix(tgt_poses[frame_id, :3, :3])
            interp_time = np.linspace(0, 1, split_num)
            sl = Slerp([0, 1], R.concatenate([src_quat, tgt_quat]))
            interp_quat = sl(interp_time)
            interp_rot = interp_quat.as_matrix()
            ret_poses[frame_id::frame_num, :3, :3] = interp_rot

        return ret_poses

    def get_batch(self, idx, validation_idx):

        if self.validation_prompts is not None:
            validation_prompt = self.validation_prompts[validation_idx]
        else:
            validation_prompt = None
        if self.validation_negative_prompts is not None:
            validation_negative_prompt = self.validation_negative_prompts[validation_idx]
        else:
            validation_negative_prompt = None
            
        clip_name, video_path, video_caption = self.load_video_reader(idx)
        cam_params = self.load_cameras(idx)
        sample_length = self.sample_n_frames * 2 - 1
        assert len(cam_params) >= sample_length
        total_frames = len(cam_params)

        stride = min(total_frames // sample_length, self.sample_stride)
        clip_length = min(total_frames, (sample_length - 1) * stride + 1)
        start_idx   = random.randint(0, total_frames - clip_length)
        frame_indices = np.linspace(start_idx, start_idx + clip_length - 1, sample_length, dtype=int)

        c2w_pose_list = []
        K_mat_list = []
        img_list = []
        intrinsic_list = []
        for frame_idx in frame_indices:
            cam = cam_params[frame_idx]
            img_path = os.path.join(video_path, "%d.png"%cam.cid)
            img = imageio.imread(img_path)
            H, W, _ = img.shape
            img = torch.from_numpy(img)[None].permute(0, 3, 1, 2).contiguous()
            img = img / 255.

            crop_size = min(H, W)
            rescale = self.sample_size[0] / crop_size 
            dH, dW = (H-crop_size)/2, (W-crop_size)/2
            K_mat = np.array([[W*rescale*cam.fx, 0, (W*cam.cx-dW)*rescale], [0, H*rescale*cam.fy, (H*cam.cy-dH)*rescale], [0, 0, 1]])
            intrinsics = [K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2]]
            # While the statement in realestate10K states that the extrinsics are w2c,
            # Seems they are c2w instead

            c2w_pose_list.append(cam.c2w_mat)
            img_list.append(img)
            K_mat_list.append(K_mat)
            intrinsic_list.append(intrinsics)

        if self.pose_file_0 is not None:
            c2w_pose_list, K_mat_list, intrinsic_list = self.load_cameras_specific()
        # from camera_utils import get_folded_ellipse_camera, get_folded_ellipse_camera_concat
        # c2w_pose_list = get_folded_ellipse_camera()
        # normalize poses
        c2w_pose_list = self.get_relative_pose(c2w_pose_list, tar_idx = self.sample_n_frames-1)
        if self.pose_file_0 is not None:
            c2w_pose_list, K_mat_list, intrinsic_list = self.load_cameras_specific()

        intrinsics = torch.as_tensor(np.array(intrinsic_list)).float()[None]   # [1, n_frame, 4]
        c2w = torch.as_tensor(c2w_pose_list)[None] # [1, n_frame, 4, 4]
        pixel_values = self.pixel_transforms(torch.as_tensor(np.concatenate(img_list, axis=0))) # [n_frame, 3, H, W]
        # dinov2_features = self.dinov2.extract_features(self.dinov2.transform(pixel_values))
        # dinov2_features = dinov2_features.permute(0, 2, 1).reshape(pixel_values.shape[0], 768, 32, 32)

        plucker_embedding = ray_condition(intrinsics, c2w, self.sample_size[0], self.sample_size[1], device='cpu'
                                          )[0].permute(0, 3, 1, 2) # n_frame, channel, H, W
        
        F_mat_list = []
        for i in range(self.sample_n_frames):
            sid = self.sample_n_frames - 1 - i
            tid = self.sample_n_frames - 1 + i
            s2t = np.linalg.inv(c2w_pose_list[tid]) @ c2w_pose_list[sid]
            F_mat = calc_fundamental_matrix(s2t, K_mat_list[sid], K_mat_list[tid])
            F_mat_list.append(torch.from_numpy(F_mat))

        F_mats = torch.as_tensor(np.array(F_mat_list)).float() # [n_frame, 3, 3]
        
        # Fold all vectors
        F_mats = torch.cat([F_mats, F_mats.permute(0, 2, 1)], dim=0).contiguous()
        fold_indices = torch.arange(self.sample_n_frames)
        fold_indices = torch.cat([self.sample_n_frames - 1 - fold_indices, 
                                  self.sample_n_frames - 1 + fold_indices])
        
        pixel_values = pixel_values[fold_indices].contiguous()
        plucker_embedding = plucker_embedding[fold_indices].contiguous()

        if self.validation_video_split is not None and self.validation_video_split != 2:
            folded_c2w_pose_list = c2w_pose_list[fold_indices] 
            inter_c2w_pose_list = self.interpolate_poses(folded_c2w_pose_list[:self.sample_n_frames], 
                                                     folded_c2w_pose_list[self.sample_n_frames:], 
                                                     self.validation_video_split)
            inter_c2w = torch.as_tensor(inter_c2w_pose_list)[None]
            folded_intrinsics = intrinsics[:, fold_indices][:, :self.sample_n_frames]
            inter_intrinsics = folded_intrinsics.repeat(1, self.validation_video_split, 1) 
            plucker_embedding = ray_condition(inter_intrinsics, inter_c2w, self.sample_size[0], self.sample_size[1], device='cpu'
                                            )[0].permute(0, 3, 1, 2) # n_frame, channel, H, W
            ret_c2w = inter_c2w[0]
            ret_K_mats = np.stack(K_mat_list, axis=0)[fold_indices][:self.sample_n_frames]
            ret_K_mats = np.tile(ret_K_mats, (self.validation_video_split, 1, 1))
        else:
            ret_c2w = c2w[:, fold_indices]
            ret_K_mats = np.stack(K_mat_list, axis=0)[fold_indices]

        return pixel_values, video_caption, plucker_embedding, F_mats, clip_name, validation_prompt, validation_negative_prompt,ret_c2w, ret_K_mats

    def __len__(self):
        return len(self.validation_prompts) if self.validation_prompts is not None else self.length

    def __getitem__(self, idx):
        idx_attempt = 0
        validation_idx = idx
        while True:
            try:
                video, video_caption, plucker_embedding, F_mats, clip_name, validation_prompt, validation_negative_prompt, ret_c2w, ret_K_mats = self.get_batch(idx, validation_idx)
                break
            except Exception as e:
                idx_attempt += 1
                if idx_attempt > 30:
                    print("Something going wrong with the data...")
                idx = random.randint(0, self.length - 1)
        # video, video_caption, plucker_embedding, F_mats, clip_name, validation_prompt, ret_c2w, ret_K_mats = self.get_batch(idx, validation_idx)

        ret_sample = {
            "pixel_values": video,
            "text": video_caption,
            "plucker_embedding": plucker_embedding,
            "F_mats": F_mats, 
            "ret_c2w": ret_c2w,
            "ret_K_mats": ret_K_mats
        }
        if self.return_clip_name:
            ret_sample["clip_name"] = clip_name
        if validation_prompt is not None:
            ret_sample["validation_prompt"] = validation_prompt
        if validation_negative_prompt is not None:
            ret_sample["validation_negative_prompt"] = validation_negative_prompt

        return ret_sample

if __name__ == "__main__":

    dataset = RealEstate10KPoseFolded(
        root_path = "/data/zhengfei/RealEstate10K_Downloader",
        validation_video_split=4
    )
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    os.makedirs("test_epi", exist_ok=True)
    for idx, batch in enumerate(dataloader):
        pixel_values = batch['pixel_values'][0]
        video_len = len(pixel_values)
        # src_imgs = pixel_values[:video_len//2]
        # tgt_imgs = pixel_values[video_len//2:]
        # F_mats = batch['F_mats'][0, :video_len//2]
        # ret_K_mats = batch['ret_K_mats'][0,0]
        # print(ret_K_mats[0,0], ret_K_mats[1,1])
        # if (ret_K_mats[0,0]-ret_K_mats[1,1]).abs() > 1e-3:
        #     for fid in range(len(src_imgs)):
        #         test_img = check_fundamental(src_imgs[fid], tgt_imgs[fid], F_mats[fid])
        #         imageio.imwrite("test_epi/%d.png"%fid, test_img)

        src_imgs = pixel_values[video_len//2:]
        tgt_imgs = pixel_values[:video_len//2]
        F_mats = batch['F_mats'][0, video_len//2:]
        ret_c2w = batch['ret_c2w'][0, video_len//2:].detach().cpu().numpy()
        tgt_c2w = batch['ret_c2w'][0, :video_len//2].detach().cpu().numpy()
        
        ret_K_mats = batch['ret_K_mats'][0].detach().cpu().numpy()
        # ret_K_mats = batch['ret_K_mats'][0,0]
        # print(ret_K_mats[0,0], ret_K_mats[1,1])
        # if (ret_K_mats[0,0]-ret_K_mats[1,1]).abs() > 1e-3:
        for fid in range(len(src_imgs)):
            s2t = np.linalg.inv(tgt_c2w[fid]) @ ret_c2w[fid]
            F_mat_calc = calc_fundamental_matrix(s2t, ret_K_mats[fid], ret_K_mats[fid])
            F_mat_calc = torch.as_tensor(F_mat_calc)
                                    
            test_img = check_fundamental(src_imgs[fid], tgt_imgs[fid], F_mats[fid])
            imageio.imwrite("test_epi/%d.png"%fid, test_img)
            test_img = check_fundamental(src_imgs[fid], tgt_imgs[fid], F_mat_calc)
            imageio.imwrite("test_epi/%d_alt.png"%fid, test_img)
        import pdb
        pdb.set_trace()
