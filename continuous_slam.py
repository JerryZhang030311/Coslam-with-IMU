import os
import sys
# os.environ['TCNN_CUDA_ARCHITECTURES'] = '86'

# Package imports
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import argparse
import shutil
import json
import cv2

from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
import config
from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from datasets.dataset import get_dataset
from utils import coordinates, extract_mesh, colormap_image
from tools.eval_ate import pose_evaluation
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, \
    matrix_to_quaternion
from PoseNet import *
from rotation_conversions import Lie, matrix_to_euler_angles

torch.cuda.empty_cache()



class CoSLAM():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)

        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()
        self.keyframeDatabase = self.create_kf_database(config)
        self.model = JointEncoding(config, self.bounding_box).to(self.device)
        self._config = {
            'device': self.device,  # "cuda:0",
            'poseNet_freq': 5,
            'layers_feat': [None, 256, 256, 256, 256, 256, 256, 256, 256],
            'skip': [4],
            'min_time': 0,
            'max_time': 100,
            'activ': 'relu',
            'cam_lr': 1e-3,
            'max_iter': 20000,
            'use_scheduler': True
        }
        self.num_posenet = 40
        self.iposenet = PoseNet(self._config)
        self.oposenet = PoseNet(self._config)
        self.lie = Lie()


    def iter_nums(self, i):
        if i < 10:
            return 0.5
        elif i < 25:
            return 1
        elif i < 50:
            return 1.5
        elif i < 75:
            return 2
        else:
            return 3

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def get_pose_representation(self):
        '''
        Get the pose representation axis-angle or quaternion
        '''
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
            print('Using axis-angle as rotation representation, identity init would cause inf')

        elif self.config['training']['rot_rep'] == "quat":
            print("Using quaternion as rotation representation")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError

    def create_pose_data(self):
        '''
        Create the pose data
        '''
        self.est_c2w_data = {}
        self.est_c2w_data_rel = {}
        self.load_gt_pose()

    def create_bounds(self):
        '''
        Get the pre-defined bounds for the scene
        '''
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(
            self.device)

    def create_kf_database(self, config):
        '''
        Create the keyframe database
        '''
        num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)
        print('#kf:', num_kf)
        print('#Pixels to save:', self.dataset.num_rays_to_save)
        return KeyFrameDatabase(config,
                                self.dataset.H,
                                self.dataset.W,
                                num_kf,
                                self.dataset.num_rays_to_save,
                                self.device)

    def load_gt_pose(self):
        '''
        Load the ground truth pose
        '''
        self.pose_gt = {}
        for i, pose in enumerate(self.dataset.poses):
            self.pose_gt[i] = pose

    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        save_dict = {'pose': self.est_c2w_data,
                     'pose_rel': self.est_c2w_data_rel,
                     'model': self.model.state_dict()}
        torch.save(save_dict, save_path)
        print('Save the checkpoint')

    def load_ckpt(self, load_path):
        '''
        Load the model parameters and the estimated pose
        '''
        dict = torch.load(load_path)
        self.model.load_state_dict(dict['model'])
        self.est_c2w_data = dict['pose']
        self.est_c2w_data_rel = dict['pose_rel']

    def select_samples(self, H, W, samples):
        '''
        randomly select samples from the image
        '''
        # indice = torch.randint(H*W, (samples,))
        indice = random.sample(range(H * W), int(samples))
        indice = torch.tensor(indice)
        return indice

    def get_loss_from_ret(self, ret, tracking=True, rgb=True, sdf=True, depth=True, fs=True, use_continuous_pose=False,
                          frame_id=None, smooth=False):
        '''
        Get the training loss
        '''
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]
        if fs:
            loss += self.config['training']['fs_weight'] * ret["fs_loss"]

        if smooth and self.config['training']['smooth_weight'] > 0:
            loss += self.config['training']['smooth_weight'] * self.smoothness(self.config['training']['smooth_pts'],
                                                                               self.config['training']['smooth_vox'],
                                                                               margin=self.config['training'][
                                                                                   'smooth_margin'])

        if use_continuous_pose and tracking:
            assert frame_id is None, "Frame id not given"
            loss += self.config["training"]["o_posenet_weight"] * self.o_posenet_loss(self.oposenet, frame_id)

            loss += self.config["training"]["i_posenet_weight"] * self.i_posenet_loss(self.iposenet, frame_id)

        return loss

    def first_frame_mapping(self, batch, n_iters=100):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float

        '''
        print('First frame mapping...')
        c2w = batch['c2w'][0].to(self.device)
        self.est_c2w_data[0] = c2w
        self.est_c2w_data_rel[0] = c2w

        self.model.train()

        # Training
        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])

            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.map_optimizer.step()

        # First frame will always be a keyframe
        self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
        if self.config['mapping']['first_mesh']:
            self.save_mesh(0)

        print('First frame mapping done')
        return ret, loss

    def current_frame_mapping(self, batch, cur_frame_id):
        '''
        Current frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float

        '''
        if self.config['mapping']['cur_frame_iters'] <= 0:
            return
        print('Current frame mapping...')

        if not self.config["tracking"]["use_continuous_pose"]:
            c2w = self.est_c2w_data[cur_frame_id].to(self.device)
        else:
            c2w = self.lie.se3_to_matrix(
                self.lie.matrix_to_se3(self.oposenet.forward(torch.tensor(cur_frame_id))) @ self.lie.matrix_to_se3(
                    self.iposenet.forward(torch.tensor(cur_frame_id)))).squeeze().to(self.device).detach()

        self.model.train()
        # Training
        for i in range(self.config['mapping']['cur_frame_iters']):
            self.cur_map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])

            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret, frame_id=cur_frame_id, tracking=False)
            loss.backward()
            self.cur_map_optimizer.step()

        return ret, loss

    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points - 1) * voxel_size
        offset_max = self.bounding_box[:, 1] - self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1, 1, 1, 3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        if self.config['grid']['tcnn_encoding']:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        sdf = self.model.query_sdf(pts_tcnn, embed=True)
        tv_x = torch.pow(sdf[1:, ...] - sdf[:-1, ...], 2).sum()
        tv_y = torch.pow(sdf[:, 1:, ...] - sdf[:, :-1, ...], 2).sum()
        tv_z = torch.pow(sdf[:, :, 1:, ...] - sdf[:, :, :-1, ...], 2).sum()

        loss = (tv_x + tv_y + tv_z) / (sample_points ** 3)

        return loss

    def i_posenet_loss(self, net, frame_id):
        times = torch.range(0, int(frame_id))
        poses = net(times)
        return self.get_dof_loss(poses)

    def my_l1_loss(self, tensor1, tensor2):
        """
        Calculate the L0 loss between two tensors.

        Parameters:
        - tensor1 (torch.Tensor): First tensor.
        - tensor2 (torch.Tensor): Second tensor.

        Returns:
        - int: L0 loss between the tensors.
        """

        return torch.mean(abs(tensor1 - tensor2))

    def get_dof_loss(self, poses):
        R = poses[:, :, :-1]
        angles = matrix_to_euler_angles(R, "XYZ")
        angles = angles / torch.norm(angles, dim=-1).view(-1, 1)
        V = poses[:, :, -1]
        v = V / torch.norm(V, dim=-1).view(-1, 1)

        return self.my_l1_loss(v, angles)

    def o_posenet_loss(self, net, frame_id):
        times = torch.range(0, int(frame_id))
        poses = net(times)
        return self.get_o_loss(poses)

    def get_o_loss(self, poses):
        v0 = poses[:, :, -1]
        oloss = torch.mean(torch.sum(abs(v0), dim=-1))
        return oloss

    def get_pose_param_optim(self, poses, mapping=True):
        task = 'mapping' if mapping else 'tracking'
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3]))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config[task]['lr_rot']},
                                           {"params": cur_trans, "lr": self.config[task]['lr_trans']}])

        return cur_rot, cur_trans, pose_optimizer

    def get_CBA_poses_loss(self, not_key_poses, frame_id):
        key_frame_ids_list = list(range(0, frame_id, self.config['mapping']['keyframe_every']))
        not_key_frame_ids_list = [i for i in range(frame_id) if i not in key_frame_ids_list]
        not_key_frame_ids = torch.tensor(not_key_frame_ids_list)
        poses = self.lie.se3_to_matrix(
            self.lie.matrix_to_se3(self.oposenet.forward(not_key_frame_ids)) @ self.lie.matrix_to_se3(
                self.iposenet.forward(not_key_frame_ids))).to(self.device).detach()
        return self.CBA_poses_loss(poses, not_key_poses)

    def CBA_poses_loss(self, poses1, poses2):
        """
            Calculate the loss between two sets of poses.

            Parameters:
            - poses1 (torch.Tensor): First set of poses with shape (batch_size, 3, 4).
            - poses2 (torch.Tensor): Second set of poses with shape (batch_size, 3, 4).

            Returns:
            - torch.Tensor: Loss between the poses.
            """
        assert poses1.shape == poses2.shape, "Poses must have the same shape"

        # 计算平移向量之差的 L1 范数
        translation_loss = torch.norm(poses1[:, :, 3] - poses2[:, :, 3], p=1, dim=1).mean()
        print("000000")
        print(translation_loss)

        # 计算旋转矩阵之差的 Frobenius 范数
        rotation_loss = torch.norm(poses1[:, :, :3] - poses2[:, :, :3], p='fro', dim=(1, 2)).mean()
        print("111111")
        print(rotation_loss)

        # 可以根据需要调整损失的组合方式
        total_loss = translation_loss * (1 - self.config["tracking"]["CBA_rotation_loss_weight"]) + rotation_loss * \
                     self.config["tracking"]["CBA_rotation_loss_weight"]

        return total_loss

    def global_CBA(self, batch, cur_frame_id):
        '''
        Global optimization for continuous pose situation.
        param:
        batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        :return: None
        '''
        key_frame_ids_list = list(range(0, cur_frame_id, self.config['mapping']['keyframe_every']))
        key_frame_ids = torch.tensor(key_frame_ids_list).to(self.device)
        not_key_frame_ids_list = [i for i in range(cur_frame_id) if i not in key_frame_ids_list]
        not_key_frame_ids = torch.tensor(not_key_frame_ids_list).to(self.device)

        poses_all = self.lie.se3_to_matrix(
            self.lie.matrix_to_se3(self.oposenet.forward(torch.tensor(range(cur_frame_id)))) @ self.lie.matrix_to_se3(
                self.iposenet.forward(torch.tensor(range(cur_frame_id)))))
        key_poses = self.lie.se3_to_matrix(
            self.lie.matrix_to_se3(self.oposenet.forward(key_frame_ids)) @ self.lie.matrix_to_se3(
                self.iposenet.forward(key_frame_ids))).to(self.device).detach()
        not_key_poses = self.lie.se3_to_matrix(
            self.lie.matrix_to_se3(self.oposenet.forward(not_key_frame_ids)) @ self.lie.matrix_to_se3(
                self.iposenet.forward(not_key_frame_ids))).to(self.device).detach()
        initial_not_key_poses = not_key_poses.clone()

        self.map_optimizer.zero_grad()

        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1])

        for i in range(self.config['mapping']['iters']):

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])

            # TODO: Checkpoint...
            idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W),
                                    max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids),
                                        self.config['mapping']['min_pixels_cur']))
            current_rays_batch = current_rays[idx_cur, :]

            rays = torch.cat([rays, current_rays_batch], dim=0)  # N, 7
            ids_all = torch.cat([ids // self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(
                torch.int64)

            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            # rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * key_poses[ids_all, None, :3, :3], -1)
            rays_o = key_poses[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            loss = self.get_loss_from_ret(ret, smooth=True, tracking=True)  # key_poses_loss

            loss += self.get_CBA_poses_loss(initial_not_key_poses, cur_frame_id) * 0.03 # not_key_poses_loss

            with torch.no_grad():
                l1_penalty = torch.tensor(0., requires_grad=False).to(self.device)
                for param in self.iposenet.parameters():
                    l1_penalty += torch.norm(param, p=1).to(self.device)
                for param in self.oposenet.parameters():
                    l1_penalty += torch.norm(param, p=1).to(self.device)

                loss += 0.01 * l1_penalty

            loss.backward(retain_graph=True)

            if (i + 1) % cfg["mapping"]["map_accum_step"] == 0:

                if (i + 1) > cfg["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()

            if (i + 1) % cfg["mapping"]["pose_accum_step"] == 0:
                self.iposenet.step()
                self.oposenet.step()

        if len(key_frame_ids_list) > 1:
            for i in range(len(key_frame_ids[1:])):
                self.est_c2w_data[int(key_frame_ids[i + 1].item())] = self.lie.matrix_to_se3(self.oposenet.forward(key_frame_ids[i + 1])) @ self.lie.matrix_to_se3(
                        self.iposenet.forward(key_frame_ids[i + 1])).detach().clone()[0]

            if self.config['mapping']['optim_cur']:
                print('Update current pose')
                self.est_c2w_data[cur_frame_id] = self.lie.matrix_to_se3(self.oposenet.forward(cur_frame_id)) @ self.lie.matrix_to_se3(
                        self.iposenet.forward(cur_frame_id)).detach().clone()[0]

    def global_BA(self, batch, cur_frame_id):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''
        pose_optimizer = None

        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack(
            [self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])

        # frame ids for all KFs, used for update poses after optimization
        frame_ids_all = torch.tensor(list(range(0, cur_frame_id, self.config['mapping']['keyframe_every'])))

        if len(self.keyframeDatabase.frame_ids) < 2:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None, ...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)

        else:
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None, ...]

            if self.config['mapping']['optim_cur']:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(torch.cat([poses[1:], current_pose]))
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

            else:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(poses[1:])
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)

        # Set up optimizer
        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()

        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1])

        for i in range(self.config['mapping']['iters']):

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])

            # TODO: Checkpoint...
            idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W),
                                    max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids),
                                        self.config['mapping']['min_pixels_cur']))
            current_rays_batch = current_rays[idx_cur, :]

            rays = torch.cat([rays, current_rays_batch], dim=0)  # N, 7
            ids_all = torch.cat([ids // self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(
                torch.int64)

            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            loss = self.get_loss_from_ret(ret, smooth=True)

            loss.backward(retain_graph=True)

            if (i + 1) % cfg["mapping"]["map_accum_step"] == 0:

                if (i + 1) > cfg["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()

            if pose_optimizer is not None and (i + 1) % cfg["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                # get SE3 poses to do forward pass
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)
                # So current pose is always unchanged
                if self.config['mapping']['optim_cur']:
                    poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

                else:
                    current_pose = self.est_c2w_data[cur_frame_id][None, ...]
                    # SE3 poses

                    poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)

                # zero_grad here
                pose_optimizer.zero_grad()

        if pose_optimizer is not None and len(frame_ids_all) > 1:
            for i in range(len(frame_ids_all[1:])):
                self.est_c2w_data[int(frame_ids_all[i + 1].item())] = \
                    self.matrix_from_tensor(cur_rot[i:i + 1], cur_trans[i:i + 1]).detach().clone()[0]

            if self.config['mapping']['optim_cur']:
                print('Update current pose')
                self.est_c2w_data[cur_frame_id] = \
                    self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]

    def predict_current_pose(self, frame_id, constant_speed=True):
        '''
        Predict current pose from previous pose using camera motion model
        '''
        if frame_id == 1 or (not constant_speed):
            c2w_est_prev = self.est_c2w_data[frame_id - 1].to(self.device)
            self.est_c2w_data[frame_id] = c2w_est_prev

        else:
            c2w_est_prev_prev = self.est_c2w_data[frame_id - 2].to(self.device)
            c2w_est_prev = self.est_c2w_data[frame_id - 1].to(self.device)
            print(c2w_est_prev.shape)
            print(c2w_est_prev_prev.shape)
            delta = c2w_est_prev @ c2w_est_prev_prev.float().inverse()
            self.est_c2w_data[frame_id] = delta @ c2w_est_prev

        return self.est_c2w_data[frame_id]

    def tracking_pc(self, batch, frame_id):
        '''
        Tracking camera pose of current frame using point cloud loss
        (Not used in the paper, but might be useful for some cases)
        '''

        c2w_gt = batch['c2w'][0].to(self.device)

        cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])

        cur_trans = torch.nn.parameter.Parameter(cur_c2w[..., :3, 3].unsqueeze(0))
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(cur_c2w[..., :3, :3]).unsqueeze(0))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config['tracking']['lr_rot']},
                                           {"params": cur_trans, "lr": self.config['tracking']['lr_trans']}])
        best_sdf_loss = None

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        thresh = 0

        if self.config['tracking']['iter_point'] > 0:
            indice_pc = self.select_samples(self.dataset.H - iH * 2, self.dataset.W - iW * 2,
                                            self.config['tracking']['pc_samples'])
            rays_d_cam = batch['direction'][:, iH:-iH, iW:-iW].reshape(-1, 3)[indice_pc].to(self.device)
            target_s = batch['rgb'][:, iH:-iH, iW:-iW].reshape(-1, 3)[indice_pc].to(self.device)
            target_d = batch['depth'][:, iH:-iH, iW:-iW].reshape(-1, 1)[indice_pc].to(self.device)

            valid_depth_mask = ((target_d > 0.) * (target_d < 5.))[:, 0]

            rays_d_cam = rays_d_cam[valid_depth_mask]
            target_s = target_s[valid_depth_mask]
            target_d = target_d[valid_depth_mask]

            for i in range(self.config['tracking']['iter_point']):
                pose_optimizer.zero_grad()
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                rays_o = c2w_est[..., :3, -1].repeat(len(rays_d_cam), 1)
                rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)
                pts = rays_o + target_d * rays_d

                pts_flat = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

                out = self.model.query_color_sdf(pts_flat)

                sdf = out[:, -1]
                rgb = torch.sigmoid(out[:, :3])

                # TODO: Change this
                loss = 5 * torch.mean(torch.square(rgb - target_s)) + 1000 * torch.mean(torch.square(sdf))

                if best_sdf_loss is None:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()

                with torch.no_grad():
                    c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                    if loss.cpu().item() < best_sdf_loss:
                        best_sdf_loss = loss.cpu().item()
                        best_c2w_est = c2w_est.detach()
                        thresh = 0
                    else:
                        thresh += 1
                if thresh > self.config['tracking']['wait_iters']:
                    break

                loss.backward()
                pose_optimizer.step()

        if self.config['tracking']['best']:
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

        if frame_id % self.config['mapping']['keyframe_every'] != 0:
            # Not a keyframe, need relative pose
            kf_id = frame_id // self.config['mapping']['keyframe_every']
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta
        print('Best loss: {}, Camera loss{}'.format(
            F.l1_loss(best_c2w_est.to(self.device)[0, :3], c2w_gt[:3]).cpu().item(),
            F.l1_loss(c2w_est[0, :3], c2w_gt[:3]).cpu().item()))

    def tracking_render(self, batch, frame_id):
        '''
        Tracking camera pose using of the current frame
        Params:
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            frame_id: Current frame id (int)
        '''

        c2w_gt = batch['c2w'][0].to(self.device)
        if self.config['tracking']['iter_point'] > 0:
            cur_c2w = self.est_c2w_data[frame_id]
        else:
            cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])

        if not self.config["tracking"]["use_continuous_pose"]:
            # Initialize current pose

            indice = None
            best_sdf_loss = None
            thresh = 0

            iW = self.config['tracking']['ignore_edge_W']
            iH = self.config['tracking']['ignore_edge_H']

            cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w[None, ...], mapping=False)

            # Start tracking
            for i in range(self.config['tracking']['iter']):
                pose_optimizer.zero_grad()
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                # Note here we fix the sampled points for optimisation
                if indice is None:
                    indice = self.select_samples(self.dataset.H - iH * 2, self.dataset.W - iW * 2,
                                                 self.config['tracking']['sample'])

                    # Slicing
                    indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (self.dataset.H - iH * 2)
                    rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
                target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
                target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)

                rays_o = c2w_est[..., :3, -1].repeat(self.config['tracking']['sample'], 1)
                rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

                ret = self.model.forward(rays_o, rays_d, target_s, target_d)
                loss = self.get_loss_from_ret(ret, frame_id=frame_id, tracking=True)

                if best_sdf_loss is None:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()

                with torch.no_grad():
                    c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                    if loss.cpu().item() < best_sdf_loss:
                        best_sdf_loss = loss.cpu().item()
                        best_c2w_est = c2w_est.detach()
                        thresh = 0
                    else:
                        thresh += 1

                if thresh > self.config['tracking']['wait_iters']:
                    break

                loss.backward()
                pose_optimizer.step()

            if self.config['tracking']['best']:
                # Use the pose with smallest loss
                self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
            else:
                # Use the pose after the last iteration
                self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

            # Save relative pose of non-keyframes
            if frame_id % self.config['mapping']['keyframe_every'] != 0:
                kf_id = frame_id // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                c2w_key = self.est_c2w_data[kf_frame_id]
                delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
                self.est_c2w_data_rel[frame_id] = delta

            print(
                'Best loss: {}, Last loss{}'.format(
                    F.l1_loss(best_c2w_est.to(self.device)[0, :3], c2w_gt[:3]).cpu().item(),
                    F.l1_loss(c2w_est[0, :3], c2w_gt[:3]).cpu().item()))

        else:
            indice = None
            best_sdf_loss = None
            thresh = 0

            iW = self.config['tracking']['ignore_edge_W']
            iH = self.config['tracking']['ignore_edge_H']

            times = torch.range(0, int(frame_id))
            train_iters = 10
            former_frames_id = torch.range(0, int(frame_id - 1))
            est_former_c2ws = self.lie.se3_to_matrix(
                self.lie.matrix_to_se3(self.oposenet.forward(torch.tensor(former_frames_id))) @ self.lie.matrix_to_se3(
                    self.iposenet.forward(torch.tensor(former_frames_id))))
            # for i in range(train_iters):
            #     est_oc2ws = self.oposenet.forward(times)
            #     est_ic2ws = self.iposenet.forward(times)
            #     est_c2ws = self.lie.se3_to_matrix(
            #         self.lie.matrix_to_se3(est_oc2ws) @ self.lie.matrix_to_se3(
            #             est_ic2ws)).squeeze()
            #     # dummy loss
            #     loss = torch.abs(est_former_c2ws - est_c2ws).mean()
            #     loss.backward(retain_graph=True)
            #     self.iposenet.step()
            #     self.oposenet.step()

            for i in range(self.config['tracking']['iter']):
                c2w_est = self.lie.se3_to_matrix(
                    self.lie.matrix_to_se3(self.oposenet.forward(torch.tensor(frame_id))) @ self.lie.matrix_to_se3(
                        self.iposenet.forward(torch.tensor(frame_id)))).to(self.device)

                # Note here we fix the sampled points for optimisation
                if indice is None:
                    indice = self.select_samples(self.dataset.H - iH * 2, self.dataset.W - iW * 2,
                                                 self.config['tracking']['sample'])

                    # Slicing
                    indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (self.dataset.H - iH * 2)
                    rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
                target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
                target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)

                rays_o = c2w_est[..., :3, -1].repeat(self.config['tracking']['sample'], 1)
                rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

                ret = self.model.forward(rays_o, rays_d, target_s, target_d)
                loss = self.get_loss_from_ret(ret, tracking=True)

                if best_sdf_loss is None:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()

                with torch.no_grad():
                    c2w_est = self.lie.se3_to_matrix(
                        self.lie.matrix_to_se3(self.oposenet.forward(torch.tensor(frame_id))) @ self.lie.matrix_to_se3(
                            self.iposenet.forward(torch.tensor(frame_id))))

                    if loss.cpu().item() < best_sdf_loss:
                        best_sdf_loss = loss.cpu().item()
                        best_c2w_est = c2w_est.detach()
                        thresh = 0
                    else:
                        thresh += 1

                if thresh > self.config['tracking']['wait_iters']:
                    break

                # 计算 L1 正则化项
                with torch.no_grad():
                    l1_penalty = torch.tensor(0., requires_grad=False).to(self.device)
                    for param in self.iposenet.parameters():
                        l1_penalty += torch.norm(param, p=1).to(self.device)
                    for param in self.oposenet.parameters():
                        l1_penalty += torch.norm(param, p=1).to(self.device)

                    loss += 0.01 * l1_penalty

                loss.backward()
                self.iposenet.step()
                self.oposenet.step()

            if self.config['tracking']['best']:
                # Use the pose with smallest loss
                self.est_c2w_data[frame_id] = (self.lie.matrix_to_se3(best_c2w_est)).detach().clone()[0]
            else:
                # Use the pose after the last iteration
                self.est_c2w_data[frame_id] = (self.lie.matrix_to_se3(c2w_est)).detach().clone()[0]

            # Save relative pose of non-keyframes
            if frame_id % self.config['mapping']['keyframe_every'] != 0:
                kf_id = frame_id // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                c2w_key = self.lie.matrix_to_se3(
                    self.oposenet.forward(torch.tensor(kf_frame_id))) @ self.lie.matrix_to_se3(
                    self.iposenet.forward(torch.tensor(kf_frame_id)))
                delta = self.lie.se3_to_matrix(self.est_c2w_data[frame_id] @ c2w_key.float().inverse())
                self.est_c2w_data_rel[frame_id] = delta

            print(
                'Best loss: {}, Last loss{}'.format(
                    F.l1_loss(best_c2w_est.to(self.device)[0, :3], c2w_gt[:3].to(self.device)).cpu().item(),
                    F.l1_loss(c2w_est[0, :3].to(self.device), c2w_gt[:3].to(self.device)).cpu().item()))

    def convert_relative_pose(self):
        poses = {}
        for i in range(len(self.est_c2w_data)):
            if i % self.config['mapping']['keyframe_every'] == 0:
                poses[i] = self.est_c2w_data[i]
            else:
                kf_id = i // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                c2w_key = self.est_c2w_data[kf_frame_id].to(self.device)
                delta = self.est_c2w_data_rel[i].to(self.device)
                poses[i] = delta @ c2w_key

        return poses

    def create_optimizer(self):
        '''
        Create optimizer for mapping
        '''
        # Optimizer for BA
        trainable_parameters = [{'params': self.model.decoder.parameters(), 'weight_decay': 1e-6,
                                 'lr': self.config['mapping']['lr_decoder']},
                                {'params': self.model.embed_fn.parameters(), 'eps': 1e-15,
                                 'lr': self.config['mapping']['lr_embed']}]

        if not self.config['grid']['oneGrid']:
            trainable_parameters.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15,
                                         'lr': self.config['mapping']['lr_embed_color']})

        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))

        # Optimizer for current frame mapping
        if self.config['mapping']['cur_frame_iters'] > 0:
            params_cur_mapping = [
                {'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
            if not self.config['grid']['oneGrid']:
                params_cur_mapping.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15,
                                           'lr': self.config['mapping']['lr_embed_color']})

            self.cur_map_optimizer = optim.Adam(params_cur_mapping, betas=(0.9, 0.99))

    def save_mesh(self, i, voxel_size=0.05):
        mesh_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'],
                                     'mesh_track{}.ply'.format(i))
        if self.config['mesh']['render_color']:
            color_func = self.model.render_surface_color
        else:
            color_func = self.model.query_color
        extract_mesh(self.model.query_sdf,
                     self.config,
                     self.bounding_box,
                     color_func=color_func,
                     marching_cube_bound=self.marching_cube_bound,
                     voxel_size=voxel_size,
                     mesh_savepath=mesh_savepath)

    def first_frame_pose_tracking(self, batch):
        l1_lambda = 0.1
        c2w = batch['c2w'][0].to(self.device)
        # train
        first_train_iters = 50
        print(first_train_iters, "000000000000000000000000000000")
        times = torch.tensor(0)
        # t = time.time()
        for i in range(first_train_iters):
            iest_c2ws = self.iposenet.forward(times)
            oest_c2ws = self.oposenet.forward(times)
            est_c2ws = self.lie.se3_to_matrix(
                self.lie.matrix_to_se3(oest_c2ws) @ self.lie.matrix_to_se3(iest_c2ws))
            # 计算 L1 正则化项
            with torch.no_grad():
                l1_penalty = torch.tensor(0., requires_grad=False).to(self.device)
                for param in self.iposenet.parameters():
                    l1_penalty += torch.norm(param, p=1).to(self.device)
                for param in self.oposenet.parameters():
                    l1_penalty += torch.norm(param, p=1).to(self.device)

            # 计算损失，加上 L1 正则化项
            # dummy loss
            loss = torch.abs(self.lie.se3_to_matrix(c2w[None, ...]).squeeze() - est_c2ws.squeeze()).mean().to(
                self.device)
            loss += l1_lambda * l1_penalty


            # dummy loss
            # loss = torch.abs(self.lie.se3_to_matrix(c2w[None, ...]).squeeze() - est_c2ws.squeeze()).mean().to(
            #     self.device)
            loss.backward()
            self.iposenet.step()
            self.oposenet.step()
            # print(f"step{i},loss{loss}")

        # print("time spent:", time.time() - t)


    def first_every_frame_pose_tracking(self, batch, i, iters):
        k = i - 1
        l1_lambda = 0.01
        c2w = self.lie.se3_to_matrix(
            self.lie.matrix_to_se3(self.oposenets[k // 50].forward(k)) @ self.lie.matrix_to_se3(
                self.iposenets[k // 50].forward(k))).to(self.device).detach()
        # train
        first_train_iters = 500
        times = torch.tensor(k)
        # t = time.time()
        for i in range(first_train_iters):
            iest_c2ws = self.iposenet.forward(times)
            oest_c2ws = self.oposenet.forward(times)
            est_c2ws = self.lie.se3_to_matrix(
                self.lie.matrix_to_se3(oest_c2ws) @ self.lie.matrix_to_se3(iest_c2ws)).to(self.device)
            # 计算 L1 正则化项
            with torch.no_grad():
                l1_penalty = torch.tensor(0., requires_grad=False).to(self.device)
                for param in self.iposenet.parameters():
                    l1_penalty += torch.norm(param, p=1).to(self.device)
                for param in self.oposenet.parameters():
                    l1_penalty += torch.norm(param, p=1).to(self.device)

            # 计算损失，加上 L1 正则化项
            # dummy loss
            loss = torch.abs(c2w[None, ...].squeeze() - est_c2ws.squeeze()).mean().to(
                self.device)
            loss += l1_lambda * l1_penalty

            loss.backward()
            self.iposenet.step()
            self.oposenet.step()

    def run(self):
        self.create_optimizer()
        data_loader = DataLoader(self.dataset, num_workers=self.config['data']['num_workers'])

        # Start Co-SLAM!
        for i, batch in tqdm(enumerate(data_loader)):
            # Visualisation
            if self.config['mesh']['visualisation']:
                rgb = cv2.cvtColor(batch["rgb"].squeeze().cpu().numpy(), cv2.COLOR_BGR2RGB)
                raw_depth = batch["depth"]
                mask = (raw_depth >= self.config["cam"]["depth_trunc"]).squeeze(0)
                depth_colormap = colormap_image(batch["depth"])
                depth_colormap[:, mask] = 255.
                depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()
                image = np.hstack((rgb, depth_colormap))
                cv2.namedWindow('RGB-D'.format(i), cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RGB-D'.format(i), image)
                key = cv2.waitKey(1)

            # First frame mapping
            if i == 0:
                self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
                self.first_frame_pose_tracking(batch)

            # Tracking + Mapping
            else:
                # if i % self.config["training"]["posenet_every"] == 0:
                #     print("000000")
                #     print("0000000")
                #     if i >= self.config["training"]["posenet_every"]:
                #         self.iposenets[i // self.config["training"]["posenet_every"] - 1].eval()
                #         self.first_every_frame_pose_tracking(batch, i, self.config['mapping']['every_first_iters'])
                #         # self.oposenet.load_state_dict(model1.state_dict())



                if self.config['tracking']['iter_point'] > 0:
                    self.tracking_pc(batch, i)
                self.tracking_render(batch, i)

                if i % self.config['mapping']['map_every'] == 0:
                    self.current_frame_mapping(batch, i)
                    if not self.config["tracking"]["use_continuous_pose"]:
                        self.global_BA(batch, i)
                    else:
                        # self.global_CBA(batch, i)
                        pass

                # Add keyframe
                if i % self.config['mapping']['keyframe_every'] == 0:
                    self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
                    print('add keyframe:', i)

                if i % self.config['mesh']['vis'] == 0:
                    self.save_mesh(i, voxel_size=self.config['mesh']['voxel_eval'])
                    pose_relative = self.convert_relative_pose()
                    pose_evaluation(self.pose_gt, self.est_c2w_data, 1,
                                    os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i)
                    # pose_evaluation(self.pose_gt, pose_relative, 1,
                    #                 os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i,
                    #                 img='pose_r', name='output_relative.txt')

                    if cfg['mesh']['visualisation']:
                        cv2.namedWindow('Traj:'.format(i), cv2.WINDOW_AUTOSIZE)
                        traj_image = cv2.imread(
                            os.path.join(self.config['data']['output'], self.config['data']['exp_name'],
                                         "pose_r_{}.png".format(i)))
                        # best_traj_image = cv2.imread(os.path.join(best_logdir_scene, "pose_r_{}.png".format(i)))
                        # image_show = np.hstack((traj_image, best_traj_image))
                        image_show = traj_image
                        cv2.imshow('Traj:'.format(i), image_show)
                        key = cv2.waitKey(1)

        model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'],
                                      'checkpoint{}.pt'.format(i))

        self.save_ckpt(model_savepath)
        self.save_mesh(i, voxel_size=self.config['mesh']['voxel_final'])

        pose_relative = self.convert_relative_pose()
        pose_evaluation(self.pose_gt, self.est_c2w_data, 1,
                        os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i)
        # pose_evaluation(self.pose_gt, pose_relative, 1,
        #                 os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i, img='pose_r',
        #                 name='output_relative.txt')

        # TODO: Evaluation of reconstruction


if __name__ == '__main__':

    print('Start running...')
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')

    args = parser.parse_args()

    cfg = config.load_config(args.config)
    if args.output is not None:
        cfg['data']['output'] = args.output

    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy("coslam.py", os.path.join(save_path, 'coslam.py'))

    with open(os.path.join(save_path, 'config.json'), "w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))

    slam = CoSLAM(cfg)

    slam.run()
