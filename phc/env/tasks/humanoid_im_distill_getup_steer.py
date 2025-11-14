# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import OrderedDict
import torch
import numpy as np
from phc.utils.torch_utils import quat_to_tan_norm
import phc.env.tasks.humanoid_im_getup as humanoid_im_getup
import phc.env.tasks.humanoid_im_distill as humanoid_im_distill
import phc.env.tasks.humanoid_im_distill_getup as humanoid_im_distill_getup
from phc.env.tasks.humanoid_amp import HumanoidAMP, remove_base_rot
from phc.utils.motion_lib_smpl import MotionLibSMPL 

from phc.utils import torch_utils
import torch.nn.functional as F
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from phc.utils.flags import flags
import joblib
import gc
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from rl_games.algos_torch import torch_ext
import torch.nn as nn
from phc.learning.network_loader import load_mcp_mlp, load_pnn
from collections import deque
import os
import time

from isaac_utils import torch_utils, rotations

ENABLE_RAND_HEADING = True
ENABLE_RAND_FACE_HEADING = False
TAR_SPEED_MIN, TAR_SPEED_MAX = 1.0, 1.8
CHANGE_STEPS_MIN, CHANGE_STEPS_MAX = 299, 300

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2

class HumanoidImDistillGetupSteer(humanoid_im_distill_getup.HumanoidImDistillGetup):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.total_num_objects = 0
        self.w_last = True  ## quaternion w_last
        self.planning_step = 28
        
        self.tar_speed_min = cfg.env.get("tar_speed_min", TAR_SPEED_MIN)
        self.tar_speed_max = cfg.env.get("tar_speed_max", TAR_SPEED_MAX)
        self.change_steps_min = cfg.env.get("change_steps_min", CHANGE_STEPS_MIN)
        self.change_steps_max = cfg.env.get("change_steps_max", CHANGE_STEPS_MAX)

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)

        self.dense_goal_positions = torch.zeros(
            [self.num_envs, self.planning_step, 3], device=self.device, dtype=torch.float
        )

        self.sparse_goal_positions = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )

        self._heading_change_steps = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )
        self._prev_root_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )

        self._tar_speed = (
            torch.ones([self.num_envs], device=self.device, dtype=torch.float) * 5
        )
        self._tar_dir = torch.zeros(
            [self.num_envs, 2], device=self.device, dtype=torch.float
        )
        self._tar_dir[..., 0] = 1.0

        self._tar_facing_dir = torch.zeros(
            [self.num_envs, 2], device=self.device, dtype=torch.float
        )
        self._tar_facing_dir[..., 0] = 1.0

        self._heading_turn_steps = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )

        self._failures = []
        self._distances = []
        self._current_accumulated_errors = (
            torch.zeros([self.num_envs], device=self.device, dtype=torch.float) - 1
        )
        self._current_failures = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.float
        )
        self._last_length = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.long
        )

        # Trajectory collection (same format as humanoid_speed_any)
        self._collect_joint_positions = getattr(cfg.env, "collect_trajectories", False) or getattr(cfg.env, "collect_joint_positions", False)
        self._joint_positions_buffer = []
        self._joint_positions_start_idx = [0] * self.num_envs
        self._max_episodes_to_collect = getattr(cfg.env, "maxEpisodesToCollect", 1000)
        self._total_episodes_collected = 0
        self._episode_id_per_env = [0] * self.num_envs
        self._episode_status_per_env = [None] * self.num_envs
        self._episode_start_time = [None] * self.num_envs
        self._all_trajectories = []
        self._trajectory_save_dir = getattr(cfg.env, "trajectory_save_dir", "output/UniPhys/steer")
        # When True, speed/direction change triggers a full scene reset (humanoid state reset)
        # after saving the current segment, so each segment starts from a fresh reset state.
        self._reset_scene_on_speed_change = getattr(cfg.env, "resetSceneOnSpeedChange", False)
        if self._collect_joint_positions:
            print(f"[DEBUG] Steer trajectory collection ENABLED: maxEpisodesToCollect={self._max_episodes_to_collect}, save_dir={self._trajectory_save_dir}", flush=True)
        else:
            print(f"[DEBUG] Steer trajectory collection OFF (collect_trajectories=False). Set +phc.env.collect_trajectories=True to collect.", flush=True)

        if not self.headless:
            self._build_marker_state_tensors()

        
        return

    def set_trajectory_save_dir(self, save_dir):
        self._trajectory_save_dir = save_dir

    def set_episode_status(self, env_ids, episode_status):
        if isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.detach().cpu().tolist()
        else:
            env_ids_list = list(env_ids)
        for env_id in env_ids_list:
            self._episode_status_per_env[env_id] = episode_status

    def finalize_episode_trajectory(self, env_ids):
        if not self._collect_joint_positions or env_ids is None:
            return
        n_envs = env_ids.numel() if isinstance(env_ids, torch.Tensor) else len(env_ids)
        print(f"[DEBUG] Steer: finalize_episode_trajectory called for {n_envs} env(s), buffer size={len(self._joint_positions_buffer)}", flush=True)
        self._save_joint_positions(env_ids)
        env_ids_list = env_ids.detach().cpu().tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        for env_id in env_ids_list:
            self._episode_start_time[env_id] = None
            self._episode_status_per_env[env_id] = None
            self._joint_positions_start_idx[env_id] = len(self._joint_positions_buffer)
            self._episode_id_per_env[env_id] = self._episode_id_per_env[env_id] + 1

    ###############################################################
    # Set up IsaacGym environment
    ###############################################################
    def _create_envs(self, num_envs, spacing, num_per_row):
        if not self.headless:
            self._marker_handles = []
            self._face_marker_handles = []
            self._load_marker_asset()

        super()._create_envs(num_envs, spacing, num_per_row)

    def _load_marker_asset(self):
        asset_root = "phc/data/assets/urdf/"

        asset_file = "heading_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

    def build_env(self, env_id, env_ptr, humanoid_asset):
        super().build_env(env_id, env_ptr, humanoid_asset)

        if not self.headless:
            self._build_marker(env_id, env_ptr)

    def _build_marker(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0
        default_pose.p.z = 0.0

        marker_handle = self.gym.create_actor(
            env_ptr,
            self._marker_asset,
            default_pose,
            "marker",
            col_group,
            col_filter,
            segmentation_id,
        )
        self.gym.set_rigid_body_color(
            env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0)
        )
        self._marker_handles.append(marker_handle)

        face_marker_handle = self.gym.create_actor(
            env_ptr,
            self._marker_asset,
            default_pose,
            "face_marker",
            col_group,
            col_filter,
            segmentation_id,
        )
        self.gym.set_rigid_body_color(
            env_ptr,
            face_marker_handle,
            0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(0.0, 0.0, 0.8),
        )
        self._face_marker_handles.append(face_marker_handle)

    def _build_marker_state_tensors(self):
        num_actors = self.get_num_actors_per_env()
        if self.total_num_objects > 0:
            self._marker_states = self._root_states[: -self.total_num_objects].view(
                self.num_envs, num_actors, self._root_states.shape[-1]
            )[..., TAR_ACTOR_ID, :]
            self._face_marker_states = self._root_states[: -self.total_num_objects].view(
                self.num_envs, num_actors, self._root_states.shape[-1]
            )[..., TAR_FACING_ACTOR_ID, :]
        else:
            self._marker_states = self._root_states.view(
                self.num_envs, num_actors, self._root_states.shape[-1]
            )[..., TAR_ACTOR_ID, :]
            self._face_marker_states = self._root_states.view(
                self.num_envs, num_actors, self._root_states.shape[-1]
            )[..., TAR_FACING_ACTOR_ID, :]

        self._marker_pos = self._marker_states[..., :3]
        self._marker_rot = self._marker_states[..., 3:7]
        self._marker_actor_ids = self._humanoid_actor_ids + TAR_ACTOR_ID

        self._face_marker_pos = self._face_marker_states[..., :3]
        self._face_marker_rot = self._face_marker_states[..., 3:7]
        self._face_marker_actor_ids = self._humanoid_actor_ids + TAR_FACING_ACTOR_ID


    ###############################################################
    # Helpers
    ###############################################################
    
    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] + 3.0, 5.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 0.0)

        # cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 10.0)
        # cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0)
        if self.viewer:
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[self.viewing_env_idx, 0:3].cpu().numpy()

        if self.viewer:
            cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
            cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        else:
            cam_pos = np.array([char_root_pos[0] + 2.5, char_root_pos[1] + 2.5, char_root_pos[2]])

        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], char_root_pos[2])
        # if np.abs(cam_pos[2] - char_root_pos[2]) > 5:
        cam_pos[2] = char_root_pos[2] + 0.5
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2])

        self.gym.set_camera_location(self.recorder_camera_handle, self.envs[self.viewing_env_idx], new_cam_pos, new_cam_target)

        if flags.follow:
            self.start = True
        else:
            self.start = False

        if self.start:
            self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_marker(self):
        """
        input: humanoid_root_pos, tar_dir, tar_facing_dir
        """
        humanoid_root_pos = self._humanoid_root_states[..., :3].clone()
        self._marker_pos[..., 0:2] = humanoid_root_pos[..., :2] + self._tar_dir[:]

        height_below_marker = humanoid_root_pos[..., -1:]
        self._marker_pos[..., -1:] = height_below_marker

        heading_theta = torch.atan2(self._tar_dir[..., 1], self._tar_dir[..., 0])
        heading_axis = torch.zeros_like(self._marker_pos)
        heading_axis[..., -1] = 1.0
        heading_q = rotations.quat_from_angle_axis(
            heading_theta, heading_axis, w_last=self.w_last
        )
        self._marker_rot[:] = heading_q

        self._face_marker_pos[..., :2] = (
            humanoid_root_pos[..., :2] + self._tar_facing_dir[:]
        )
        self._face_marker_pos[..., -1:] = height_below_marker

        face_theta = torch.atan2(
            self._tar_facing_dir[..., 1], self._tar_facing_dir[..., 0]
        )
        face_axis = torch.zeros_like(self._marker_pos)
        face_axis[..., -1] = 1.0
        face_q = rotations.quat_from_angle_axis(
            face_theta, heading_axis, w_last=self.w_last
        )
        self._face_marker_rot[:] = face_q

        marker_ids = torch.cat(
            [self._marker_actor_ids, self._face_marker_actor_ids], dim=0
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(marker_ids),
            len(marker_ids),
        )

    def _draw_task(self):
        self._update_marker()

        vel_scale = 0.2
        heading_cols = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        root_pos = self._humanoid_root_states[..., 0:3]
        prev_root_pos = self._prev_root_pos[:]
        sim_vel = (root_pos - prev_root_pos) / self.dt
        sim_vel[..., -1] = 0

        starts = root_pos
        tar_ends = torch.clone(starts)
        tar_ends[..., 0:2] += (
            vel_scale * self._tar_speed[:].unsqueeze(-1) * self._tar_dir[:]
        )
        sim_ends = starts + vel_scale * sim_vel

        verts = torch.cat([starts, tar_ends, starts, sim_ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i : i + 1]
            curr_verts = curr_verts.reshape([2, 6])
            self.gym.add_lines(
                self.viewer, env_ptr, curr_verts.shape[0], curr_verts, heading_cols
            )

    def post_physics_step(self):
        super().post_physics_step()
        if not self._collect_joint_positions or self._total_episodes_collected >= self._max_episodes_to_collect:
            return
        current_time = time.time()
        for env_id in range(self.num_envs):
            if self._episode_start_time[env_id] is None:
                self._episode_start_time[env_id] = current_time
        num_bodies = self._rigid_body_pos.shape[1]
        n_joints = min(24, num_bodies)
        joint_pos = self._rigid_body_pos[:, :n_joints, :].cpu().numpy()
        joint_rot = self._rigid_body_rot[:, :n_joints, :].cpu().numpy()
        root_pos = self._humanoid_root_states[:, 0:3].cpu().numpy()
        root_rot = self._humanoid_root_states[:, 3:7].cpu().numpy()
        root_vel = self._humanoid_root_states[:, 7:10].cpu().numpy()
        root_ang_vel = self._humanoid_root_states[:, 10:13].cpu().numpy()
        dof_pos = self._dof_pos.cpu().numpy()
        dof_vel = self._dof_vel.cpu().numpy()
        tar_speed_frame = self._tar_speed.cpu().numpy()
        tar_dir_frame = self._tar_dir.cpu().numpy()
        tar_facing_dir_frame = self._tar_facing_dir.cpu().numpy()
        frame_data = {
            "rigid_body_pos": joint_pos,
            "rigid_body_rot": joint_rot,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "root_vel": root_vel,
            "root_ang_vel": root_ang_vel,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "tar_speed": tar_speed_frame,
            "tar_dir": tar_dir_frame,
            "tar_facing_dir": tar_facing_dir_frame,
        }
        self._joint_positions_buffer.append(frame_data)

###############################################################
# Handle resets
###############################################################
    def reset(self, env_ids=None):
        if self._collect_joint_positions and env_ids is not None:
            env_ids_list = env_ids.detach().cpu().tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
            if len(env_ids_list) > 0 and self._total_episodes_collected < self._max_episodes_to_collect:
                self._save_joint_positions(env_ids)
            for env_id in env_ids_list:
                self._episode_start_time[env_id] = None
                self._episode_status_per_env[env_id] = None
                self._joint_positions_start_idx[env_id] = len(self._joint_positions_buffer)
                self._episode_id_per_env[env_id] = self._episode_id_per_env[env_id] + 1
        super().reset(env_ids=env_ids)

    def reset_task(self, env_ids):
        # Save trajectory on speed/direction change (same as humanoid_speed_any),
        # but only if we actually have frames collected for that env.
        if len(env_ids) > 0 and self._collect_joint_positions and self._total_episodes_collected < self._max_episodes_to_collect:
            env_ids_list = env_ids.detach().cpu().tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
            # Avoid saving at initialization (progress_buf==0) or when buffer has no new frames yet.
            saveable = []
            buf_len = len(self._joint_positions_buffer)
            for eid in env_ids_list:
                if buf_len == 0:
                    continue
                if int(self.progress_buf[eid].item()) == 0:
                    continue
                if self._joint_positions_start_idx[eid] >= buf_len:
                    continue
                saveable.append(eid)
            if len(saveable) > 0:
                save_env_ids = torch.tensor(saveable, device=self.device, dtype=torch.long)
                for eid in saveable:
                    self._episode_status_per_env[eid] = "speed_change"
                self._save_joint_positions(save_env_ids)
                for eid in saveable:
                    self._episode_start_time[eid] = None
                    self._episode_status_per_env[eid] = None
                    self._joint_positions_start_idx[eid] = len(self._joint_positions_buffer)
                    self._episode_id_per_env[eid] = self._episode_id_per_env[eid] + 1
        if len(env_ids) > 0:
            # Make sure the test has started + agent started from a valid position (if it failed, then it's not valid)
            active_envs = (self._current_accumulated_errors[env_ids] > 0) & (
                (self._last_length[env_ids] - self._heading_turn_steps[env_ids]) > 0
            )
            average_distances = self._current_accumulated_errors[env_ids][
                active_envs
            ] / (
                self._last_length[env_ids][active_envs]
                - self._heading_turn_steps[env_ids][active_envs]
            )
            self._distances.extend(average_distances.cpu().tolist())
            self._current_accumulated_errors[env_ids] = 0
            self._failures.extend(
                (self._current_failures[env_ids][active_envs] > 0).cpu().tolist()
            )
            self._current_failures[env_ids] = 0

        # Optional: reset humanoid/scene state on speed change (so each segment starts fresh).
        # Use super().reset (NOT self.reset) to avoid double-saving in this class's reset().
        if len(env_ids) > 0 and getattr(self, "_reset_scene_on_speed_change", False):
            super().reset(env_ids=env_ids)

        # super().reset_task(env_ids)
        n = len(env_ids)
        if ENABLE_RAND_HEADING:
            rand_theta = (2 * np.pi * torch.rand(n, device=self.device) - np.pi) #[-pi, pi]
            # rand_theta = np.pi * torch.rand(n, device=self.device) #[0, pi]
        else:
            rand_theta = torch.ones(n, device=self.device) * np.pi * 0.5

        if ENABLE_RAND_FACE_HEADING:
            rand_theta_face = (2 * np.pi * torch.rand(n, device=self.device) - np.pi) #[-pi, pi]
        else:
            rand_theta_face = torch.ones(n, device=self.device) * np.pi * 0.5

        tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)
        tar_face_dir = torch.stack([torch.cos(rand_theta_face), torch.sin(rand_theta_face)], dim=-1) if ENABLE_RAND_FACE_HEADING else tar_dir
        tar_speed = (
            self.tar_speed_max
            - self.tar_speed_min
        ) * torch.rand(n, device=self.device) + self.tar_speed_min
        change_steps = torch.randint(
            low=self.change_steps_min,
            high=self.change_steps_max,
            size=(n,),
            device=self.device,
            dtype=torch.int64,
        )

        self._tar_speed[env_ids] = tar_speed
        self._tar_dir[env_ids] = tar_dir
        self._tar_facing_dir[env_ids] = tar_face_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        self._heading_turn_steps[env_ids] = 60 * 1 + self.progress_buf[env_ids]
        self._prev_root_pos[:, :2] = self.get_bodies_state().body_pos[:, 0, :2]

        # # dense goal
        time_steps = torch.linspace(0, self.planning_step, steps=self.planning_step).cuda()
        dense_goal_offset = time_steps.view(1, -1, 1) * self._tar_speed[env_ids] * self._tar_dir[env_ids].view(1, 1, -1) / 30  # n, t, d

        # # sparse goal
        sparse_goal_offset = self._tar_speed[env_ids] * self._tar_dir[env_ids] * self.planning_step / 30

        self.dense_goal_positions[env_ids, ..., 0] =  dense_goal_offset[..., 0]
        self.dense_goal_positions[env_ids, ..., 1] =  dense_goal_offset[..., 1]
        self.dense_goal_positions[env_ids, ..., -1] =  0

        self.sparse_goal_positions[env_ids, 0] =  sparse_goal_offset[..., 0]
        self.sparse_goal_positions[env_ids, 1] =  sparse_goal_offset[..., 1]
        self.sparse_goal_positions[env_ids,  -1] =  0


    def update_task(self, ):
        reset_task_mask = (self.progress_buf >= self._heading_change_steps or self.progress_buf == 0)
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self.reset_task(rest_env_ids)
    

    def _save_joint_positions(self, env_ids):
        if len(self._joint_positions_buffer) == 0:
            print("[WARNING] Steer: joint positions buffer empty, no trajectory to save. (post_physics_step may not be appending?)", flush=True)
            return
        if isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.detach().cpu().tolist()
        elif isinstance(env_ids, np.ndarray):
            env_ids_list = env_ids.tolist()
        else:
            env_ids_list = list(env_ids)
        if len(env_ids_list) == 0 or self._total_episodes_collected >= self._max_episodes_to_collect:
            return
        os.makedirs(self._trajectory_save_dir, exist_ok=True)
        timestamp = int(time.time())
        for env_id in env_ids_list:
            if self._total_episodes_collected >= self._max_episodes_to_collect:
                break
            start_idx = self._joint_positions_start_idx[env_id]
            if start_idx >= len(self._joint_positions_buffer):
                continue
            rigid_body_pos_all = []
            rigid_body_rot_all = []
            root_pos_all = []
            root_rot_all = []
            root_vel_all = []
            root_ang_vel_all = []
            dof_pos_all = []
            dof_vel_all = []
            tar_speed_all = []
            tar_dir_all = []
            tar_facing_dir_all = []
            for frame_data in self._joint_positions_buffer[start_idx:]:
                rigid_body_pos_all.append(frame_data["rigid_body_pos"][env_id])
                rigid_body_rot_all.append(frame_data["rigid_body_rot"][env_id])
                root_pos_all.append(frame_data["root_pos"][env_id])
                root_rot_all.append(frame_data["root_rot"][env_id])
                root_vel_all.append(frame_data["root_vel"][env_id])
                root_ang_vel_all.append(frame_data["root_ang_vel"][env_id])
                dof_pos_all.append(frame_data["dof_pos"][env_id])
                dof_vel_all.append(frame_data["dof_vel"][env_id])
                tar_speed_all.append(frame_data["tar_speed"][env_id])
                tar_dir_all.append(frame_data["tar_dir"][env_id])
                tar_facing_dir_all.append(frame_data["tar_facing_dir"][env_id])
            rigid_body_pos_stacked = np.stack(rigid_body_pos_all, axis=0)
            rigid_body_rot_stacked = np.stack(rigid_body_rot_all, axis=0)
            root_pos_stacked = np.stack(root_pos_all, axis=0)
            root_rot_stacked = np.stack(root_rot_all, axis=0)
            root_vel_stacked = np.stack(root_vel_all, axis=0)
            root_ang_vel_stacked = np.stack(root_ang_vel_all, axis=0)
            dof_pos_stacked = np.stack(dof_pos_all, axis=0)
            dof_vel_stacked = np.stack(dof_vel_all, axis=0)
            tar_speed_stacked = np.stack(tar_speed_all, axis=0)
            tar_dir_stacked = np.stack(tar_dir_all, axis=0)
            tar_facing_dir_stacked = np.stack(tar_facing_dir_all, axis=0)
            episode_status = self._episode_status_per_env[env_id] or "unknown"
            episode_id = self._episode_id_per_env[env_id]
            num_frames = rigid_body_pos_stacked.shape[0]
            start_time = self._episode_start_time[env_id]
            elapsed_wall_time = (time.time() - start_time) if start_time is not None else num_frames * self.dt
            fps_measured = num_frames / elapsed_wall_time if elapsed_wall_time > 0 else 0.0
            total_time_sim = num_frames * self.dt
            fps_sim = 1.0 / self.dt if self.dt > 0 else 0.0
            tar_speed_np = float(self._tar_speed[env_id].detach().cpu().item())
            tar_dir_np = self._tar_dir[env_id].detach().cpu().numpy()
            tar_facing_dir_np = self._tar_facing_dir[env_id].detach().cpu().numpy()
            target_velocity_2d = tar_dir_np * tar_speed_np
            trajectory_data = {
                "rigid_body_pos": rigid_body_pos_stacked,
                "rigid_body_rot": rigid_body_rot_stacked,
                "root_pos": root_pos_stacked,
                "root_rot": root_rot_stacked,
                "root_vel": root_vel_stacked,
                "root_ang_vel": root_ang_vel_stacked,
                "dof_pos": dof_pos_stacked,
                "dof_vel": dof_vel_stacked,
                "tar_speed": tar_speed_np,
                "tar_dir": tar_dir_np,
                "tar_facing_dir": tar_facing_dir_np,
                "any_direction": True,
                "separate_facing_dir": True,
                "target_velocity_2d": target_velocity_2d,
                "tar_speed_stacked": tar_speed_stacked,
                "tar_dir_stacked": tar_dir_stacked,
                "tar_facing_dir_stacked": tar_facing_dir_stacked,
                "dt": self.dt,
                "num_envs": 1,
                "num_frames": num_frames,
                "total_time": elapsed_wall_time,
                "fps": fps_measured,
                "total_time_sim": total_time_sim,
                "fps_sim": fps_sim,
                "env_id": env_id,
                "episode_id": episode_id,
                "episode_status": episode_status,
                "total_episode_index": self._total_episodes_collected,
            }
            self._all_trajectories.append(trajectory_data)
            self._total_episodes_collected += 1
            print(f"[DEBUG] Steer saved trajectory {self._total_episodes_collected}/{self._max_episodes_to_collect}: env {env_id}, episode_id {episode_id}, status={episode_status}, frames={num_frames}", flush=True)
            self._joint_positions_start_idx[env_id] = len(self._joint_positions_buffer)
        if self._total_episodes_collected >= self._max_episodes_to_collect and len(self._all_trajectories) > 0:
            print(f"\n[SAVE] Steer: reached collection limit. Saving {len(self._all_trajectories)} trajectories...")
            self._save_all_trajectories(self._trajectory_save_dir, timestamp)
        min_start = min(self._joint_positions_start_idx)
        if min_start > 0:
            self._joint_positions_buffer = self._joint_positions_buffer[min_start:]
            self._joint_positions_start_idx = [idx - min_start for idx in self._joint_positions_start_idx]

    def _save_all_trajectories(self, save_dir, timestamp):
        if len(self._all_trajectories) == 0:
            return
        save_path = os.path.join(save_dir, f"speed_steer_{timestamp}.pkl")
        total_trajectories = len(self._all_trajectories)
        total_frames = sum(t["num_frames"] for t in self._all_trajectories)
        total_wall_time = sum(t["total_time"] for t in self._all_trajectories)
        status_counts = {}
        for t in self._all_trajectories:
            s = t.get("episode_status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1
        all_data = {
            "trajectories": self._all_trajectories,
            "metadata": {
                "num_trajectories": total_trajectories,
                "total_frames": total_frames,
                "total_wall_time": total_wall_time,
                "avg_frames_per_trajectory": total_frames / total_trajectories if total_trajectories > 0 else 0,
                "avg_wall_time_per_trajectory": total_wall_time / total_trajectories if total_trajectories > 0 else 0,
                "status_counts": status_counts,
                "dt": self.dt,
                "timestamp": timestamp,
                "fps": total_frames / total_wall_time if total_wall_time > 0 else 0,
            },
        }
        joblib.dump(all_data, save_path)
        print(f"\n[SUCCESS] Steer saved {total_trajectories} trajectories ({total_frames} frames) to {os.path.abspath(save_path)}")
        print(f"         Status breakdown: {status_counts}\n")
        self._all_trajectories = []

    def save_remaining_trajectories(self):
        if len(self._all_trajectories) == 0:
            print("[DEBUG] Steer: save_remaining_trajectories called but no trajectories in memory (none saved yet this run).", flush=True)
            return
        os.makedirs(self._trajectory_save_dir, exist_ok=True)
        timestamp = int(time.time())
        self._save_all_trajectories(self._trajectory_save_dir, timestamp)

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self.update_task()

    """
    useless
    """
    def update_goal_state(self, goal_pos):
        """
        goal_pos: if sparse: [B, 3]; else [B, N, 3]
        """
        if goal_pos.ndim == 2:
            self.goal_pos = goal_pos.unsqueeze(1) # [B, 1, 3]
        self._num_goals = self.goal_pos.shape[1]
        # assert self._num_goals <= self._max_num_goals

    def capture_new_goal_from_keyborad(self):
        T = 28
        print("\033[1mCurrent root position: {}\033[0m".format(self._root_states[self._humanoid_actor_ids][..., :3].cpu().numpy()))
        text = input("\033[34mEnter the text prompt: \033[0m")
        velocity = float(input("\033[34mEnter velocity: \033[0m"))
        steer = input("\033[34mEnter xy steering vector separated by a space: \033[0m")
        velocity_vec = input("\033[34mEnter xy velocity vector separated by a space: \033[0m")
        curr_root_pos = self._root_states[self._humanoid_actor_ids][..., :3].clone()
        try:
            steer_x, steer_y = map(float, steer.split())
            steer_vector = torch.tensor([steer_x, steer_y, 0.0]).cuda().float()

            vel_x, vel_y = map(float, velocity_vec.split())
            vel_vector = torch.tensor([vel_x, vel_y, 0.0]).cuda().float()


            print("\033[32mSetting text command: {}\033[0m".format(text))
            print("\033[32mSetting velocity value = {}, vec = {}\033[0m".format(velocity, vel_vector.cpu().numpy()))
            print("\033[32mSetting steering: {}\033[0m".format(steer_vector.cpu().numpy()))

            # # dense goal
            time_steps = torch.linspace(0, T, steps=T).cuda()
            new_goal_position_offset = time_steps.view(1, -1, 1, 1) * (velocity * vel_vector.view(1, 1, 1, -1)) / 30

            # sparse goal
            # new_goal_position_offset = velocity * vel_vector.view(1, 1, 1, -1) * T / 30 * 1

            self.update_goal_state(new_goal_position_offset)
        except ValueError:
            print("Invalid input. Please enter two numbers separated by a space. The goal position remains unchanged")
            print("\033[32mSetting the goal position as {}\033[0m".format(new_goal_position.cpu().numpy()))
            self.update_goal_state(new_goal_position_offset)
        
        self.velocity = velocity
        self.steer_vector = steer_vector
        self.vel_vector = vel_vector
        self.goal_pos = new_goal_position_offset
        
        return text, new_goal_position_offset
    
    def _draw_task(self):
        # uncomment it if you want to see the marker
        self._update_marker()
        return

