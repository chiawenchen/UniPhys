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

class HumanoidImDistillGetupGoal(humanoid_im_distill_getup.HumanoidImDistillGetup):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._max_num_goals = 1
        self._num_goals = 0
        self.joint_idx = cfg.env.joint_idx  # [0: Pelvis; 18: L_Hand, 23: R_Hand, 13: Head]
        self.device_type = device_type
        self._goal_pos = None
        self.tar_min = cfg.env.tar_min
        self.tar_max = cfg.env.tar_max
        self.tar_min_height = cfg.env.tar_min_height
        self.tar_max_height = cfg.env.tar_max_height
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        self._goal_pos = torch.zeros([self.num_envs, 3], dtype=torch.float, device="cuda")

        # Trajectory collection (same format as humanoid_reach)
        self._collect_joint_positions = getattr(cfg.env, "collect_trajectories", False) or getattr(cfg.env, "collect_joint_positions", False)
        self._joint_positions_buffer = []
        self._joint_positions_start_idx = [0] * self.num_envs
        self._max_goals_to_collect = getattr(cfg.env, "maxGoalsToCollect", 1000)
        self._total_goals_collected = 0
        self._goal_id_per_env = [0] * self.num_envs
        self._goal_status_per_env = [None] * self.num_envs
        self._episode_start_time = [None] * self.num_envs
        self._all_trajectories = []
        self._trajectory_save_dir = getattr(cfg.env, "trajectory_save_dir", "output/UniPhys/goal_reaching")
        if self._collect_joint_positions:
            print(f"[DEBUG] Trajectory collection enabled: maxGoalsToCollect={self._max_goals_to_collect}, save_dir={self._trajectory_save_dir}")

        return

    def set_trajectory_save_dir(self, save_dir):
        """Allow algorithm to set save directory (e.g. to match algorithm save path)."""
        self._trajectory_save_dir = save_dir

    def set_episode_status(self, env_ids, goal_status):
        """Set goal_status for envs before reset, so _save_joint_positions can record it."""
        if isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.detach().cpu().tolist()
        else:
            env_ids_list = list(env_ids)
        for env_id in env_ids_list:
            self._goal_status_per_env[env_id] = goal_status

        return

    def finalize_episode_trajectory(self, env_ids):
        """Save trajectory for the given envs (call before env_reset when episode ended)."""
        if not self._collect_joint_positions or env_ids is None:
            return
        self._save_joint_positions(env_ids)
        env_ids_list = env_ids.detach().cpu().tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        for env_id in env_ids_list:
            self._episode_start_time[env_id] = None
            self._goal_status_per_env[env_id] = None
            self._joint_positions_start_idx[env_id] = len(self._joint_positions_buffer)
            self._goal_id_per_env[env_id] = self._goal_id_per_env[env_id] + 1
        return

    def _build_marker(self, env_id, env_ptr):
        default_pose = gymapi.Transform()

        for i in range(self._max_num_goals):

            marker_handle = self.gym.create_actor(
                env_ptr,
                self._marker_asset,
                default_pose,
                "marker",
                self.num_envs + 10,
                0,
                0,
            )
            self.gym.set_rigid_body_color(
                env_ptr,
                marker_handle,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(1.0, 0.0, 0.0),
            )
            self._marker_handles[env_id].append(marker_handle)

    def _build_marker_state_tensors(self):
        num_actors = self.get_num_actors_per_env()
        
        self._marker_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1]
        )[..., 1 : (1 + self._max_num_goals), :]
        self._marker_pos = self._marker_states[..., :3]

        self._marker_actor_ids = self._humanoid_actor_ids.unsqueeze(
            -1
        ) + torch_utils.to_torch(
            self._marker_handles, dtype=torch.int32, device=self.device
        )
        self._marker_actor_ids = self._marker_actor_ids.flatten()


    ###############################################################
    # Helpers
    ###############################################################
    def _load_marker_asset(self):
        asset_root = "phc/data/assets/urdf/"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.density = 0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, "traj_marker_large.urdf", asset_options)
        
        self._marker_asset_small = self.gym.load_asset(self.sim, asset_root, "traj_marker_small.urdf", asset_options)

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


    def update_goal_state(self, goal_pos):
        """
        goal_pos: if sparse: [B, 3]; else [B, N, 3]
        """
        if goal_pos.ndim == 2:
            self._goal_pos = goal_pos.unsqueeze(1) # [B, 1, 3]
        self._num_goals = self._goal_pos.shape[1]
        assert self._num_goals <= self._max_num_goals

    def set_random_goal(self, x=None, y=None, z=None):
        n = self.num_envs
        # _tar_dist_min, _tar_dist_max = self.state_machine_conditions[task]['tar_dist_range']
        
        goal_pos = torch.zeros([n, 3], dtype=torch.float, device="cuda")

        if x is not None and y is not None and z is not None:
            goal_pos[:, 0] = x
            goal_pos[:, 1] = y
            goal_pos[:, 2] = z
        else:
            _tar_dist_min, _tar_dist_max = self.tar_min, self.tar_max
            _tar_height_min, _tar_height_max = self.tar_min_height, self.tar_max_height
            rand_dist = (_tar_dist_max - _tar_dist_min) * torch.rand([n], dtype=float, device=self.device_type) + _tar_dist_min
            rand_height = (_tar_height_max - _tar_height_min) * torch.rand([n], dtype=float, device=self.device_type) + _tar_height_min
            rand_theta = 0.5 * np.pi * torch.rand([n], dtype=float, device=self.device_type) + 0.5 * np.pi
            goal_pos[:, 0] = rand_dist * torch.cos(rand_theta) + self._humanoid_root_states[:, 0]
            goal_pos[:, 1] = rand_dist * torch.sin(rand_theta) + self._humanoid_root_states[:, 1]
            goal_pos[:, 2] = rand_height

        self.update_goal_state(goal_pos)

        return goal_pos


    def capture_new_goal_from_keyborad(self):
        print("\033[1mCurrent root position: {}\033[0m".format(self._root_states[self._humanoid_actor_ids][..., :3].cpu().numpy()))
        text = input("\033[34mEnter the text prompt: \033[0m")
        goal_position_xyz = input("\033[34mEnter goal xyz position separated by a space: \033[0m")
        new_goal_position = self._root_states[self._humanoid_actor_ids][..., :3].clone()
        try:
            goal_x, goal_y, goal_z = map(float, goal_position_xyz.split())
            new_goal_position[..., 0] = goal_x
            new_goal_position[..., 1] = goal_y
            new_goal_position[..., 2] = goal_z
            print("\033[32mSetting text command: {}\033[0m".format(text))
            print("\033[32mSetting the goal position as {}\033[0m".format(new_goal_position.cpu().numpy()))
            self.update_goal_state(new_goal_position)
        except ValueError:
            print("Invalid input. Please enter two numbers separated by a space. The goal position remains unchanged")
            print("\033[32mSetting the goal position as {}\033[0m".format(new_goal_position.cpu().numpy()))
            self.update_goal_state(new_goal_position)
        return text, new_goal_position

    def set_state(self, root_state, dof_state, goal_pos):
        self._root_states[self._humanoid_actor_ids, ...] = torch.from_numpy(root_state[...]).cuda()
        self._root_states[self._humanoid_actor_ids, 7:] = 0
        self._root_states[self._marker_actor_ids, :3] = torch.from_numpy(goal_pos[...]).cuda()
        
        self._dof_state[...] = torch.from_numpy(dof_state[...]).cuda()
        # self._dof_state[..., 1] = 0
        self._goal_pos[:] = torch.from_numpy(goal_pos).cuda()

        return

    def vis_step(self, root_state, dof_state, goal_pos):
        if not self.paused and self.enable_viewer_sync:

            for _ in range(2):
                self.set_state(root_state, dof_state, goal_pos)
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_states))
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self._dof_state))

                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_rigid_body_state_tensor(self.sim)


                if self.device == 'cpu':
                    self.gym.fetch_results(self.sim, True)
            
                self.render()

        root_state_diff = self._root_states[self._humanoid_actor_ids, ...] - torch.from_numpy(root_state[...]).cuda()
        dof_state_diff = self._dof_state - torch.from_numpy(dof_state[...]).cuda()
        print(root_state_diff[..., :7].mean(), dof_state_diff[:, 0].mean())

        return

    def _update_marker(self):

        if self._goal_pos is not None:
            self._marker_pos[...] = self._goal_pos

            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(self._marker_actor_ids),
                len(self._marker_actor_ids),
            )

    def _draw_task(self):
        # uncomment it if you want to see the marker
        self._update_marker()
        return

    def post_physics_step(self):
        super().post_physics_step()
        if not self._collect_joint_positions or self._total_goals_collected >= self._max_goals_to_collect:
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
        goal_pos = self._goal_pos.cpu().numpy()
        if goal_pos.ndim == 3:
            goal_pos = goal_pos[:, 0, :]
        frame_data = {
            "rigid_body_pos": joint_pos,
            "rigid_body_rot": joint_rot,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "root_vel": root_vel,
            "root_ang_vel": root_ang_vel,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "goal_pos": goal_pos,
        }
        self._joint_positions_buffer.append(frame_data)
        return

    def reset(self, env_ids=None):
        # When algorithm calls env_reset() with no args, env_ids is None; saving is done via finalize_episode_trajectory before reset.
        if self._collect_joint_positions and env_ids is not None:
            env_ids_list = env_ids.detach().cpu().tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
            if len(env_ids_list) > 0 and self._total_goals_collected < self._max_goals_to_collect:
                self._save_joint_positions(env_ids)
            for env_id in env_ids_list:
                self._episode_start_time[env_id] = None
                self._goal_status_per_env[env_id] = None
                self._joint_positions_start_idx[env_id] = len(self._joint_positions_buffer)
                self._goal_id_per_env[env_id] = self._goal_id_per_env[env_id] + 1
        super().reset(env_ids=env_ids)
        return

    def _save_joint_positions(self, env_ids):
        if len(self._joint_positions_buffer) == 0:
            print("[WARNING] Joint positions buffer is empty! No trajectory data to save.")
            return
        if isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.detach().cpu().tolist()
        elif isinstance(env_ids, np.ndarray):
            env_ids_list = env_ids.tolist()
        else:
            env_ids_list = list(env_ids)
        if len(env_ids_list) == 0 or self._total_goals_collected >= self._max_goals_to_collect:
            return
        os.makedirs(self._trajectory_save_dir, exist_ok=True)
        timestamp = int(time.time())
        for env_id in env_ids_list:
            if self._total_goals_collected >= self._max_goals_to_collect:
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
            goal_pos_all = []
            for frame_data in self._joint_positions_buffer[start_idx:]:
                rigid_body_pos_all.append(frame_data["rigid_body_pos"][env_id])
                rigid_body_rot_all.append(frame_data["rigid_body_rot"][env_id])
                root_pos_all.append(frame_data["root_pos"][env_id])
                root_rot_all.append(frame_data["root_rot"][env_id])
                root_vel_all.append(frame_data["root_vel"][env_id])
                root_ang_vel_all.append(frame_data["root_ang_vel"][env_id])
                dof_pos_all.append(frame_data["dof_pos"][env_id])
                dof_vel_all.append(frame_data["dof_vel"][env_id])
                goal_pos_all.append(frame_data["goal_pos"][env_id])
            rigid_body_pos_stacked = np.stack(rigid_body_pos_all, axis=0)
            rigid_body_rot_stacked = np.stack(rigid_body_rot_all, axis=0)
            root_pos_stacked = np.stack(root_pos_all, axis=0)
            root_rot_stacked = np.stack(root_rot_all, axis=0)
            root_vel_stacked = np.stack(root_vel_all, axis=0)
            root_ang_vel_stacked = np.stack(root_ang_vel_all, axis=0)
            dof_pos_stacked = np.stack(dof_pos_all, axis=0)
            dof_vel_stacked = np.stack(dof_vel_all, axis=0)
            goal_pos_stacked = np.stack(goal_pos_all, axis=0)
            goal_status = self._goal_status_per_env[env_id] or "unknown"
            goal_id = self._goal_id_per_env[env_id]
            num_frames = rigid_body_pos_stacked.shape[0]
            start_time = self._episode_start_time[env_id]
            elapsed_wall_time = (time.time() - start_time) if start_time is not None else num_frames * self.dt
            fps_measured = num_frames / elapsed_wall_time if elapsed_wall_time > 0 else 0.0
            total_time_sim = num_frames * self.dt
            fps_sim = 1.0 / self.dt if self.dt > 0 else 0.0
            trajectory_data = {
                "rigid_body_pos": rigid_body_pos_stacked,
                "rigid_body_rot": rigid_body_rot_stacked,
                "root_pos": root_pos_stacked,
                "root_rot": root_rot_stacked,
                "root_vel": root_vel_stacked,
                "root_ang_vel": root_ang_vel_stacked,
                "dof_pos": dof_pos_stacked,
                "dof_vel": dof_vel_stacked,
                "goal_pos": goal_pos_stacked,
                "dt": self.dt,
                "num_envs": 1,
                "num_frames": num_frames,
                "total_time": elapsed_wall_time,
                "fps": fps_measured,
                "total_time_sim": total_time_sim,
                "fps_sim": fps_sim,
                "env_id": env_id,
                "goal_id": goal_id,
                "goal_status": goal_status,
                "total_goal_index": self._total_goals_collected,
            }
            self._all_trajectories.append(trajectory_data)
            self._total_goals_collected += 1
            print(f"[DEBUG] Saved trajectory {self._total_goals_collected}/{self._max_goals_to_collect}: env {env_id}, goal_id {goal_id}, status={goal_status}, frames={num_frames}")
            self._joint_positions_start_idx[env_id] = len(self._joint_positions_buffer)
        if self._total_goals_collected >= self._max_goals_to_collect and len(self._all_trajectories) > 0:
            print(f"\n[SAVE] Reached collection limit. Saving {len(self._all_trajectories)} trajectories...")
            self._save_all_trajectories(self._trajectory_save_dir, timestamp)
        min_start = min(self._joint_positions_start_idx)
        if min_start > 0:
            self._joint_positions_buffer = self._joint_positions_buffer[min_start:]
            self._joint_positions_start_idx = [idx - min_start for idx in self._joint_positions_start_idx]
        return

    def _save_all_trajectories(self, save_dir, timestamp):
        if len(self._all_trajectories) == 0:
            return
        save_path = os.path.join(save_dir, f"all_trajectories_{timestamp}.pkl")
        total_trajectories = len(self._all_trajectories)
        total_frames = sum(t["num_frames"] for t in self._all_trajectories)
        total_wall_time = sum(t["total_time"] for t in self._all_trajectories)
        status_counts = {}
        for t in self._all_trajectories:
            s = t["goal_status"]
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
        print(f"\n[SUCCESS] Saved {total_trajectories} trajectories ({total_frames} frames) to {os.path.abspath(save_path)}")
        print(f"         Status breakdown: {status_counts}")
        print(f"         Avg frames/trajectory: {all_data['metadata']['avg_frames_per_trajectory']:.1f}\n")
        self._all_trajectories = []
        return

    def save_remaining_trajectories(self):
        """Save any accumulated trajectories (e.g. at end of run)."""
        if len(self._all_trajectories) == 0:
            print("[DEBUG] No remaining trajectories to save.")
            return
        os.makedirs(self._trajectory_save_dir, exist_ok=True)
        timestamp = int(time.time())
        self._save_all_trajectories(self._trajectory_save_dir, timestamp)

    def draw_task(self):
        self._update_marker()

