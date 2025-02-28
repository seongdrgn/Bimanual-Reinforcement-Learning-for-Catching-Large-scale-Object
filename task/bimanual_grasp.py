import numpy as np
import torch
import os
import math
import imageio
import random
import time
from PIL import Image as im
import threading
import matplotlib.pyplot as plt
import shutil

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

class BimanualGrasp(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.dof_speed_scales = 20.
        self.ur_dof_scales = 2*np.pi

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.ur5e_dof_noise = self.cfg["env"]["ur5eDofNoise"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.allegro_speed_scale = self.cfg["env"]["allegroSpeedScale"]

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # obs : eef_pose (7) + eef_vel(3) + joint position (6) + allegro position (16)
        self.cfg["env"]["numObservations"] = 116 #(7 + 3 + 6 + 16)*2 + 10 + 16*2 + 3
        # actions include : UR5e joint position + Allegro joint position
        self.cfg["env"]["numActions"] = (6 + 16)*2

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed

        self.num_episode = 0
        self.num_success = 0
        self.ep_num = 0

        # Tensor placeholders
        self._root_state = None                 # State of root body        (n_envs, 13)
        self._dof_state = None                  # State of all joints       (n_envs, n_dof)
        self._q = None                          # Joint positions           (n_envs, n_dof)
        self._qd = None                         # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None           # State of all rigid bodies (n_envs, n_bodies, 13)
        self._contact_forces = None             # Contact forces in sim
        self._eef_state = None                  # end effector state (at grasping point)
        self._eef_lf_state = None               # end effector state (at left fingertip)
        self._eef_rf_state = None               # end effector state (at left fingertip)
        self._j_eef = None                      # Jacobian for end effector
        self._mm = None                         # Mass matrix
        self._arm_control = None                # Tensor buffer for controlling arm
        self._allegro_control = None            # Tensor buffer for controlling allegro
        self._pos_control = None                # Position actions
        self._effort_control = None             # Torque actions
        self._global_indices = None             # Unique indices corresponding to all envs in flattened array

        self.ur5e_base_height = 0.81 + 0.015 # table height : 0.81+0.015

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2
        self.dt = 1/60.

        torch.manual_seed(826)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(-0.8, 0.0, 2.5)
            cam_target = gymapi.Vec3(1.5, 0.0, 0.0)
            # cam_pos = gymapi.Vec3(1.5, 2.5, 2.5)
            # cam_target = gymapi.Vec3(1.5, -1.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Set Default Dof Pose
        """
            0,1,2,3,4,5 -> ur5e joint
        """
        ur5e_default_dof_pos_0 = [0.0*math.pi, -(2/5)*math.pi, -(2/6)*math.pi, -(4/8)*math.pi, 0.5*math.pi, 1.0*math.pi,]
        ur5e_default_dof_pos_1 = [0.0*math.pi, -(2/5)*math.pi, -(2/6)*math.pi, -(4/8)*math.pi, 0.5*math.pi, 1.0*math.pi,]
        allegro_default_dof_pos_0 = [0.0]*16
        allegro_default_dof_pos_1 = [0.0]*16
        self.ur5e_allegro_default_dof_pos = to_torch(
            ur5e_default_dof_pos_0 + allegro_default_dof_pos_0 + ur5e_default_dof_pos_1 + allegro_default_dof_pos_1, device=self.device)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # Refresh tensors
        self._refresh()

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.
        """

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)

        # Define closest distance of manipulator 0
        self.d_closest_thumb_0 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.d_closest_index_0 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.d_closest_middle_0 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.d_closest_ring_0 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.d_closest_target_0 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.d_closest_align_0 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.d_closest_palm_0 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)

        # Define closest distance of manipulator 1
        self.d_closest_thumb_1 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.d_closest_index_1 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.d_closest_middle_1 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.d_closest_ring_1 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.d_closest_target_1 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.d_closest_align_1 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.d_closest_palm_1 = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)

        self.num_trans = torch.zeros(
            self.num_envs, device=self.device, dtype = torch.float)
        self.is_located = torch.zeros(
            self.num_envs, device=self.device, dtype = torch.float)
        self.ep_num = torch.zeros(
            self.num_envs, device=self.device, dtype = torch.float)
        self.extras = {}

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim_params.use_gpu_pipeline = True
        self.sim_params.dt = 1/60.
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) # z-up
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        camera_height = 1.0

        # Camera Position for external view
        ext_camera_pos = gymapi.Transform()
        camera_height = 1.2
        ext_camera_pos.p = gymapi.Vec3(0.0, 0.0, 0.81+0.015 + camera_height)
        ext_camera_pos.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0,1.0,0.0), np.radians(30.0))

        ext_camera_props = gymapi.CameraProperties()
        ext_camera_props.width = 128
        ext_camera_props.height = 128
        ext_camera_props.enable_tensors = True
        ext_camera_props.horizontal_fov = 86.0

        # Table Asset Configuration
        table_pos = [0.0, 0.0, 0.81 + 0.015/2] # Table h : 0.81 + 0.015
        table_thickness = 0.015
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        # table_asset = self.gym.create_box(self.sim, *[1.38, 1.02, table_thickness], table_opts)
        table_asset = self.gym.create_box(self.sim, *[1.02, 1.38, table_thickness], table_opts)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(*table_pos)
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Target Position & Orientation Visualize
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["targetAssetFileName"]
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        target_point_opts = gymapi.AssetOptions()
        target_point_opts.disable_gravity = True
        target_point_asset = self.gym.load_asset(self.sim, asset_root, asset_file, target_point_opts)
        target_point_pose = gymapi.Transform()
        target_point_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        target_point_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        self._target_point_base_pos = np.array([0.0, 0.0, (0.81 + table_thickness / 2)])

        # Wall Configuration
        wall_height = 5.0
        wall_width = 5.0
        wall_thickness = 0.1
        wall_center_pos = [4.0, 0.0, wall_height/2]
        wall_left_pos = [1.5, 2.5, wall_height/2]
        wall_right_pos = [1.5, -2.5, wall_height/2]
        wall_opts = gymapi.AssetOptions()
        wall_opts.fix_base_link = True
        wall_center_asset = self.gym.create_box(self.sim, *[wall_thickness, wall_width, wall_height], wall_opts)
        wall_left_asset = self.gym.create_box(self.sim, *[wall_width, wall_thickness, wall_height], wall_opts)
        wall_right_asset = self.gym.create_box(self.sim, *[wall_width, wall_thickness, wall_height], wall_opts)
        wall_center_pose = gymapi.Transform()
        wall_center_pose.p = gymapi.Vec3(*wall_center_pos)
        wall_left_pose = gymapi.Transform()
        wall_left_pose.p = gymapi.Vec3(*wall_left_pos)
        wall_right_pose = gymapi.Transform()
        wall_right_pose.p = gymapi.Vec3(*wall_right_pos)

        cube_asset_file = self.cfg["env"]["asset"]["cubeAssetFileName"]
        cylinder_asset_file = self.cfg["env"]["asset"]["cylinderAssetFileName"]
        sphere_asset_file = self.cfg["env"]["asset"]["sphereAssetFileName"]
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        meat_can_asset_file = self.cfg["env"]["asset"]["meatcanAssetFileName"]
        brick_asset_file = self.cfg["env"]["asset"]["brickAssetFileName"]
        multicolor_cube_asset_file = self.cfg["env"]["asset"]["multicolorcubeAssetFileName"]

        cube_asset_path = os.path.join(asset_root, cube_asset_file)
        cube_asset_root = os.path.dirname(cube_asset_path)
        cube_asset_file = os.path.basename(cube_asset_path)
        cube_opts = gymapi.AssetOptions()
        # cube_opts.disable_gravity = True
        cube_asset = self.gym.load_asset(self.sim, cube_asset_root, cube_asset_file, cube_opts)
        cube_pose = gymapi.Transform()
        cube_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        cube_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        multicolor_cube_asset_path = os.path.join(asset_root, multicolor_cube_asset_file)
        multicolor_cube_asset_root = os.path.dirname(multicolor_cube_asset_path)
        multicolor_cube_asset_file = os.path.basename(multicolor_cube_asset_path)
        multicolor_cube_opts = gymapi.AssetOptions()
        multicolor_cube_asset = self.gym.load_asset(self.sim, multicolor_cube_asset_root, multicolor_cube_asset_file, multicolor_cube_opts)
        multicolor_cube_pose = gymapi.Transform()
        multicolor_cube_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        multicolor_cube_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        cylinder_asset_path = os.path.join(asset_root, cylinder_asset_file)
        cylinder_asset_root = os.path.dirname(cylinder_asset_path)
        cylinder_asset_file = os.path.basename(cylinder_asset_path)
        cylinder_opts = gymapi.AssetOptions()
        # cylinder_opts.disable_gravity = True
        cylinder_asset = self.gym.load_asset(self.sim, cylinder_asset_root, cylinder_asset_file, cylinder_opts)
        cylinder_pose = gymapi.Transform()
        cylinder_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        cylinder_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        sphere_asset_path = os.path.join(asset_root, sphere_asset_file)
        sphere_asset_root = os.path.dirname(sphere_asset_path)
        sphere_asset_file = os.path.basename(sphere_asset_path)
        sphere_opts = gymapi.AssetOptions()
        # sphere_opts.disable_gravity = True
        sphere_asset = self.gym.load_asset(self.sim, sphere_asset_root, sphere_asset_file, sphere_opts)
        sphere_pose = gymapi.Transform()
        sphere_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        sphere_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        meat_can_asset_path = os.path.join(asset_root, meat_can_asset_file)
        meat_can_asset_root = os.path.dirname(meat_can_asset_path)
        meat_can_asset_file = os.path.basename(meat_can_asset_path)
        meat_can_opts = gymapi.AssetOptions()
        meat_can_asset = self.gym.load_asset(self.sim, meat_can_asset_root, meat_can_asset_file, meat_can_opts)
        meat_can_pose = gymapi.Transform()
        meat_can_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        meat_can_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        brick_asset_path = os.path.join(asset_root, brick_asset_file)
        brick_asset_root = os.path.dirname(brick_asset_path)
        brick_asset_file = os.path.basename(brick_asset_path)
        brick_opts = gymapi.AssetOptions()
        brick_asset = self.gym.load_asset(self.sim, brick_asset_root, brick_asset_file, brick_opts)
        brick_pose = gymapi.Transform()
        brick_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        brick_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # ur5e Asset Configuration
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file = self.cfg["env"]["asset"]["ur5eAssetFileName"]
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        ur5e_opts = gymapi.AssetOptions()
        ur5e_opts.fix_base_link = True
        ur5e_opts.flip_visual_attachments = False
        ur5e_opts.disable_gravity = True
        ur5e_opts.collapse_fixed_joints = False
        ur5e_opts.thickness = 0.001
        ur5e_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS

        ur5e_asset_0 = self.gym.load_asset(self.sim, asset_root, asset_file, ur5e_opts)
        ur5e_asset_1 = self.gym.load_asset(self.sim, asset_root, asset_file, ur5e_opts)
        self.num_ur5e_bodies_0 = self.gym.get_asset_rigid_body_count(ur5e_asset_0)
        self.num_ur5e_dofs_0 = self.gym.get_asset_dof_count(ur5e_asset_0)
        self.num_ur5e_bodies_1 = self.gym.get_asset_rigid_body_count(ur5e_asset_1)
        self.num_ur5e_dofs_1 = self.gym.get_asset_dof_count(ur5e_asset_1)

        allegro_dof_stiffness = [3]*16
        allegro_dof_damping = [0.1]*16

        ur5e_dof_stiffness = [400]*6
        ur5e_dof_damping = [80]*6

        ur5e_allegro_dof_stiffness = to_torch(ur5e_dof_stiffness + allegro_dof_stiffness, dtype=torch.float, device=self.device)
        ur5e_allegro_dof_damping = to_torch(ur5e_dof_damping + allegro_dof_damping, dtype=torch.float, device=self.device)

        print("num UR5e with Allegro Hand bodies: ", self.num_ur5e_bodies_0 + self.num_ur5e_bodies_1)
        print("num UR5e with Allegro Hand dofs: ", self.num_ur5e_dofs_0 + self.num_ur5e_dofs_1)

        # Configure DOF properties
        """
            Drive Mode : EFFORT, POS, VEL
                EFFORT : If the DOF is linear, the effort is a force in Newtons.
                            If the DOF is angular, the effort is a torque in Nm.
        """
        ur5e_allegro_dof_props = self.gym.get_asset_dof_properties(ur5e_asset_0)
        self._ur5e_allegro_dof_lower_limits = []
        self._ur5e_allegro_dof_upper_limits = []
        self._ur5e_allegro_effort_limits = []
        for i in range(self.num_ur5e_dofs_0):
            ur5e_allegro_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            ur5e_allegro_dof_props['stiffness'][i] = ur5e_allegro_dof_stiffness[i]
            ur5e_allegro_dof_props['damping'][i] = ur5e_allegro_dof_damping[i]
            self._ur5e_allegro_dof_lower_limits.append(ur5e_allegro_dof_props["lower"][i])
            self._ur5e_allegro_dof_upper_limits.append(ur5e_allegro_dof_props["upper"][i])
            self._ur5e_allegro_effort_limits.append(ur5e_allegro_dof_props["effort"][i])
            if i > 6:
                ur5e_allegro_dof_props['friction'][i] = 0.01
                ur5e_allegro_dof_props['armature'][i] = 0.001

        self._ur5e_allegro_dof_lower_limits = to_torch(self._ur5e_allegro_dof_lower_limits + self._ur5e_allegro_dof_lower_limits, device=self.device)
        self._ur5e_allegro_dof_upper_limits = to_torch(self._ur5e_allegro_dof_upper_limits + self._ur5e_allegro_dof_upper_limits, device=self.device)
        self._ur5e_allegro_effort_limits = to_torch(self._ur5e_allegro_effort_limits + self._ur5e_allegro_effort_limits, device=self.device)
        self.ur5e_dof_speed_scales = torch.ones_like(self._ur5e_allegro_dof_lower_limits + self._ur5e_allegro_dof_lower_limits)

        # Set Start ur5e Pose
        ur5e_start_pose = gymapi.Transform()
        ur5e_start_pose.p = gymapi.Vec3(0.0, -0.6, self.ur5e_base_height)
        ur5e_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        ur5e_start_pose_1 = gymapi.Transform()
        ur5e_start_pose_1.p = gymapi.Vec3(0.0, 0.6, self.ur5e_base_height)
        ur5e_start_pose_1.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        # compute aggregate size
        self.num_ur5e_allegro_bodies = self.gym.get_asset_rigid_body_count(ur5e_asset_0)
        self.num_ur5e_allegro_shapes = self.gym.get_asset_rigid_shape_count(ur5e_asset_0)
        max_agg_bodies = self.num_ur5e_allegro_bodies*2 + 10
        max_agg_shapes = self.num_ur5e_allegro_shapes*2 + 10

        # Define Parameters in Every Environments
        self.ur5es = []
        self.envs = []
        self.objects = []
        self.ext_cam_tensors = []

        # Create "num_envs" environments
        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)
            self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.5,0.5,0.5), gymapi.Vec3(0.5,0.5,0.5), gymapi.Vec3(-0,0,1.0))
            self.gym.set_light_parameters(self.sim, 1, gymapi.Vec3(0.5,0.5,0.5), gymapi.Vec3(0.5,0.5,0.5), gymapi.Vec3(-0,0,1.0))
            self.gym.set_light_parameters(self.sim, 2, gymapi.Vec3(0.5,0.5,0.5), gymapi.Vec3(0.5,0.5,0.5), gymapi.Vec3(-0,0,1.0))

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env, max_agg_bodies, max_agg_shapes, True)

            # Add ur5e
            # NOTE: ur5e should ALWAYS be loaded first in sim!
            ur5e_handle_0 = self.gym.create_actor(env, ur5e_asset_0, ur5e_start_pose, "ur5e_0", i, 0, 0)
            ur5e_handle_1 = self.gym.create_actor(env, ur5e_asset_1, ur5e_start_pose_1, "ur5e_1", i, 0, 0)

            self.gym.set_actor_dof_properties(env, ur5e_handle_0, ur5e_allegro_dof_props)
            self.gym.set_actor_dof_properties(env, ur5e_handle_1, ur5e_allegro_dof_props)

            # Add table
            table_handle = self.gym.create_actor(env, table_asset, table_pose, "table", i, 1, 0)

            # Add chroma
            wall_center_handle = self.gym.create_actor(env, wall_center_asset, wall_center_pose, "wall_center", i+2, 0, 0)
            wall_left_handle = self.gym.create_actor(env, wall_left_asset, wall_left_pose, "wall_left", i+2, 0, 0)
            wall_right_handle = self.gym.create_actor(env, wall_right_asset, wall_right_pose, "wall_right", i+2, 0, 0)
            self.gym.set_rigid_body_color(env, wall_center_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(32/255,55/255,201/255))
            self.gym.set_rigid_body_color(env, wall_left_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(32/255,55/255,201/255))
            self.gym.set_rigid_body_color(env, wall_right_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(32/255,55/255,201/255))

            # Add target point
            self._target_point_id = self.gym.create_actor(env, target_point_asset, target_point_pose, "target", i+1, -1, 1)

            # Add objects
            if i % 3 == 0:
                self.obj_id = self.gym.create_actor(env, cube_asset, cube_pose, "object", i, 5, 0)
            elif i % 3 == 1:
                self.obj_id = self.gym.create_actor(env, cylinder_asset, cylinder_pose, "object", i, 5, 0)
            else :
                self.obj_id = self.gym.create_actor(env, sphere_asset, sphere_pose, "object", i, 5, 0)

            # self.gym.set_rigid_body_segmentation_id(env, self.obj_id, 0, 123123)

            # Add external camera
            # self._ext_camera_handle = self.gym.create_camera_sensor(env, ext_camera_props)
            # # self.ext_cameras.append(self._ext_camera_handle)
            # self.gym.set_camera_transform(self._ext_camera_handle, env, ext_camera_pos)
            # ext_cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, self._ext_camera_handle, gymapi.IMAGE_COLOR)
            # torch_ext_cam_tensor = gymtorch.wrap_tensor(ext_cam_tensor)
            # self.ext_cam_tensors.append(torch_ext_cam_tensor[...,:3])

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env)

            # Store the created env pointers
        
        # Define initial cylinder state
        self._init_object_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Define target point state
        self._init_target_point_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        ur5e_handle_0 = 0
        ur5e_handle_1 = 1

        self.handles = {
            # ur5e_0
            "palm_link_0": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "palm_link"),
            "thumb_tip_0": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_thumb_tip"),
            "index_tip_0": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_index_tip"),
            "middle_tip_0": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_middle_tip"),
            "ring_tip_0": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_ring_tip"),
            "ee_link_0": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "ee_link"),

#             "index_fsr_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_index_fsr_1"),
#             "index_fsr_2": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_index_fsr_2"),
#             "index_fsr_3": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_index_fsr_tip"),

#             "middle_fsr_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_middle_fsr_1"),
#             "middle_fsr_2": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_middle_fsr_2"),
#             "middle_fsr_3": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_middle_fsr_tip"),

#             "ring_fsr_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_ring_fsr_1"),
#             "ring_fsr_2": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_ring_fsr_2"),
#             "ring_fsr_3": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_ring_fsr_tip"),

#             "thumb_fsr_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_thumb_fsr_1"),
#             "thumb_fsr_2": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_thumb_fsr_2"),
#             "thumb_fsr_3": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_thumb_fsr_tip"),

#             "palm_fsr_0": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_palm_fsr_0"),
#             "palm_fsr_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_palm_fsr_1"),
#             "palm_fsr_2": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_palm_fsr_2"),
#             "palm_fsr_3": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_0, "link_palm_fsr_3"),
# ###########################
#             "1_index_fsr_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_index_fsr_1"),
#             "1_index_fsr_2": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_index_fsr_2"),
#             "1_index_fsr_3": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_index_fsr_tip"),

#             "1_middle_fsr_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_middle_fsr_1"),
#             "1_middle_fsr_2": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_middle_fsr_2"),
#             "1_middle_fsr_3": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_middle_fsr_tip"),

#             "1_ring_fsr_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_ring_fsr_1"),
#             "1_ring_fsr_2": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_ring_fsr_2"),
#             "1_ring_fsr_3": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_ring_fsr_tip"),

#             "1_thumb_fsr_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_thumb_fsr_1"),
#             "1_thumb_fsr_2": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_thumb_fsr_2"),
#             "1_thumb_fsr_3": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_thumb_fsr_tip"),

#             "1_palm_fsr_0": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_palm_fsr_0"),
#             "1_palm_fsr_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_palm_fsr_1"),
#             "1_palm_fsr_2": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_palm_fsr_2"),
#             "1_palm_fsr_3": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_palm_fsr_3"),

            # ur5e_1
            "palm_link_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "palm_link"),
            "thumb_tip_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_thumb_tip"),
            "index_tip_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_index_tip"),
            "middle_tip_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_middle_tip"),
            "ring_tip_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "link_ring_tip"),
            "ee_link_1": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle_1, "ee_link"),

            # object
            "object_body_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self.obj_id, "object"),

            # Target
            "target": self.gym.find_actor_rigid_body_handle(env_ptr, self._target_point_id, "target"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        """
            Actor root states buffer
            The buffer has shape (num_actors, 13).
            State for each actor root contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        
            Dof states buffer
            Each DOF state contains position and velocity.
        """
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)

        _contact_forces_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self._contact_forces_state = gymtorch.wrap_tensor(_contact_forces_tensor).view(self.num_envs, -1, 3)

        # Initialize joint state
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]

        # Initialize end-effector state
        self._palm_state_0 = self._rigid_body_state[:, self.handles["palm_link_0"], :]
        self._thumb_state_0 = self._rigid_body_state[:, self.handles["thumb_tip_0"],:]
        self._index_state_0 = self._rigid_body_state[:, self.handles["index_tip_0"],:]
        self._middle_state_0 = self._rigid_body_state[:, self.handles["middle_tip_0"],:]
        self._ring_state_0 = self._rigid_body_state[:, self.handles["ring_tip_0"],:]
        self._eef_state_0 = self._rigid_body_state[:, self.handles["ee_link_0"], :]

        self._palm_state_1 = self._rigid_body_state[:, self.handles["palm_link_1"], :]
        self._thumb_state_1 = self._rigid_body_state[:, self.handles["thumb_tip_1"],:]
        self._index_state_1 = self._rigid_body_state[:, self.handles["index_tip_1"],:]
        self._middle_state_1 = self._rigid_body_state[:, self.handles["middle_tip_1"],:]
        self._ring_state_1 = self._rigid_body_state[:, self.handles["ring_tip_1"],:]
        self._eef_state_1 = self._rigid_body_state[:, self.handles["ee_link_1"], :]

        # Initialize cylinder state
        self._object_state = self._root_state[:, self.obj_id, :]

        # Initialize target cylinder state
        self._target_point_state = self._root_state[:, self._target_point_id, :]

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 8, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        roll_0,pitch_0,yaw_0 = get_euler_xyz(self._rigid_body_state[:,self.handles["palm_link_0"], 3:7])
        self._palm_quat_0 = quat_from_euler_xyz(roll_0,pitch_0,yaw_0)
        eef_roll_0, eef_pitch_0, eef_yaw_0 = get_euler_xyz(self._rigid_body_state[:,self.handles["ee_link_0"], 3:7])
        self._eef_quat_0 = quat_from_euler_xyz(eef_roll_0,eef_pitch_0,eef_yaw_0)

        roll_1,pitch_1,yaw_1 = get_euler_xyz(self._rigid_body_state[:,self.handles["palm_link_1"], 3:7])
        self._palm_quat_1 = quat_from_euler_xyz(roll_1,pitch_1,yaw_1)
        eef_roll_1, eef_pitch_1, eef_yaw_1 = get_euler_xyz(self._rigid_body_state[:,self.handles["ee_link_1"], 3:7])
        self._eef_quat_1 = quat_from_euler_xyz(eef_roll_1,eef_pitch_1,eef_yaw_1)

        t_roll,t_pitch,t_yaw = get_euler_xyz(self._rigid_body_state[:,self.handles["target"], 3:7])
        target_quat = quat_from_euler_xyz(t_roll,t_pitch,t_yaw)

        contact_force_0 = torch.cat((self._contact_forces_state[:,14:17,:], self._contact_forces_state[:,22:25,:],self._contact_forces_state[:,34:37,:],self._contact_forces_state[:,44:45,:],self._contact_forces_state[:,41:42,:],self._contact_forces_state[:,43:44,:],self._contact_forces_state[:,25:29,:]),dim=1)
        mean_contact_force_0 = torch.mean(contact_force_0,dim=2)
        binary_contact_info_0 = torch.where(torch.abs(mean_contact_force_0) > 0.3, torch.ones_like(mean_contact_force_0), torch.zeros_like(mean_contact_force_0))

        contact_force_1 = torch.cat((self._contact_forces_state[:,61:64,:], self._contact_forces_state[:,69:72,:],self._contact_forces_state[:,81:84,:],self._contact_forces_state[:,90:91,:],self._contact_forces_state[:,88:89,:],self._contact_forces_state[:,91:92,:],self._contact_forces_state[:,72:76,:]),dim=1)
        mean_contact_force_1 = torch.mean(contact_force_1,dim=2)
        binary_contact_info_1 = torch.where(torch.abs(mean_contact_force_1) > 0.3, torch.ones_like(mean_contact_force_1), torch.zeros_like(mean_contact_force_1))

        self.states.update({
            # ur5e_0
            "palm_pos_0": self._palm_state_0[:,:3],
            "palm_rot_0": self._palm_state_0[:,3:7],
            # "palm_rot_0": self._palm_quat_0,
            "palm_vel_0": self._palm_state_0[:,7:10],
            "q_ur5e_0": self._q[:,:6],

            "ur5e_eef_pos_0": self._eef_state_0[:,:3],
            "ur5e_eef_rot_0": self._eef_quat_0,

            # allegro hand_0
            "q_allegro_0" : self._q[:,6:22],

            # ur5e_1
            "palm_pos_1": self._palm_state_1[:,:3],
            # "palm_rot_1": self._palm_quat_1,
            "palm_rot_1": self._palm_state_1[:,3:7],
            "palm_vel_1": self._palm_state_1[:,7:10],
            "q_ur5e_1": self._q[:,22:28],

            "ur5e_eef_pos_1": self._eef_state_1[:,:3],
            "ur5e_eef_rot_1": self._eef_quat_1,

            # allegro hand_1
            "q_allegro_1" : self._q[:,28:],

            # object
            "object_pos": self._object_state[:,:3],
            "object_quat": self._object_state[:,3:7],
            "object_vel": self._object_state[:,7:10],

            # reward
            "obj_palm_align_0": self._object_state[:,1:2] - self._palm_state_0[:,1:2],
            "obj_palm_dist_0": self._object_state[:,:3] - self._palm_state_0[:,:3],
            "target_palm_dist_0": self._target_point_state[:,:3] - self._palm_state_0[:,:3],

            "obj_palm_align_1": self._object_state[:,1:2] - self._palm_state_1[:,1:2],
            "obj_palm_dist_1": self._object_state[:,:3] - self._palm_state_1[:,:3],
            "target_palm_dist_1": self._target_point_state[:,:3] - self._palm_state_1[:,:3],

            # distance tip from obejct
            "obj_thumb_dist_0": self._object_state[:,:3] - self._thumb_state_0[:,:3],
            "obj_index_dist_0": self._object_state[:,:3] - self._index_state_0[:,:3],
            "obj_middle_dist_0": self._object_state[:,:3] - self._middle_state_0[:,:3],
            "obj_ring_dist_0": self._object_state[:,:3] - self._ring_state_0[:,:3],

            "obj_thumb_dist_1": self._object_state[:,:3] - self._thumb_state_1[:,:3],
            "obj_index_dist_1": self._object_state[:,:3] - self._index_state_1[:,:3],
            "obj_middle_dist_1": self._object_state[:,:3] - self._middle_state_1[:,:3],
            "obj_ring_dist_1": self._object_state[:,:3] - self._ring_state_1[:,:3],

            # target
            "target_pose": self._target_point_state[:,:7],
            "target_pos": self._target_point_state[:,:3],

            # contact force
            "binary_contact_info_0": binary_contact_info_0,
            "binary_contact_info_1": binary_contact_info_1,

        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions, d_closest_thumb_0, d_closest_index_0, d_closest_middle_0, d_closest_ring_0, d_closest_target_0, d_closest_align_0, d_closest_palm_0,
                                        d_closest_thumb_1, d_closest_index_1, d_closest_middle_1, d_closest_ring_1, d_closest_target_1, d_closest_align_1, d_closest_palm_1):
        self.rew_buf[:], self.reset_buf[:], self.d_closest_thumb_0, self.d_closest_index_0, self.d_closest_middle_0, self.d_closest_ring_0, self.d_closest_target_0, self.d_closest_align_0, self.d_closest_palm_0, self.d_closest_thumb_1, self.d_closest_index_1, self.d_closest_middle_1, self.d_closest_ring_1, self.d_closest_target_1, self.d_closest_align_1, self.d_closest_palm_1 = compute_ur5e_reward(
                            self.reset_buf, self.progress_buf, actions, self.states, self.max_episode_length, d_closest_thumb_0, d_closest_index_0, d_closest_middle_0, d_closest_ring_0, d_closest_target_0, d_closest_align_0, d_closest_palm_0,
                                                                                                                d_closest_thumb_1, d_closest_index_1, d_closest_middle_1, d_closest_ring_1, d_closest_target_1, d_closest_align_1, d_closest_palm_1)

    def crop_center_numpy(self, image, crop_width, crop_height):
        batch_size, channels, img_height, img_width = image.shape
        crop_left = (img_width - crop_width) // 2
        crop_top = (img_height - crop_height) // 2
        
        cropped_images = image[:, :, crop_top:crop_top + crop_height, crop_left:crop_left + crop_width]
        return cropped_images

    def compute_observations(self):
        self._refresh()

        # obs = ["palm_pos_0","palm_rot_0","palm_vel_0","q_ur5e_0","q_allegro_0",
        #        "palm_pos_1","palm_rot_1","palm_vel_1","q_ur5e_1","q_allegro_1",
        #        "object_pos","object_quat","object_vel","binary_contact_info_0","binary_contact_info_1","target_pos"]

        # self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        # maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        self.obs_buf[:,:3] = self.states["palm_pos_0"]
        self.obs_buf[:,3:7] = self.states["palm_rot_0"]
        self.obs_buf[:,7:10] = self.states["palm_vel_0"]
        self.obs_buf[:,10:16] = self.states["q_ur5e_0"]
        self.obs_buf[:,16:32] = self.states["q_allegro_0"]
        self.obs_buf[:,32:48] = self.states["binary_contact_info_0"]

        self.obs_buf[:,48:51] = self.states["palm_pos_1"]
        self.obs_buf[:,51:55] = self.states["palm_rot_1"]
        self.obs_buf[:,55:58] = self.states["palm_vel_1"]
        self.obs_buf[:,58:64] = self.states["q_ur5e_1"]
        self.obs_buf[:,64:80] = self.states["q_allegro_1"]
        self.obs_buf[:,80:96] = self.states["binary_contact_info_1"]

        self.obs_buf[:,96:99] = self.states["object_pos"]
        self.obs_buf[:,99:103] = self.states["object_quat"]
        self.obs_buf[:,103:106] = self.states["object_vel"]

        self.obs_buf[:,106] = torch.norm(self.states["obj_palm_dist_0"], dim=-1)
        self.obs_buf[:,107] = torch.norm(self.states["obj_thumb_dist_0"], dim=-1)
        self.obs_buf[:,108] = torch.norm(self.states["obj_index_dist_0"], dim=-1)
        self.obs_buf[:,109] = torch.norm(self.states["obj_middle_dist_0"], dim=-1)
        self.obs_buf[:,110] = torch.norm(self.states["obj_ring_dist_0"], dim=-1)

        self.obs_buf[:,111] = torch.norm(self.states["obj_palm_dist_1"], dim=-1)
        self.obs_buf[:,112] = torch.norm(self.states["obj_thumb_dist_1"], dim=-1)
        self.obs_buf[:,113] = torch.norm(self.states["obj_index_dist_1"], dim=-1)
        self.obs_buf[:,114] = torch.norm(self.states["obj_middle_dist_1"], dim=-1)
        self.obs_buf[:,115] = torch.norm(self.states["obj_ring_dist_1"], dim=-1)

        # ext_img = torch.stack(self.ext_cam_tensors, dim=0)
        # self.obs_dict["img"] = ext_img/255.0

        return self.obs_dict

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset cylinder
        self._reset_init_object_state(env_ids=env_ids)
        self._object_state[env_ids] = self._init_object_state[env_ids]

        self._reset_init_target_point_state(env_ids=env_ids)
        self._target_point_state[env_ids] = self._init_target_point_state[env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 22*2), device=self.device)
        pos = tensor_clamp(
            self.ur5e_allegro_default_dof_pos.unsqueeze(0) +
            self.ur5e_dof_noise * 1.5 * (reset_noise),
            self._ur5e_allegro_dof_lower_limits.unsqueeze(0), self._ur5e_allegro_dof_upper_limits.unsqueeze(0)
        )

        # Reset the internal obs
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ur5e_ids = self._global_indices[env_ids, :2].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ur5e_ids),
                                                        len(multi_env_ur5e_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ur5e_ids),
                                              len(multi_env_ur5e_ids))

        # Update cylinder states
        multi_env_cylinder_ids = self._global_indices[env_ids, -2:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_cylinder_ids), len(multi_env_cylinder_ids))

        self.ep_num[env_ids] += 1

        # Print success rate
        # if self.is_located[0] == True :
        #     self.num_success += 1

        # if (env_ids[0] == 0) & (self.ep_num > 0):
        #     print("[Success Ep / Total Ep] : %d / %d, [Success Rate(%%)] : %.2f %%"%(self.num_success, self.ep_num, (self.num_success/self.ep_num)*100))

        # if env_ids[0] == 0:
        #     self.ep_num+=1

        self.gym.simulate(self.sim)
        self.progress_buf[env_ids] = 0
        self.num_trans[env_ids] = 0
        # self.reset_buf[env_ids] = 0

        self._refresh()
        self.d_closest_thumb_0[env_ids] = torch.norm(self.states["obj_thumb_dist_0"][env_ids], dim=-1)
        self.d_closest_index_0[env_ids] = torch.norm(self.states["obj_index_dist_0"][env_ids], dim=-1)
        self.d_closest_middle_0[env_ids] = torch.norm(self.states["obj_middle_dist_0"][env_ids], dim=-1)
        self.d_closest_ring_0[env_ids] = torch.norm(self.states["obj_ring_dist_0"][env_ids], dim=-1)
        self.d_closest_target_0[env_ids] = torch.norm(self.states["target_palm_dist_0"][env_ids], dim=-1)
        self.d_closest_align_0[env_ids] = torch.norm(self.states["obj_palm_align_0"][env_ids], dim=-1)
        self.d_closest_palm_0[env_ids] = torch.norm(self.states["obj_palm_dist_0"][env_ids], dim=-1)

        self.d_closest_thumb_1[env_ids] = torch.norm(self.states["obj_thumb_dist_1"][env_ids], dim=-1)
        self.d_closest_index_1[env_ids] = torch.norm(self.states["obj_index_dist_1"][env_ids], dim=-1)
        self.d_closest_middle_1[env_ids] = torch.norm(self.states["obj_middle_dist_1"][env_ids], dim=-1)
        self.d_closest_ring_1[env_ids] = torch.norm(self.states["obj_ring_dist_1"][env_ids], dim=-1)
        self.d_closest_target_1[env_ids] = torch.norm(self.states["target_palm_dist_1"][env_ids], dim=-1)
        self.d_closest_align_1[env_ids] = torch.norm(self.states["obj_palm_align_1"][env_ids], dim=-1)
        self.d_closest_palm_1[env_ids] = torch.norm(self.states["obj_palm_dist_1"][env_ids], dim=-1)

    def reset(self):
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # self.obs_dict["img"] = torch.stack(self.ext_cam_tensors, dim=0)/255.0

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

    def get_random_quat(self, env_ids):
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L261

        uvw = torch_rand_float(0, 1.0, (len(env_ids), 3), device=self.device)
        q_w = torch.sqrt(1.0 - uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 1]))
        q_x = torch.sqrt(1.0 - uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 1]))
        q_y = torch.sqrt(uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 2]))
        q_z = torch.sqrt(uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 2]))
        new_rot = torch.cat((q_x.unsqueeze(-1), q_y.unsqueeze(-1), q_z.unsqueeze(-1), q_w.unsqueeze(-1)), dim=-1)

        return new_rot

    def _reset_init_target_point_state(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_target_point_state = torch.zeros(num_resets, 13, device=self.device)

        this_target_point_state_all = self._init_target_point_state

        # Sampling "centered" around middle of table
        base_cylinder_xyz_state = torch.tensor(self._target_point_base_pos[:3], device=self.device, dtype=torch.float32)

        target_pose = torch.rand(num_resets, 7, device=self.device)
        target_pose_x = torch.rand_like(target_pose[:,0])*0.1 + 0.5 # [0.5, 0.6]
        target_pose_y = torch.rand_like(target_pose_x)*0.6 - 0.3 # [-0.3, 0.3]
        target_pose_z = torch.rand_like(target_pose_x)*0.1 + 0.6 # [0.6, 0.7]

        target_pose[:,0] = target_pose_x
        target_pose[:,1] = target_pose_y
        target_pose[:,2] = target_pose_z

        random_rot = self.get_random_quat(env_ids)
        sampled_target_point_state[:,3:7] = random_rot[:,:]

        sampled_target_point_state[:,:3] = base_cylinder_xyz_state.unsqueeze(0) + target_pose[:,:3]
        this_target_point_state_all[env_ids, :] = sampled_target_point_state

    def get_object_random_pose(self, num_resets):
        x = torch.rand_like(self._root_state[:num_resets,-1,0])*0.5 # [0, 0.5]
        x = 2.5 + x # [2.5, 3.0]

        y = torch.rand_like(x)*0.5 - 0.25 # [-0.25,0.25]

        z = torch.rand_like(x)*0.1 # [0, 0.1]
        z = z + 0.9 # [0.9, 1.0]

        vx = torch.rand_like(x)*0.2 - 0.1 # [-0.1,0.1]
        vx = vx - 3.0 # [-3.1,-2.9]

        vz = torch.rand_like(x)*0.2 - 0.1 # [-0.1,0.1]
        vz = vz + 4.5 # [4.4,4.6]

        return x,y,z,vx,vz

    def _reset_init_object_state(self, env_ids):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_object_state = torch.zeros(num_resets, 13, device=self.device)

        this_object_state_all = self._init_object_state

        # Randomize x, y, z value
        x,y,z,vx,vz = self.get_object_random_pose(num_resets)
        random_xyz = torch.zeros(num_resets, 3, device=self.device)
        random_xyz[:,0] = x
        random_xyz[:,1] = y
        random_xyz[:,2] = z

        sampled_object_state[:,7] = vx
        sampled_object_state[:,9] = vz

        sampled_object_state[:,:3] = random_xyz
        sampled_object_state[:,6] = 1

        this_object_state_all[env_ids, :] = sampled_object_state

    def pre_physics_step(self, actions):
        """
            actions : [-1,1]
        """
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.reset_buf[env_ids] = 0

        self.actions = actions.clone().to(self.device)

        d_ur5e_0 = self.actions[:,:6]
        d_allegro_0 = self.actions[:,6:22]

        d_ur5e_1 = self.actions[:,22:28]
        d_allegro_1 = self.actions[:,28:]

        # d_ur5e_0 = torch.zeros_like(d_ur5e_0)
        # d_allegro_0 = torch.zeros_like(d_allegro_0)
        # d_ur5e_1 = torch.zeros_like(d_ur5e_1)
        # d_allegro_1 = torch.zeros_like(d_allegro_1)

        ur5e_target_pos_0 =  self._pos_control[:,:6] + self.dt * self.action_scale * d_ur5e_0
        self._pos_control[:, :6] = tensor_clamp(ur5e_target_pos_0, self._ur5e_allegro_dof_lower_limits[:6], self._ur5e_allegro_dof_upper_limits[:6])

        allegro_target_pos_0 = self._pos_control[:,6:22] + self.dt * self.allegro_speed_scale * d_allegro_0
        self._pos_control[:, 6:22] = tensor_clamp(allegro_target_pos_0, self._ur5e_allegro_dof_lower_limits[6:22], self._ur5e_allegro_dof_upper_limits[6:22])

        ur5e_target_pos_1 = self._pos_control[:,22:28] + self.dt * self.action_scale * d_ur5e_1
        self._pos_control[:, 22:28] = tensor_clamp(ur5e_target_pos_1, self._ur5e_allegro_dof_lower_limits[22:28], self._ur5e_allegro_dof_upper_limits[22:28])

        allegro_target_pos_1 = self._pos_control[:,28:] + self.dt * self.allegro_speed_scale * d_allegro_1
        self._pos_control[:, 28:] = tensor_clamp(allegro_target_pos_1, self._ur5e_allegro_dof_lower_limits[28:], self._ur5e_allegro_dof_upper_limits[28:])

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

    def post_physics_step(self):
        self.progress_buf += 1

        # Using Camera Sensor in Headless Mode
        if self.viewer == None:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        self.compute_observations()
        # self.collect_expert_data()

        self.compute_reward(self.actions, self.d_closest_thumb_0, self.d_closest_index_0, self.d_closest_middle_0, self.d_closest_ring_0, self.d_closest_target_0, self.d_closest_align_0, self.d_closest_palm_0,
                                            self.d_closest_thumb_1, self.d_closest_index_1, self.d_closest_middle_1, self.d_closest_ring_1, self.d_closest_target_1, self.d_closest_align_1, self.d_closest_palm_1)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.gym.end_access_image_tensors(self.sim)

    def collect_expert_data(self):
        """
        robot_prop = self.obs_buf
        future_traj = self.obs_dict["pred_traj"]
        external_img = self.obs_dict["external_imgs"]
        """
        self.num_trans += 1
        for i in range(self.num_envs):
            # Set path to datasets
            dataset_root = "/home/lr-drgn/rl_env/expert_dataset_eef_diff/env%d"%(i)
            if not os.path.exists(dataset_root):
                os.makedirs(dataset_root)

            if not os.path.exists(os.path.join(dataset_root,"%dep"%(self.ep_num[i]))):
                os.makedirs(os.path.join(dataset_root,"%dep"%(self.ep_num[i])))

            robot_prop_pth   = os.path.join(os.path.join(dataset_root,"%dep"%(self.ep_num[i])),"robot_prop")
            # future_traj_pth  = os.path.join(dataset_root,"future_traj")
            # action_pth       = os.path.join(dataset_root,"action")
            # contact_info_pth = os.path.join(dataset_root,"binary_contact_info")
            # obj_pos_pth      = os.path.join(dataset_root,"obj_pos")
            # depth_img_pth        = os.path.join(dataset_root,"depth_img")
            # ext_img_pth      = os.path.join(dataset_root,"ext_img")
            # target_pth       = os.path.join(dataset_root,"target")
            # r_img_pth        = os.path.join(dataset_root,"r_img")
            # l_img_pth        = os.path.join(dataset_root,"l_img")
            front_img_pth    = os.path.join(os.path.join(dataset_root,"%dep"%(self.ep_num[i])),"front_img")

            # Create directories of datasets
            if not os.path.exists(robot_prop_pth):
                os.makedirs(robot_prop_pth)
            # if not os.path.exists(future_traj_pth):
            #     os.makedirs(future_traj_pth)
            # if not os.path.exists(action_pth):
            #     os.makedirs(action_pth)
            # if not os.path.exists(contact_info_pth):
            #     os.makedirs(contact_info_pth)
            # if not os.path.exists(obj_pos_pth):
            #         os.makedirs(obj_pos_pth)
            # if not os.path.exists(depth_img_pth):
            #         os.makedirs(depth_img_pth)
            # if not os.path.exists(ext_img_pth):
            #         os.makedirs(ext_img_pth)
            # if not os.path.exists(target_pth):
            #         os.makedirs(target_pth)
            # if not os.path.exists(r_img_pth):
            #         os.makedirs(r_img_pth)
            # if not os.path.exists(l_img_pth):
            #     os.makedirs(l_img_pth)
            if not os.path.exists(front_img_pth):
                os.makedirs(front_img_pth)

            # Save data
            if self.num_trans[i] == 1:
                np.savetxt(os.path.join(robot_prop_pth,"%dstep.txt"%(self.num_trans[i])),self.obs_buf[i,:32].cpu().numpy())
            # np.savetxt(os.path.join(robot_prop_pth,"last_step.txt"),self.obs_buf[i,:32].cpu().numpy())
            np.savetxt(os.path.join(robot_prop_pth,"eef_pos.txt"),self.states["ur5e_eef_pos"][i].cpu().numpy())
            # np.savetxt(os.path.join(future_traj_pth,"%dstep.txt"%(self.num_trans[i])),self.obs_dict["pred_traj"][i].cpu().numpy())
            # np.savetxt(os.path.join(action_pth,"%dstep.txt"%(self.num_trans[i])),self.actions[i].cpu().numpy())
            # np.savetxt(os.path.join(contact_info_pth,"%dstep.txt"%(self.num_trans[i])),self.obs_buf[i,42:58].cpu().numpy())
            # np.savetxt(os.path.join(obj_pos_pth,"%dstep.txt"%(self.num_trans[i])),self.obs_buf[i,32:35].cpu().numpy())
            if self.num_trans[i] < 5:
                front_img = im.fromarray(self.front_imgs[i].cpu().numpy().astype(np.uint8),mode="L")
                front_img.save(os.path.join(front_img_pth,"%dstep.png"%(self.num_trans[i])))
            # np.savetxt(os.path.join(target_pth,"%dstep.txt"%(self.num_trans[i])),self.obs_buf[i,-3:].cpu().numpy())
            # depth_img = im.fromarray(self.wrist_imgs[i].cpu().numpy().astype(np.uint8),mode="L")
            # depth_img.save(os.path.join(depth_img_pth,"%dstep.png"%(self.num_trans[i])))
            # ext_img = im.fromarray(self.ext_imgs[i].cpu().numpy().astype(np.uint8),mode="L")
            # ext_img.save(os.path.join(ext_img_pth,"%dstep.png"%(self.num_trans[i])))
            # r_img = im.fromarray(self.r_img[i].cpu().numpy())
            # r_img.save(os.path.join(r_img_pth,"%dstep.png"%(self.num_trans[i])))
            # l_img = im.fromarray(self.l_img[i].cpu().numpy())
            # l_img.save(os.path.join(l_img_pth,"%dstep.png"%(self.num_trans[i])))
            # l_img_int32 = self.l_img[i].cpu().numpy()
            # l_img_int8 = np.zeros((l_img_int32.shape[0], l_img_int32.shape[1], 3), dtype=np.uint8)
            # l_img_int8[:, :, 0] = (l_img_int32 >> 16) & 0xFF  # Red 채널
            # l_img_int8[:, :, 1] = (l_img_int32 >> 8) & 0xFF   # Green 채널
            # l_img_int8[:, :, 2] = l_img_int32 & 0xFF
            # l_img_int8 = im.fromarray(l_img_int8)
            # l_img_int8.save(os.path.join(l_img_pth,"%dstep.png"%(self.num_trans[i])))

from torch import Tensor
from typing import Dict, Tuple, List

@torch.jit.script
def compute_ur5e_reward(
    reset_buf, progress_buf, actions, states, max_episode_length, d_closest_thumb_0, d_closest_index_0, d_closest_middle_0, d_closest_ring_0, d_closest_target_0, d_closest_align_0, d_closest_palm_0,
                                                                     d_closest_thumb_1, d_closest_index_1, d_closest_middle_1, d_closest_ring_1, d_closest_target_1, d_closest_align_1, d_closest_palm_1):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    # Distance
    d_thumb_0 = torch.norm(states["obj_thumb_dist_0"], dim=-1)
    d_index_0 = torch.norm(states["obj_index_dist_0"], dim=-1)
    d_middle_0 = torch.norm(states["obj_middle_dist_0"], dim=-1)
    d_ring_0 = torch.norm(states["obj_ring_dist_0"], dim=-1)
    d_palm_0 = torch.norm(states["obj_palm_dist_0"], dim=-1)
    d_align_0 = torch.norm(states["obj_palm_align_0"], dim=-1)
    d_target_eef_0 = torch.norm(states["target_palm_dist_0"], dim=-1)

    d_thumb_1 = torch.norm(states["obj_thumb_dist_1"], dim=-1)
    d_index_1 = torch.norm(states["obj_index_dist_1"], dim=-1)
    d_middle_1 = torch.norm(states["obj_middle_dist_1"], dim=-1)
    d_ring_1 = torch.norm(states["obj_ring_dist_1"], dim=-1)
    d_palm_1 = torch.norm(states["obj_palm_dist_1"], dim=-1)
    d_align_1 = torch.norm(states["obj_palm_align_1"], dim=-1)
    d_target_eef_1 = torch.norm(states["target_palm_dist_1"], dim=-1)

    # False / True Tensor
    falseTensor = torch.zeros_like(d_thumb_0)
    trueTensor = torch.ones_like(d_thumb_0)

    # Base Position
    base_pos = torch.zeros_like(states["object_pos"])
    base_pos[:,2] = 0.825

    # Reward for aligning
    r_align_0 = torch.max(d_closest_align_0 - d_align_0, torch.zeros_like(d_thumb_0))
    r_align_1 = torch.max(d_closest_align_1 - d_align_1, torch.zeros_like(d_thumb_0))

    d_closest_align_0 = torch.where(d_closest_align_0 > d_align_0, d_align_0, d_closest_align_0)
    d_closest_align_1 = torch.where(d_closest_align_1 > d_align_1, d_align_1, d_closest_align_1)

    # Reward for grasping
    r_thumb_0 = torch.max(d_closest_thumb_0 - d_thumb_0, torch.zeros_like(d_thumb_0))
    r_index_0 = torch.max(d_closest_index_0 - d_index_0, torch.zeros_like(d_thumb_0))
    r_middle_0 = torch.max(d_closest_middle_0 - d_middle_0, torch.zeros_like(d_thumb_0))
    r_ring_0 = torch.max(d_closest_ring_0 - d_ring_0, torch.zeros_like(d_thumb_0))
    r_palm_0 = torch.max(d_closest_palm_0 - d_palm_0, torch.zeros_like(d_palm_0))

    d_closest_thumb_0 = torch.where(d_closest_thumb_0 > d_thumb_0, d_thumb_0, d_closest_thumb_0)
    d_closest_index_0 = torch.where(d_closest_index_0 > d_index_0, d_index_0, d_closest_index_0)
    d_closest_middle_0 = torch.where(d_closest_middle_0 > d_middle_0, d_middle_0, d_closest_middle_0)
    d_closest_ring_0 = torch.where(d_closest_ring_0 > d_ring_0, d_ring_0, d_closest_ring_0)
    d_closest_palm_0 = torch.where(d_closest_palm_0 > d_palm_0, d_palm_0, d_closest_palm_0)

    r_thumb_1 = torch.max(d_closest_thumb_1 - d_thumb_1, torch.zeros_like(d_thumb_1))
    r_index_1 = torch.max(d_closest_index_1 - d_index_1, torch.zeros_like(d_thumb_1))
    r_middle_1 = torch.max(d_closest_middle_1 - d_middle_1, torch.zeros_like(d_thumb_1))
    r_ring_1 = torch.max(d_closest_ring_1 - d_ring_1, torch.zeros_like(d_thumb_1))
    r_palm_1 = torch.max(d_closest_palm_1 - d_palm_1, torch.zeros_like(d_palm_1))

    d_closest_thumb_1 = torch.where(d_closest_thumb_1 > d_thumb_1, d_thumb_1, d_closest_thumb_1)
    d_closest_index_1 = torch.where(d_closest_index_1 > d_index_1, d_index_1, d_closest_index_1)
    d_closest_middle_1 = torch.where(d_closest_middle_1 > d_middle_1, d_middle_1, d_closest_middle_1)
    d_closest_ring_1 = torch.where(d_closest_ring_1 > d_ring_1, d_ring_1, d_closest_ring_1)
    d_closest_palm_1 = torch.where(d_closest_palm_1 > d_palm_1, d_palm_1, d_closest_palm_1)

    r_grasp_0 = 0.1*r_thumb_0 + 0.1*r_index_0 + 0.1*r_middle_0 + 0.1*r_ring_0 + 0.6*r_palm_0
    r_grasp_1 = 0.1*r_thumb_1 + 0.1*r_index_1 + 0.1*r_middle_1 + 0.1*r_ring_1 + 0.6*r_palm_1

    mean_d_fingers_0 = (d_thumb_0 + d_index_0 + d_middle_0 + d_ring_0) / 4
    mean_d_fingers_1 = (d_thumb_1 + d_index_1 + d_middle_1 + d_ring_1) / 4

    mean_d_hand = (mean_d_fingers_0 + mean_d_fingers_1 + d_palm_0 + d_palm_1) / 4

    # Penalty for collision between hands
    d_palms = torch.norm(states["palm_pos_0"]-states["palm_pos_1"], dim=-1)
    p_collision = 1 - torch.tanh(4*d_palms)

    # Success condition & Bonus reward
    is_grasped = torch.where((d_palm_0 + d_palm_1)/2 < 0.11, trueTensor, falseTensor)
    is_aligned = torch.where((d_align_0 < 0.05) & (d_align_1 < 0.05), trueTensor, falseTensor)
    grasp_bonus = torch.where(is_grasped, 3.0*trueTensor, falseTensor)

    align_alpha = 1.0
    grasp_alpha = 10.0

    # print(d_fingers[0])
    r_align = align_alpha*(r_align_0 + r_align_1)
    r_grasp = (1-is_grasped)*grasp_alpha*(r_palm_0 + r_palm_1)
    # r_task = (1-is_aligned)*r_align + is_aligned*r_grasp + is_grasped*grasp_bonus - 0.2*p_collision
    r_task = (1-is_aligned)*r_align + is_aligned*r_grasp + is_grasped*grasp_bonus

    # Reset condition
    d_reset = (states["object_pos"][:,0] < (-0.1)) | ((progress_buf > 10)&(states["object_pos"][:,2] < base_pos[:,2] + 0.35))

    # print(progress_buf[0])
    # print(r_task[0])
    # print(torch.sum(is_located))

    # Compute Reset
    reset_buf = torch.where((progress_buf >= max_episode_length) | d_reset, torch.ones_like(reset_buf), reset_buf)

    return r_task, reset_buf, d_closest_thumb_0, d_closest_index_0, d_closest_middle_0, d_closest_ring_0, d_closest_target_0, d_closest_align_0, d_closest_palm_0, d_closest_thumb_1, d_closest_index_1, d_closest_middle_1, d_closest_ring_1, d_closest_target_1, d_closest_align_1, d_closest_palm_1