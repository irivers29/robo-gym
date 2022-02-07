#!/usr/bin/env python3
import copy
from typing import Tuple

import cv2
import gym
import numpy as np
import robo_gym_server_modules.robot_server.client as rs_client
import rospy
import tf2_ros
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym.envs.ur.ur_shelf_env import URBaseEnv
from robo_gym.utils import ur_utils, utils
from robo_gym.utils.exceptions import (InvalidActionError, InvalidStateError,
                                       RobotServerError)
from robo_gym_server_modules.robot_server.grpc_msgs.python import \
    robot_server_pb2
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Header

# base, shoulder, elbow, wrist_1, wrist_2, wrist_3, gripper
JOINT_POSITIONS = [0.0, -2.5, 1.5, 0.0, -1.4, 2.0, 0.7]
DISTANCE_THRESHOLD = 0.1

class URShelfPositioning(URBaseEnv):
    def __init__(self, rs_address=None, fix_base=False, fix_shoulder=False,
                 fix_elbow=False, fix_wrist_1=False, fix_wrist_2=False, fix_wrist_3=True, fix_gripper=False,
                 ur_model='ur10e', include_polar_to_elbow=False, rs_state_to_info=True, **kwargs):
        self.include_polar_to_elbow = include_polar_to_elbow
        super().__init__(rs_address=rs_address, fix_base=fix_base, fix_shoulder=fix_shoulder,
                        fix_elbow=fix_elbow, fix_wrist_1=fix_wrist_1, fix_wrist_2=fix_wrist_2,
                        fix_wrist_3=fix_wrist_3, fix_gripper=fix_gripper, ur_model=ur_model)

    def _get_observation_space(self) -> gym.spaces.Box:
        
        #Joint position range tolerance
        pos_tolerance = np.full(7,0.1)

        #Joint position range used to determine if there is an error in the sensor readigs
        max_joint_positions = np.add(np.full(7, 1.0), pos_tolerance)
        min_joint_positions = np.subtract(np.full(7, -1.0), pos_tolerance)
        #min_joint_pos_gripper = np.full(1,0.0)

        #min_joint_positions = np.concatenate((min_joint_positions, min_joint_pos_gripper))
        #min_joint_positions = np.subtract(min_joint_positions, pos_tolerance)
        # Joint velocities range 
        positions_max = np.array([np.inf]*10)
        positions_min = -np.array([np.inf]*10)
        max_joint_velocities = np.array([np.inf] * 7)
        min_joint_velocities = -np.array([np.inf] * 7)

        target_range = np.full(3,np.inf)

        # Definition of environment observation_space
        max_obs = np.concatenate((positions_max, max_joint_positions, max_joint_velocities))
        min_obs = np.concatenate((positions_min, min_joint_positions, min_joint_velocities))
        
        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def _set_initial_robot_server_state(self, rs_state, ee_target_pose) -> robot_server_pb2.State:
        string_params = {"object_0_function": "fixed_position"}
        float_params = {"object_0_x": ee_target_pose[0], 
                        "object_0_y": ee_target_pose[1], 
                        "object_0_z": ee_target_pose[2]}
        state = {}

        state_msg = robot_server_pb2.State(state = state, float_params = float_params, string_params = string_params, state_dict = rs_state)
        return state_msg

    def _robot_server_state_to_env_state(self, rs_state) -> np.array:
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """
        # Target polar coordinates
        # Transform cartesian coordinates of target to polar coordinates 
        # with respect to the end effector frame
        target_coord = np.array([
            rs_state['object_0_to_ref_translation_x'], 
            rs_state['object_0_to_ref_translation_y'],
            rs_state['object_0_to_ref_translation_z']])

        ee_to_ref_frame_translation = np.array([
            rs_state['ee_to_ref_translation_x'], 
            rs_state['ee_to_ref_translation_y'],
            rs_state['ee_to_ref_translation_z']])

        ee_to_ref_frame_quaternion = np.array([
            rs_state['ee_to_ref_rotation_x'], 
            rs_state['ee_to_ref_rotation_y'],
            rs_state['ee_to_ref_rotation_z'],
            rs_state['ee_to_ref_rotation_w']])

        #ee_to_ref_frame_rotation = R.from_quat(ee_to_ref_frame_quaternion)
        #ref_frame_to_ee_rotation = ee_to_ref_frame_rotation.inv()
        # to invert the homogeneous transformation
        # R' = R^-1
        #ref_frame_to_ee_quaternion = ref_frame_to_ee_rotation.as_quat()
        # t' = - R^-1 * t
        #ref_frame_to_ee_translation = -ref_frame_to_ee_rotation.apply(ee_to_ref_frame_translation)

        #target_coord_ee_frame = utils.change_reference_frame(target_coord,ref_frame_to_ee_translation,ref_frame_to_ee_quaternion)
        #target_polar = utils.cartesian_to_polar_3d(target_coord_ee_frame)


        # Joint positions 
        joint_positions = []
        joint_positions_keys = ['base_joint_position', 'shoulder_joint_position', 'elbow_joint_position',
                            'wrist_1_joint_position', 'wrist_2_joint_position', 'wrist_3_joint_position', 'robotiq_85_left_knuckle_joint_position']
        for position in joint_positions_keys:
            joint_positions.append(rs_state[position])
        joint_positions = np.array(joint_positions)
        # Normalize joint position values
        joint_positions = self.ur.normalize_joint_values(joints=joint_positions)

        # Joint Velocities
        joint_velocities = [] 
        joint_velocities_keys = ['base_joint_velocity', 'shoulder_joint_velocity', 'elbow_joint_velocity',
                            'wrist_1_joint_velocity', 'wrist_2_joint_velocity', 'wrist_3_joint_velocity',
                            'robotiq_85_left_knuckle_joint_velocity']
        for velocity in joint_velocities_keys:
            joint_velocities.append(rs_state[velocity])
        joint_velocities = np.array(joint_velocities)



        # Compose environment state
        #state = np.concatenate((target_polar, joint_positions, joint_velocities))
        state = np.concatenate((target_coord, ee_to_ref_frame_translation, ee_to_ref_frame_quaternion, joint_positions, joint_velocities))
        state = np.float32(state)

        return state
    
    def get_robot_server_composition(self) -> list:
        rs_state_keys = [
            'object_0_to_ref_translation_x', 
            'object_0_to_ref_translation_y',
            'object_0_to_ref_translation_z',
            'object_0_to_ref_rotation_x',
            'object_0_to_ref_rotation_y',
            'object_0_to_ref_rotation_z',
            'object_0_to_ref_rotation_w',

            'base_joint_position',
            'shoulder_joint_position',
            'elbow_joint_position',
            'wrist_1_joint_position',
            'wrist_2_joint_position',
            'wrist_3_joint_position',
            'robotiq_85_left_knuckle_joint_position',

            'base_joint_velocity',
            'shoulder_joint_velocity',
            'elbow_joint_velocity',
            'wrist_1_joint_velocity',
            'wrist_2_joint_velocity',
            'wrist_3_joint_velocity',
            'robotiq_85_left_knuckle_joint_velocity',

            'ee_to_ref_translation_x',
            'ee_to_ref_translation_y',
            'ee_to_ref_translation_z',
            'ee_to_ref_rotation_x',
            'ee_to_ref_rotation_y',
            'ee_to_ref_rotation_z',
            'ee_to_ref_rotation_w',

            'in_collision'
        ]
        return rs_state_keys
    def reset(self, joint_positions = None, ee_target_pose=None) -> np.array:
        """Environment reset.

        Args:
            joint_positions (list[7] or np.array[7]): robot joint positions in radians. Order is defined by 
        
        Returns:
            np.array: Environment state.

        """
        info = {}

        if joint_positions: 
            assert len(joint_positions) == 7
        else:
            joint_positions = JOINT_POSITIONS

        self.elapsed_steps = 0

        # Initialize environment state
        state_len = self.observation_space.shape[0]
        state = np.zeros(state_len)
        rs_state = dict.fromkeys(self.get_robot_server_composition(), 0.0)

        ## Maybe write code to randomize positions
        
        # Set initial robot joint positions
        self._set_joint_positions(joint_positions)    

        # Update joint positions in rs_state
        rs_state.update(self.joint_positions)

        # Set target End Effector pose
        if ee_target_pose:
            assert len(ee_target_pose) == 6
        else:
            ee_target_pose = self._get_target_pose()

        # Set initial state of the Robot Server
        state_msg = self._set_initial_robot_server_state(rs_state, ee_target_pose)
        
        if not self.client.set_state_msg(state_msg):
            print("Failed because couln't set state_msg")
            raise RobotServerError("set_state")
        # Get Robot Server state

        rs_state = self.client.get_state_msg().state_dict
        image = self.client.get_state_msg().image
        #state_ign = self.client.get_state

        # Check if the length and keys of the Robot Server state received is correct
        self._check_rs_state_keys(rs_state)

        # Convert the initial state from Robot Server format to environment format
        state = self._robot_server_state_to_env_state(rs_state)


        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(state):
            print("failed because state not in state space", state)
            raise InvalidStateError()

        # Check if current position is in the range of the initial joint positions
        for joint in self.joint_positions.keys():
            if not np.isclose(self.joint_positions[joint], rs_state[joint], atol=0.05):
                print("failed because is not close")
                raise InvalidStateError('Reset joint positions are not within defined range')


        target_coord = np.array([rs_state['object_0_to_ref_translation_x'], rs_state['object_0_to_ref_translation_y'], rs_state['object_0_to_ref_translation_z']])
        ee_coord = np.array([rs_state['ee_to_ref_translation_x'], rs_state['ee_to_ref_translation_y'], rs_state['ee_to_ref_translation_z']])

        info['target_coord'] = target_coord
        info['ee_coord'] = ee_coord

        return state#, info

    def _get_target_pose(self):
        """Generate target End Effector pose.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """
        return self.ur.get_random_workspace_pose()
    
    def _check_rs_state_keys(self, rs_state) -> None:
        keys = self.get_robot_server_composition()
        if not len(keys) == len(rs_state.keys()):
            raise InvalidStateError("Robot Server state keys to not match. Different lengths.")

        
        for key in keys:
            if key not in rs_state.keys():
                raise InvalidStateError("Robot Server state keys to not match")

    def compute_reward_HER(self, achieved_goal, desired_goal, rs_state):
        reward = 0
        done = False
        info = {}

        # Calculate distance to the target
        target_coord = desired_goal
        ee_coord = achieved_goal

        euclidean_dist_3d = np.linalg.norm(target_coord - ee_coord)


        if euclidean_dist_3d <= DISTANCE_THRESHOLD:
            reward = 100
            done = True
            info['final_status'] = 'success'

        if rs_state['in_collision'] == 1:
            collision = True
        else:
            collision = False
            
        if collision:
            reward = -400
            done = True
            info['final_status'] = 'collision'
            
            
        # Check if robot is in collision
            
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'
            
        info['target_coord'] = target_coord
        info['ee_coord'] = ee_coord

        
        return reward, done, info

    def reward(self, rs_state, action) -> Tuple[float, bool, dict]:
        reward = 0
        done = False
        info = {}

        # Calculate distance to the target
        target_coord = np.array([rs_state['object_0_to_ref_translation_x'], rs_state['object_0_to_ref_translation_y'], rs_state['object_0_to_ref_translation_z']])
        ee_coord = np.array([rs_state['ee_to_ref_translation_x'], rs_state['ee_to_ref_translation_y'], rs_state['ee_to_ref_translation_z']])
        euclidean_dist_3d = np.linalg.norm(target_coord - ee_coord)
    

        # Reward base
        reward = 0
        if euclidean_dist_3d < 0.8:
            reward = 0.8 - euclidean_dist_3d
        # reward = reward + (-1/300)
        
        # Joint positions 
        #joint_positions = []
        #joint_positions_keys = ['base_joint_position', 'shoulder_joint_position', 'elbow_joint_position',
        #                    'wrist_1_joint_position', 'wrist_2_joint_position', 'wrist_3_joint_position']
        #for position in joint_positions_keys:
        #    joint_positions.append(rs_state[position])
        #joint_positions = np.array(joint_positions)
        #joint_positions_normalized = self.ur.normalize_joint_values(joint_positions)
        
        #delta = np.abs(np.subtract(joint_positions_normalized, action))
        # reward = reward - (0.05 * np.sum(delta))


        if euclidean_dist_3d <= DISTANCE_THRESHOLD:
            reward = 200
            done = True
            info['final_status'] = 'success'
        
        if rs_state['in_collision'] == 1:
            collision = True
        else:
            collision = False
        
        if collision:
            reward = 0
            done = True
            info['final_status'] = 'collision'
            
            
        # Check if robot is in collision
            
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'
            
        info['target_coord'] = target_coord
        info['ee_coord'] = ee_coord
        print(reward)
        return reward, done, info

class URShelfPositioningSim(URShelfPositioning, Simulation):
    cmd = "roslaunch ur_robot_server pruebas.launch \
        world_name:=shelf.world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        reference_frame:=base_link \
        rviz_gui:=false \
        gazebo_gui:=true \
        rs_mode:=1object \
        n_objects:=6.0 \
        gripper:=true \
        objects_controller:=true \
        object_0_model_name:=coke_can \
        object_0_frame:=target \
        object_1_model_name:=apple_juice_box \
        object_1_frame:=target1 \
        object_2_model_name:=spaghetti \
        object_2_frame:=target2 \
        object_3_model_name:=pringles_red \
        object_3_frame:=target3 \
        object_4_model_name:=tomato_sauce \
        object_4_frame:=target4 \
        object_5_model_name:=milk \
        object_5_frame:=target5"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, ur_model='ur10e', **kwargs):
        self.cmd = self.cmd + ' ' + 'ur_model:=' + ur_model
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        URShelfPositioning.__init__(self, rs_address=self.robot_server_ip, ur_model=ur_model, **kwargs)

class URShelfPositioningRob(URShelfPositioning):
    real_robot = True
    
