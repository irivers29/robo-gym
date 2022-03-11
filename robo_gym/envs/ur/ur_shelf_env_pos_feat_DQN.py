#!/usr/bin/env python3
import copy
from typing import Tuple
import matplotlib.pyplot as plt

import cv2
import gym
import numpy as np
import robo_gym_server_modules.robot_server.client as rs_client
import rospy
import tf2_ros
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym.envs.ur.ur_shelf_env_positioning import URShelfPositioning
from robo_gym.utils import ur_utils, utils
from robo_gym.utils.exceptions import (InvalidActionError, InvalidStateError,
                                       RobotServerError)
from robo_gym_server_modules.robot_server.grpc_msgs.python import \
    robot_server_pb2
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Header
from trac_ik_python.trac_ik import IK

JOINT_POSITIONS = [-1.383, -1.804, -1.526, -2.952, -1.382, -1.571, 0.7]
DISTANCE_THRESHOLD = 0.1

class URShelfFeatDQN(URShelfPositioning):
    def __init__(self, rs_address=None, fix_base=False, fix_shoulder=False,
                 fix_elbow=False, fix_wrist_1=False, fix_wrist_2=False, fix_wrist_3=False, fix_gripper=False,
                 ur_model='ur10e', include_polar_to_elbow=False, rs_state_to_info=True, **kwargs):
        
        with open('robot_description_urdf', 'r') as file:
            data = file.read()

        self.include_polar_to_elbow = include_polar_to_elbow
        self.ik_solver = IK("base_link","tool0", urdf_string = data)
        self.roll, self.pitch, self.yaw = 0.0, 1.56, 1.57
        
        super().__init__(rs_address=rs_address, fix_base=fix_base, fix_shoulder=fix_shoulder,
                        fix_elbow=fix_elbow, fix_wrist_1=fix_wrist_1, fix_wrist_2=fix_wrist_2,
                        fix_wrist_3=fix_wrist_3, fix_gripper=fix_gripper, ur_model=ur_model)

    def _get_observation_space(self) -> gym.spaces.Dict:

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

        # Definition of environment observation_space
        max_obs = np.concatenate((positions_max, max_joint_positions, max_joint_velocities))
        min_obs = np.concatenate((positions_min, min_joint_positions, min_joint_velocities))
          
        #return gym.spaces.Dict({
        #            'image': gym.spaces.Box(low=0, high=255, shape = (388, 363, 3), dtype=np.uint8),
        #            'feature_vector': gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32),
        #            'depth_image': gym.spaces.Box(low=0, high=255, shape = (388, 363, 1), dtype=np.uint8)})
        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

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
        #state_len = self.observation_space.shape[0]
        #state = np.zeros(state_len)
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
        #state_image = self.client.get_state_msg().image
        #state_depth = self.client.get_state_msg().depth
        
        # Convert image to numpy
        #state_image = self.image_state_to_numpy(state_image)
        #state_depth = self._depth_to_numpy(state_depth)

        # Check if the length and keys of the Robot Server state received is correct
        self._check_rs_state_keys(rs_state)

        state_feature = self._robot_server_state_to_env_state(rs_state)

        state = state_feature


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

        return state
    
    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        if type(action) == list: action = np.array(action)
        info_dict = {}
            
        self.elapsed_steps += 1

        # Check if the action is contained in the action space
        if not self.action_space.contains(action):
            print(action)
            raise InvalidActionError()

        # Add missing joints which were fixed at initialization

        state_dict = self.client.get_state_msg().state_dict

        # Convert environment action to robot server action
        rs_action = self.env_action_to_rs_action(action, state_dict)

        no_ik = False

        if rs_action is None:
            rs_state = state_dict
            no_ik = True
        else:
            # Send action to Robot Server and get state
            rs_state = self.client.send_action_get_state(rs_action.tolist()).state_dict
        
        
        #state_image = self.client.get_state_msg().image
        #state_depth = self.client.get_state_msg().depth
        # Convert image to numpy
        #state_image = self.image_state_to_numpy(state_image)
        #state_depth = self._depth_to_numpy(state_depth)

        
        self._check_rs_state_keys(rs_state)
        state_feature = self._robot_server_state_to_env_state(rs_state)

        state = state_feature

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(state):
            print(state)
            raise InvalidStateError()

        self.rs_state = rs_state

        # Assign reward
        reward = 0
        done = False
        reward, done, info = self.reward(rs_state=rs_state, action=action, no_ik=no_ik)
        if self.rs_state_to_info: info['rs_state'] = self.rs_state

        return state, reward, done, info

    def image_state_to_numpy(self, byte_image):
        "reconvert image in bytes to numpy array"
        np_image = np.frombuffer(byte_image, dtype=np.uint8)
        np_image = np_image.reshape((480,640,3))
        np_image = np_image[0:388, 227:590]
        return np_image

    def _depth_to_numpy(self, byte_depth):
        im = np.ndarray(shape=(480, 640, 1),
                            dtype=np.uint16, buffer=byte_depth)
        np_depth = im.copy()
        np_depth = np_depth[0:388, 227:590]
        np_depth = np.uint8(np_depth*255.0/np_depth.max())
        #np_depth = np_depth.reshape((480,640,1))
        
        return np_depth
        
    def reward(self, rs_state, action, no_ik=False) -> Tuple[float, bool, dict]:
            done = False
            info = {}

            # Calculate distance to the target
            target_coord = np.array([rs_state['object_0_to_ref_translation_x'], rs_state['object_0_to_ref_translation_y'], rs_state['object_0_to_ref_translation_z']])
            ee_coord = np.array([rs_state['ee_to_ref_translation_x'], rs_state['ee_to_ref_translation_y'], rs_state['ee_to_ref_translation_z']])
            euclidean_dist_3d = np.linalg.norm(target_coord - ee_coord)
        

            # Reward base

            reward = -1
            #if euclidean_dist_3d != 0:
            reward += (1/euclidean_dist_3d)


            if euclidean_dist_3d <= DISTANCE_THRESHOLD:
                reward += 50
                done = True
                info['final_status'] = 'success'
            
            if rs_state['in_collision'] == 1:
                collision = True
            else:
                collision = False
            if no_ik:
                reward = -2
                done = True
                info['final_status'] = 'no_ik'
            if collision:
                reward = -20
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

            'tip_to_ref_translation_x',
            'tip_to_ref_translation_y',
            'tip_to_ref_translation_z',
            'tip_to_ref_rotation_x',
            'tip_to_ref_rotation_y',
            'tip_to_ref_rotation_z',
            'tip_to_ref_rotation_w',

            'in_collision'
        ]
        return rs_state_keys

    def env_action_to_rs_action(self, action, state_dict) -> np.array:
        # convert discrete action to rs_action
        discrete_action = copy.deepcopy(action)
        seed_state = self._get_seed_state(state_dict)

        x = state_dict['tip_to_ref_translation_x']
        y = state_dict['tip_to_ref_translation_y']
        z = state_dict['tip_to_ref_translation_z']

        gripper_act = None
        # Forwards
        if discrete_action == 0:
            y += 0.02
        # Backwards
        elif discrete_action == 1:
            y -= 0.02
        # Left
        elif discrete_action == 2:
            x -= 0.02
        # Right
        elif discrete_action == 3:
            x += 0.02
        # Up
        elif discrete_action == 4:
            z += 0.02
        # Down
        elif discrete_action == 5:
            z -= 0.02
        # Open
        elif discrete_action == 6:
            gripper_act = "open"
        # Close
        elif discrete_action == 7:
            gripper_act = "close"
        # Nothing
        elif discrete_action == 8:
            pass

        z += 0.004
        
        print("seed state:", seed_state)
        seed_state = self._get_seed_state(state_dict)
        qx, qy, qz, qw = self.euler_to_quat(self.roll, self.pitch, self.yaw)
        solution = self.ik_solver.get_ik(seed_state, x,y,z,qx,qy,qz,qw)
        print("solution", solution)
        print("discrete_action", discrete_action)
        print("position", x,y,z,qx,qy,qz,qw)
        if solution == None:
            return None

        solution = list(solution)
        if gripper_act == "open":
            solution.append(0.79)
        elif gripper_act == "close":
            solution.append(0.05)
        else:
            solution.append(state_dict['robotiq_85_left_knuckle_joint_position'])
        
        rs_action = np.array(solution)
        rs_action = self.ur._ur_joint_list_to_ros_joint_list(rs_action)
        return rs_action

    def _get_action_space(self)-> gym.spaces.Box:
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """
        return gym.spaces.Discrete(9)

    def euler_to_quat(self, roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return qx,qy,qz,qw

    def _get_seed_state(self, state_dict):
        seed_state = [state_dict['base_joint_position'], state_dict['shoulder_joint_position'], state_dict['elbow_joint_position'],
                        state_dict['wrist_1_joint_position'], state_dict['wrist_2_joint_position'], state_dict['wrist_3_joint_position']]

        return seed_state

class URShelfFeatDQNSim(URShelfFeatDQN, Simulation):
    cmd = "roslaunch ur_robot_server pruebas.launch \
        world_name:=shelf.world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        reference_frame:=base_link \
        rviz_gui:=false \
        gazebo_gui:=true \
        rs_mode:=DQN \
        n_objects:=3.0 \
        gripper:=true \
        objects_controller:=true \
        object_0_model_name:=coke_can \
        object_0_frame:=target \
        object_1_model_name:=pringles_red \
        object_1_frame:=target1 \
        object_2_model_name:=tomato_sauce \
        object_2_frame:=target2 \
        "
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, ur_model='ur10e', **kwargs):
        self.cmd = self.cmd + ' ' + 'ur_model:=' + ur_model
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        URShelfFeatDQN.__init__(self, rs_address=self.robot_server_ip, ur_model=ur_model, **kwargs)

class URShelfFeatDQNRob(URShelfFeatDQN):
    real_robot = True
    
#object_3_model_name:=pringles_red \
#object_3_frame:=target3 \
#object_4_model_name:=tomato_sauce \
#object_4_frame:=target4 \
#object_5_model_name:=milk \
#object_5_frame:=target5"
