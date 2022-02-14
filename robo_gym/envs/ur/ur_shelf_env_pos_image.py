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
from robo_gym.envs.ur.ur_shelf_env_positioning import URShelfPositioning
from robo_gym.utils import ur_utils, utils
from robo_gym.utils.exceptions import (InvalidActionError, InvalidStateError,
                                       RobotServerError)
from robo_gym_server_modules.robot_server.grpc_msgs.python import \
    robot_server_pb2
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Header

JOINT_POSITIONS = [0.0, -2.5, 1.5, 0.0, -1.4, 2.0, 0.7]
DISTANCE_THRESHOLD = 0.05

class URShelfImPositioning(URShelfPositioning):
    def __init__(self, rs_address=None, fix_base=False, fix_shoulder=False,
                 fix_elbow=False, fix_wrist_1=False, fix_wrist_2=False, fix_wrist_3=False, fix_gripper=False,
                 ur_model='ur10e', include_polar_to_elbow=False, rs_state_to_info=True, **kwargs):
        self.include_polar_to_elbow = include_polar_to_elbow
        super().__init__(rs_address=rs_address, fix_base=fix_base, fix_shoulder=fix_shoulder,
                        fix_elbow=fix_elbow, fix_wrist_1=fix_wrist_1, fix_wrist_2=fix_wrist_2,
                        fix_wrist_3=fix_wrist_3, fix_gripper=fix_gripper, ur_model=ur_model)

    def _get_observation_space(self) -> gym.spaces.Box:
        
        return gym.spaces.Box(low=0, high=255, shape = (480, 640, 3), dtype=np.float32)

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
        state = self.client.get_state_msg().image
        
        # Convert image to numpy
        state = self.image_state_to_numpy(state)

        # Check if the length and keys of the Robot Server state received is correct
        self._check_rs_state_keys(rs_state)


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
        action = self.add_fixed_joints(action)

        # Convert environment action to robot server action
        rs_action = self.env_action_to_rs_action(action)
        # Send action to Robot Server and get state
        rs_state = self.client.send_action_get_state(rs_action.tolist()).state_dict
        state = self.client.get_state_msg().image

        # Convert byte image to numpy image
        state = self.image_state_to_numpy(state)
        
        self._check_rs_state_keys(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(state):
            print(state)
            raise InvalidStateError()

        self.rs_state = rs_state

        # Assign reward
        reward = 0
        done = False
        reward, done, info = self.reward(rs_state=rs_state, action=action)
        if self.rs_state_to_info: info['rs_state'] = self.rs_state

        return state, reward, done, info

    def image_state_to_numpy(self, byte_image):
        "reconvert image in bytes to numpy array"
        np_image = np.frombuffer(byte_image, dtype=np.uint8)
        np_image = np_image.reshape((480,640,3))
        #plt.imshow(np_image)
        #plt.show()

        return np_image

class URShelfPosImSim(URShelfImPositioning, Simulation):
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
        URShelfImPositioning.__init__(self, rs_address=self.robot_server_ip, ur_model=ur_model, **kwargs)

class URShelfPosImRob(URShelfImPositioning):
    real_robot = True
    