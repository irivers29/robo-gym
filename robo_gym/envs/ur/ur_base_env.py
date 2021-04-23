#!/usr/bin/env python3
import copy
import numpy as np
import gym
from typing import Tuple
from robo_gym.utils import ur_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError, InvalidActionError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from robo_gym.envs.simulation_wrapper import Simulation

# base, shoulder, elbow, wrist_1, wrist_2, wrist_3
JOINT_POSITIONS = [0.0, -2.5, 1.5, 0.0, -1.4, 0.0]

class URBaseEnv(gym.Env):
    """Universal Robots UR base environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.
        fix_base (bool): Wether or not the base joint stays fixed or is moveable. Defaults to False.
        fix_shoulder (bool): Wether or not the shoulder joint stays fixed or is moveable. Defaults to False.
        fix_elbow (bool): Wether or not the elbow joint stays fixed or is moveable. Defaults to False.
        fix_wrist_1 (bool): Wether or not the wrist 1 joint stays fixed or is moveable. Defaults to False.
        fix_wrist_2 (bool): Wether or not the wrist 2 joint stays fixed or is moveable. Defaults to False.
        fix_wrist_3 (bool): Wether or not the wrist 3 joint stays fixed or is moveable. Defaults to True.
        ur_model (str): determines which ur model will be used in the environment. Default to 'ur5'.

    Attributes:
        ur (:obj:): Robot utilities object.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.

    """
    real_robot = False
    max_episode_steps = 300

    def __init__(self, rs_address=None, fix_base=False, fix_shoulder=False, fix_elbow=False, fix_wrist_1=False, fix_wrist_2=False, fix_wrist_3=True, ur_model='ur5', **kwargs):
        self.ur = ur_utils.UR(model=ur_model)
        self.elapsed_steps = 0

        self.fix_base = fix_base
        self.fix_shoulder = fix_shoulder
        self.fix_elbow = fix_elbow
        self.fix_wrist_1 = fix_wrist_1
        self.fix_wrist_2 = fix_wrist_2
        self.fix_wrist_3 = fix_wrist_3

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.distance_threshold = 0.1
        self.abs_joint_pos_range = self.ur.get_max_joint_positions()
        self.last_action = None
        
        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")


    def _set_initial_robot_server_state(self, rs_state) -> robot_server_pb2.State:
        string_params = {}
        float_params = {}
        state = {}

        state_msg = robot_server_pb2.State(state = state, float_params = float_params, 
                                            string_params = string_params, state_dict = rs_state)
        return state_msg

    def reset(self, joint_positions = None) -> np.array:
        """Environment reset.

        Args:
            joint_positions (list[6] or np.array[6]): robot joint positions in radians. Order is defined by 
        
        Returns:
            np.array: Environment state.

        """
        if joint_positions: 
            assert len(joint_positions) == 6
        else:
            joint_positions = JOINT_POSITIONS

        self.elapsed_steps = 0

        # Initialize environment state
        state_len = self.observation_space.shape[0]
        self.state = np.zeros(state_len)
        rs_state = self._get_robot_server_composition()

        # Set initial robot joint positions
        self._set_joint_positions(joint_positions)

        # Update joint positions in rs_state
        rs_state.update(self.joint_positions)

        # Set initial state of the Robot Server
        state_msg = self._set_initial_robot_server_state(rs_state)
        
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = self.client.get_state_msg().state_dict

        # Check if the length of the Robot Server state received is correct
        if not len(rs_state)== self._get_robot_server_state_len():
            raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        # Check if current position is in the range of the initial joint positions
        for joint in self.joint_positions.keys():
            if not np.isclose(self.joint_positions[joint], rs_state[joint], atol=0.05):
                raise InvalidStateError('Reset joint positions are not within defined range')

        return self.state

    def _reward(self, rs_state, action) -> Tuple[float, bool, dict]:
        done = False
        info = {}

        # Check if robot is in collision
        collision = True if rs_state['in_collision'] == 1 else False
        if collision:
            done = True
            info['final_status'] = 'collision'

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'
            
        
        return 0, done, info

    def add_fixed_joints(self, action) -> np.array:
        action = action.tolist()
        fixed_joints = np.array([self.fix_base, self.fix_shoulder, self.fix_elbow, self.fix_wrist_1, self.fix_wrist_2, self.fix_wrist_3])
        fixed_joint_indices = np.where(fixed_joints)[0]

        joint_pos_names = ['base_joint_position', 'shoulder_joint_position', 'elbow_joint_position',
                            'wrist_1_joint_position', 'wrist_2_joint_position', 'wrist_3_joint_position']
        joint_positions_dict = self._get_joint_positions()
        
        joint_positions = np.array([joint_positions_dict.get(joint_pos) for joint_pos in joint_pos_names])


        joints_position_norm = self.ur.normalize_joint_values(joints=joint_positions)

        temp = []
        for joint in range(len(fixed_joints)):
            if joint in fixed_joint_indices:
                temp.append(joints_position_norm[joint])
            else:
                temp.append(action.pop(0))
        return np.array(temp)

    def env_action_to_rs_action(self, action) -> np.array:
        """Convert environment action to Robot Server action"""
        action = self.add_fixed_joints(action)
        rs_action = copy.deepcopy(action)

        # Scale action
        rs_action = np.multiply(rs_action, self.abs_joint_pos_range)
        # Convert action indexing from ur to ros
        rs_action = self.ur._ur_joint_list_to_ros_joint_list(rs_action)

        return action, rs_action        

    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        if type(action) == list: action = np.array(action)
            
        self.elapsed_steps += 1

        # Check if the action is contained in the action space
        if not self.action_space.contains(action):
            raise InvalidActionError()

        # Convert environment action to robot server action
        action, rs_action = self.env_action_to_rs_action(action)

        # Send action to Robot Server and get state
        rs_state = self.client.send_action_get_state(rs_action.tolist()).state_dict

        # Convert the state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        # Assign reward
        reward = 0
        done = False
        reward, done, info = self._reward(rs_state=rs_state, action=action)

        return self.state, reward, done, info

    def render():
        pass
    

    def _get_robot_server_composition(self) -> dict:
        rs_state_keys = dict.fromkeys([
            'object_0_position_x', 
            'object_0_position_y',
            'object_0_position_z',
            'object_0_orientation_x',
            'object_0_orientation_y',
            'object_0_orientation_z',
            'object_0_orientation_w',

            'base_joint_position',
            'shoulder_joint_position',
            'elbow_joint_position',
            'wrist_1_joint_position',
            'wrist_2_joint_position',
            'wrist_3_joint_position',

            'base_joint_velocity',
            'shoulder_joint_velocity',
            'elbow_joint_velocity',
            'wrist_1_joint_velocity',
            'wrist_2_joint_velocity',
            'wrist_3_joint_velocity',

            'ee_to_ref_translation_x',
            'ee_to_ref_translation_y',
            'ee_to_ref_translation_z',
            'ee_to_ref_rotation_x',
            'ee_to_ref_rotation_y',
            'ee_to_ref_rotation_z',
            'ee_to_ref_rotation_w',

            'in_collision'
        ], 0.0)
        return rs_state_keys



    def _get_robot_server_state_len(self) -> int:
        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.
        """
        return len(self._get_robot_server_composition())


    def _set_joint_positions(self, joint_positions) -> None:
        """Set desired robot joint positions with standard indexing."""
        # Set initial robot joint positions
        self.joint_positions = {}
        self.joint_positions['base_joint_position'] = joint_positions[0]
        self.joint_positions['shoulder_joint_position'] = joint_positions[1]
        self.joint_positions['elbow_joint_position'] = joint_positions[2]
        self.joint_positions['wrist_1_joint_position'] = joint_positions[3]
        self.joint_positions['wrist_2_joint_position'] = joint_positions[4]
        self.joint_positions['wrist_3_joint_position'] = joint_positions[5]

    def _get_joint_positions(self) -> dict:
        """Get robot joint positions with standard indexing."""
        return self.joint_positions

    def _get_joint_positions_as_array(self) -> np.array:
        """Get robot joint positions with standard indexing."""
        joint_positions = []
        joint_positions.append(self.joint_positions['base_joint_position'])
        joint_positions.append(self.joint_positions['shoulder_joint_position'])
        joint_positions.append(self.joint_positions['elbow_joint_position'])
        joint_positions.append(self.joint_positions['wrist_1_joint_position'])
        joint_positions.append(self.joint_positions['wrist_2_joint_position'])
        joint_positions.append(self.joint_positions['wrist_3_joint_position'])
        return np.array(joint_positions)

    # TODO: remove from ur base env
    def _get_target_pose(self) -> np.array:
        """Generate target End Effector pose.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """
        return self.ur.get_random_workspace_pose()

    def get_joint_name_order(self) -> list:
        return ['base', 'shoulder', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']

    def _robot_server_state_to_env_state(self, rs_state) -> np.array:
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """
        # Joint positions 
        joint_positions = []
        joint_positions_keys = ['base_joint_position', 'shoulder_joint_position', 'elbow_joint_position',
                            'wrist_1_joint_position', 'wrist_2_joint_position', 'wrist_3_joint_position']
        for position in joint_positions_keys:
            joint_positions.append(rs_state[position])
        joint_positions = np.array(joint_positions)
        # Normalize joint position values
        joint_positions = self.ur.normalize_joint_values(joints=joint_positions)

        # Joint Velocities
        joint_velocities = [] 
        joint_velocities_keys = ['base_joint_velocity', 'shoulder_joint_velocity', 'elbow_joint_velocity',
                            'wrist_1_joint_velocity', 'wrist_2_joint_velocity', 'wrist_3_joint_velocity']
        for velocity in joint_velocities_keys:
            joint_velocities.append(rs_state[velocity])
        joint_velocities = np.array(joint_velocities)

        # Compose environment state
        state = np.concatenate((joint_positions, joint_velocities))

        return state


    def _get_observation_space(self) -> gym.spaces.Box:
        """Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """
        # Joint position range tolerance
        pos_tolerance = np.full(6,0.1)

        # Joint positions range used to determine if there is an error in the sensor readings
        max_joint_positions = np.add(np.full(6, 1.0), pos_tolerance)
        min_joint_positions = np.subtract(np.full(6, -1.0), pos_tolerance)
        # Joint velocities range 
        max_joint_velocities = np.array([np.inf] * 6)
        min_joint_velocities = -np.array([np.inf] * 6)
        # Definition of environment observation_space
        max_obs = np.concatenate((max_joint_positions, max_joint_velocities))
        min_obs = np.concatenate((min_joint_positions, min_joint_velocities))

        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    
    def _get_action_space(self)-> gym.spaces.Box:
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """
        fixed_joints = [self.fix_base, self.fix_shoulder, self.fix_elbow, self.fix_wrist_1, self.fix_wrist_2, self.fix_wrist_3]
        num_control_joints = len(fixed_joints) - sum(fixed_joints)

        return gym.spaces.Box(low=np.full((num_control_joints), -1.0), high=np.full((num_control_joints), 1.0), dtype=np.float32)


# TODO: remove object target
class EmptyEnvironmentURSim(URBaseEnv, Simulation):
    cmd = "roslaunch ur_robot_server ur_robot_server.launch \
        world_name:=tabletop_sphere50.world \
        yaw:=-0.78 \
        reference_frame:=base_link \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=1object \
        n_objects:=1.0 \
        object_0_model_name:=sphere50 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, ur_model='ur5', **kwargs):
        self.cmd = self.cmd + ' ' + 'ur_model:=' + ur_model
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        URBaseEnv.__init__(self, rs_address=self.robot_server_ip, ur_model=ur_model, **kwargs)

class EmptyEnvironmentURRob(URBaseEnv):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving
