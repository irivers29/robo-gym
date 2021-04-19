import gym
import robo_gym
from robo_gym.utils import ur_utils
import math
import numpy as np 

import pytest

ur_models = [pytest.param('ur3', marks=pytest.mark.skip(reason='not implemented yet')), \
             pytest.param('ur3e', marks=pytest.mark.skip(reason='not implemented yet')), \
             pytest.param('ur5', marks=pytest.mark.commit), \
             pytest.param('ur5e', marks=pytest.mark.skip(reason='not implemented yet')), \
             pytest.param('ur10', marks=pytest.mark.skip(reason='not implemented yet')), \
             pytest.param('ur10e', marks=pytest.mark.skip(reason='not implemented yet')), \
             pytest.param('ur16e', marks=pytest.mark.skip(reason='not implemented yet')), \
]

@pytest.fixture(autouse=True, scope='module', params=ur_models)
def env(request):
    env = gym.make('MovingBoxTargetURSim-v0', ip='robot-servers', ur_model=request.param)
    yield env
    env.kill_sim()
    env.close()

@pytest.mark.commit 
def test_initialization(env):
    env.reset()
    done = False
    for _ in range(10):
        if not done:
            action = env.action_space.sample()
            observation, _, done, _ = env.step(action)

    assert env.observation_space.contains(observation)

@pytest.mark.skip(reason="This fails only in CI")
@pytest.mark.flaky(reruns=3)
def test_object_collision(env):
   params = {
       'ur5': {'joint_positions': [0.0, -1.57, 1.57, -1.57, 0.0, 0.0], 'object_coords':[0.2, -0.1, 0.52], 'action':[-1,0,0,0,0], 'n_steps': 30},
   }
   
   env.reset(desired_joint_positions=params[env.ur.model]['joint_positions'], fixed_object_position=params[env.ur.model]['object_coords'])
   done = False
   i = 0
   while (not done) or i<=params[env.ur.model]['n_steps'] :
       _, _, done, info = env.step(params[env.ur.model]['action'])
       i += 1
   assert info['final_status'] == 'collision'
  
@pytest.mark.nightly    
def test_reset_joint_positions(env):
   joint_positions =  [0.5, -2.7, 1.3, -1.7, -1.9, 1.6]

   state = env.reset(joint_positions=joint_positions)
   assert np.isclose(env.ur.normalize_joint_values(joint_positions), state[3:9], atol=0.1).all()

@pytest.mark.commit 
def test_object_coordinates(env):

   params = {
   #? robot up-right, target_coord_in_ee_frame 0.0, -0.3, 0.2, coordinates of target calculated using official dimensions from DH parameters. 
   #? first value is d4+d6
   #? second value is: d1+a2+a3+d5
   'ur5': {'joint_positions':[0.0, -1.57, 0.0, -1.57, 0.0, 0.0], 'object_coords':[0.0, (0.191 +0.2), (1.001 + 0.3), 0.0, 0.0, 0.0], 'polar_coords':{'r': 0.360, 'theta': 0.983, 'phi': -1.571}}  
   # ('EndEffectorPositioningUR10Sim-v0', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], [0.0, (0.256 +0.2), (1.428 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571},'ur10')
   # ('ur3_env', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], [0.0, (0.194 +0.2), (0.692 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571},'ur3')
   # ('ur3e_env', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], [0.0, (0.223 +0.2), (0.694 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571},'ur3e')
   # ('ur5e_env', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], [0.0, (0.233 +0.2), (1.079 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571},'ur5e')
   # ('ur10e_env', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], [0.0, (0.291 +0.2), (1.485 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571},'ur10e')
   # ('ur16e_env', [0.0, -1.57, 0.0, -1.57, 0.0, 0.0], [0.0, (0.291 +0.2), (1.139 + 0.3), 0.0, 0.0, 0.0], {'r': 0.360, 'theta': 0.983, 'phi': -1.571},'ur16e')
   }

   state = env.reset(joint_positions=params[env.ur.model]['joint_positions'], fixed_object_position=params[env.ur.model]['object_coords'])
   assert np.isclose([params[env.ur.model]['polar_coords']['r'], params[env.ur.model]['polar_coords']['phi'], params[env.ur.model]['polar_coords']['theta']], state[0:3], atol=0.1).all()
   

test_ur_fixed_joints = [
    ('MovingBoxTargetURSim-v0', False, False, False, False, False, True, 'ur5'), # fixed wrist_3
    ('MovingBoxTargetURSim-v0', True, False, True, False, False, False, 'ur5'), # fixed Base and Elbow
    ('MovingBoxTargetURSim-v0', False, False, False, False, False, True, 'ur10'), # fixed wrist_3
    ('MovingBoxTargetURSim-v0', True, False, True, False, False, False, 'ur10'), # fixed Base and Elbow
]

@pytest.mark.nightly
@pytest.mark.parametrize('env_name, fix_base, fix_shoulder, fix_elbow, fix_wrist_1, fix_wrist_2, fix_wrist_3, ur_model', test_ur_fixed_joints)
@pytest.mark.flaky(reruns=3)
def test_fixed_joints(env_name, fix_base, fix_shoulder, fix_elbow, fix_wrist_1, fix_wrist_2, fix_wrist_3, ur_model):
    env = gym.make(env_name, ip='robot-servers', fix_base=fix_base, fix_shoulder=fix_shoulder, fix_elbow=fix_elbow, 
                                                fix_wrist_1=fix_wrist_1, fix_wrist_2=fix_wrist_2, fix_wrist_3=fix_wrist_3, ur_model=ur_model)
    state = env.reset()
    
    initial_joint_positions = [0.0]*6
    initial_joint_positions = state[3:9]
    
    # Take 20 actions
    action = env.action_space.sample()
    for _ in range(20):
        state, _, _, _ = env.step(action)
    
    joint_positions = [0.0]*6
    joint_positions = state[3:9]

    if fix_base:
        assert math.isclose(initial_joint_positions[0], joint_positions[0], abs_tol=0.05)
    if fix_shoulder:
        assert math.isclose(initial_joint_positions[1], joint_positions[1], abs_tol=0.05)
    if fix_elbow:
        assert math.isclose(initial_joint_positions[2], joint_positions[2], abs_tol=0.05)
    if fix_wrist_1:
        assert math.isclose(initial_joint_positions[3], joint_positions[3], abs_tol=0.05)
    if fix_wrist_2:
        assert math.isclose(initial_joint_positions[4], joint_positions[4], abs_tol=0.05)
    if fix_wrist_3:
        assert math.isclose(initial_joint_positions[5], joint_positions[5], abs_tol=0.05)

    env.kill_sim()
    env.close()