from gym.envs.registration import register
from importlib_metadata import entry_points

# naming convention: EnvnameRobotSim

# Example Environments
register(
    id='ExampleEnvSim-v0',
    entry_point='robo_gym.envs:ExampleEnvSim',
)

register(
    id='ExampleEnvRob-v0',
    entry_point='robo_gym.envs:ExampleEnvRob',
)

# MiR100 Environments
register(
    id='NoObstacleNavigationMir100Sim-v0',
    entry_point='robo_gym.envs:NoObstacleNavigationMir100Sim',
)

register(
    id='NoObstacleNavigationMir100Rob-v0',
    entry_point='robo_gym.envs:NoObstacleNavigationMir100Rob',
)

register(
    id='ObstacleAvoidanceMir100Sim-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceMir100Sim',
)

register(
    id='ObstacleAvoidanceMir100Rob-v0',
    entry_point='robo_gym.envs:ObstacleAvoidanceMir100Rob',
)

# UR Environments
register(
    id='EmptyEnvironmentURSim-v0',
    entry_point='robo_gym.envs:EmptyEnvironmentURSim',
)

register(
    id='EmptyEnvironmentURRob-v0',
    entry_point='robo_gym.envs:EmptyEnvironmentURRob',
)

register(
    id='ShelfEnvironmentURSim-v0',
    entry_point='robo_gym.envs:ShelfEnvironmentURSim',
)

register(
    id='ShelfEnvironmentURRob-v0',
    entry_point='robo_gym.envs:ShelfEnvironmentURRob',
)

register(
    id='ShelfEnvironmentPositioningURSim-v0',
    entry_point='robo_gym.envs:URShelfPositioningSim'
)

register(
    id='ShelfEnvironmentPositioningURRob-v0',
    entry_point='robo_gym.envs:URShelfPositioningRob'
)

register(
    id='ShelfEnvironmentImPositioninsURSim-v0',
    entry_point='robo_gym.envs:URShelfPosImSim'
)

register(
    id='ShelfEnvironmentImPositioninsURRob-v0',
    entry_point='robo_gym.envs:URShelfPosImRob'
)

register(
    id='ShelfEnvironmentImMIPositioninsURSim-v0',
    entry_point='robo_gym.envs:URShelfPosImMISim'
)

register(
    id='ShelfEnvironmentImMIPositioninsURRob-v0',
    entry_point='robo_gym.envs:URShelfPosImMIRob'
)

register(
    id='ShelfEnvironmentDQNSim-v0',
    entry_point='robo_gym.envs:URShelfMiDQNSim'
)

register(
    id='ShelfEnvironmentDQNRob-v0',
    entry_point='robo_gym.envs:URShelfMiDQNRob'
)

register(
    id='EndEffectorPositioningURSim-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningURSim',
)

register(
    id='EndEffectorPositioningURRob-v0',
    entry_point='robo_gym.envs:EndEffectorPositioningURRob',
)

register(
    id='BasicAvoidanceURSim-v0',
    entry_point='robo_gym.envs:BasicAvoidanceURSim',
)

register(
    id='BasicAvoidanceURRob-v0',
    entry_point='robo_gym.envs:BasicAvoidanceURRob',
)

register(
    id='AvoidanceIros2021URSim-v0',
    entry_point='robo_gym.envs:AvoidanceIros2021URSim',
)

register(
    id='AvoidanceIros2021URRob-v0',
    entry_point='robo_gym.envs:AvoidanceIros2021URRob',
)

register(
    id='AvoidanceIros2021TestURSim-v0',
    entry_point='robo_gym.envs:AvoidanceIros2021TestURSim',
)

register(
    id='AvoidanceIros2021TestURRob-v0',
    entry_point='robo_gym.envs:AvoidanceIros2021TestURRob',
)




