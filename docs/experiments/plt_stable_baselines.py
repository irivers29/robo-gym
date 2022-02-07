from stable_baselines3 import TD3
from stable_baselines3.td3 import MlpPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor

import matplotlib.pyplot as plt

log_dir='mon/03_02_TD3_I'

plot_results([log_dir], 1000000, results_plotter.X_TIMESTEPS, "TD3")
plt.show()