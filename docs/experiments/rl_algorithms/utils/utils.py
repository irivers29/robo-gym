import matplotlib.pyplot as plt
import os

class utils():

    def __init__(self, name, plot_folder='tmp/ddpg'):
        self.plot_file = os.path.join(plot_folder, name+'_ddpg_plot.png')

    def reward_graphic(self, reward_history):
        print("... saving reward score ...")
        print(reward_history)
        plt.plot(reward_history)
        plt.ylabel("Reward")
        plt.xlabel("Episode")
        plt.savefig(self.plot_file)