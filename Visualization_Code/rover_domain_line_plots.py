#!/usr/bin/env python3
#  This is code simply helps Enna plot the AE vs without AE line plots

# Created by Ashwin Vinoo
# Date: 3/20/2019

# import all the necessary modules
from matplotlib import pyplot as plot
import numpy as np
import pickle

# ---------- hyper parameters ----------
# The pickle files to load for with ae
with_ae_file = "with_ae_4_rovers_10_POI_rewards.pkl"
# The pickle files to load for without ae
without_ae_file = "without_ae_4_rovers_10_POI_rewards.pkl"
# This variable controls whether we can plot both
plot_both = False
# In case we only want to plot one set of rewards which file do we choose
single_file = with_ae_file
# --------------------------------------

# If this file is the main one called for execution
if __name__ == "__main__":

    # We want to have two plots side by side
    if plot_both:
        # Reading in the pickle files to load for with ae
        with open(with_ae_file, "rb") as file_1:
            # Loads the with ae rewards list
            with_ae_reward_list = pickle.load(file_1)
            # Reading in the pickle files to load for without ae
            with open(without_ae_file, "rb") as file_2:
                # Loads the without ae rewards list
                without_ae_reward_list = pickle.load(file_2)

                # The length of the x-axis we have to plot
                x_length = min(len(with_ae_reward_list), len(without_ae_reward_list))
                # The episode list
                episode_list = [_ for _ in range(x_length)]
                # plots the two rewards over episodes
                plot.plot(episode_list, with_ae_reward_list, 'r',
                          episode_list, without_ae_reward_list, 'b')
                # Specifying the plot title
                plot.title('Rewards obtained across training episodes')
                # Specifying the label for the x-axis
                plot.xlabel('Episodes')
                # Specifying the label for the y-axis
                plot.ylabel('Global Reward Obtained')
                # Marking the plot legends
                plot.gca().legend(('with AE', 'without AE'))
                # Display the plot
                plot.show()
    else:
            # Reading in the pickle files to load for without ae
            with open(single_file, "rb") as file:
                # Loads the without ae rewards list
                reward_list = pickle.load(file)
                # The length of the x-axis we have to plot
                x_length = len(reward_list)
                # The episode list
                episode_list = [_ for _ in range(x_length)]
                # Fit a numpy polynomial line
                z = np.polyfit(episode_list, reward_list, 9)
                trend_line = np.poly1d(z)
                # plots the two rewards over episodes
                plot.plot(episode_list, reward_list, color=[1, 0, 0])
                plot.plot(episode_list, trend_line(episode_list), color=[0.5, 0, 0], linewidth=3)
                # Marking the plot legends
                plot.gca().legend(('Actual Rewards', 'Reward Trend Line'))
                # Specifying the plot title
                plot.title('Rewards obtained across training episodes')
                # Specifying the label for the x-axis
                plot.xlabel('Episodes')
                # Specifying the label for the y-axis
                plot.ylabel('Global Reward Obtained')
                # Display the plot
                plot.show()
