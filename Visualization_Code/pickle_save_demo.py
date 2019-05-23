#!/usr/bin/env python3
#  This is code simply helps Enna learn saving variables with pickle super fast

# Created by Ashwin Vinoo
# Date: 3/20/2019

# import all the necessary modules
import pickle
import numpy as np

# ---------- hyper parameters ----------
# The pickle files to save for with ae
with_ae_file = "ae_file.pkl"
# The pickle files to save for without ae
without_ae_file = "without_ae_file.pkl"
# The number of episodes to generate data
episode_count = 1000
# --------------------------------------

# If this file is the main one called for execution
if __name__ == "__main__":

    # The list of rewards with AE goes here
    reward_list_1 = [np.sin(5*np.pi*i/episode_count) for i in range(episode_count)]
    # The list of rewards without AE goes here
    reward_list_2 = [np.cos(2*np.pi*i/episode_count) for i in range(episode_count)]

    # Writes the reward list for with AE to the with AE file
    with open(with_ae_file, "wb") as file:
        pickle.dump(reward_list_1, file)
    # Writes the reward list for without AE to the without AE file
    with open(without_ae_file, "wb") as file:
        pickle.dump(reward_list_2, file)
