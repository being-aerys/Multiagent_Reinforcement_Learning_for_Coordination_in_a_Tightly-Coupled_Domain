#!/usr/bin/env python3
#  This is code helps to simulate the AE path

# Created by Ashwin Vinoo
# Date: 3/20/2019

# importing all the necessary modules
from rover_domain_visualizer import RoverDomainVisualizer
from matplotlib import pyplot as plot
import pickle
import time

# ---------- hyper parameters ----------
# The pickle files to load
file_to_load = "without_ae_4_rovers_10_POI_path.pkl"
# --------------------------------------

# If this file is the main one called for execution
if __name__ == "__main__":

    # We create a visualization object
    visualizer = RoverDomainVisualizer(4, (30, 30), 4, 2)
    # Reading in the pickle files to load for with ae
    with open(file_to_load, "rb") as file:
        # Loads the with ae rewards list
        primary_list = pickle.load(file)
        # We obtain the number of time steps to simulate
        total_time_steps = len(primary_list[0][0])
        # We iterate through the total number of time steps
        for time_step in range(total_time_steps):
            # We initialize the rover list
            rover_list = []
            # We iterate through the list of rovers
            for i in range(len(primary_list[0])):
                # We append the rover coordinates at that time step to the rover list
                rover_list.append(primary_list[0][i][time_step])
            # We append the poi coordinates at that time step to the poi list
            poi_list = primary_list[1][time_step]
            # We append the poi status at that time step to the poi status list
            poi_status_list = primary_list[2][time_step]
            # We update the visualizer
            visualizer.update_visualizer(rover_list, poi_list, False, 0.2)
            # In case we are in the first iteration
            if time_step == 0:
                # We sleep for a few seconds
                time.sleep(5)
