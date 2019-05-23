import numpy as np
from parameters import Parameters as p
import math


# GLOBAL REWARDS ------------------------------------------------------------------------------------------------------
cpdef calc_global(rover_path, poi_values, poi_positions):
    cdef int nrovers = int(p.num_rovers*p.num_types)
    cdef int npois = int(p.num_pois)
    cdef int coupling = int(p.coupling)
    cdef int poi_id, step_number, rover_id, rv, observer_count, od_index
    cdef double min_dist = p.min_distance
    cdef double act_dist = p.activation_dist
    cdef double rover_x_dist, rover_y_dist, distance, summed_distances, current_poi_reward, temp_reward
    cdef int num_steps = int(p.num_steps + 1)
    cdef double inf = 1000.00
    cdef double g_reward = 0.0

    # For all POIs
    for poi_id in range(p.num_pois):
        current_poi_reward = 0.0

        # For all timesteps (rover steps)
        for step_number in range(num_steps):
            observer_count = 0  # Track number of observers for given POI
            observer_distances = [0.0 for _ in range(nrovers)]
            summed_distances = 0.0  # Denominator of reward function

            # For all agents
            # Calculate distance between poi and agent
            for rover_id in range(nrovers):
                rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_number, rover_id, 0]
                rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_number, rover_id, 1]
                distance = math.sqrt((rover_x_dist**2) + (rover_y_dist**2))

                if distance <= min_dist:
                    distance = min_dist  # Clip distance

                observer_distances[rover_id] = distance

                # Check if agent observes poi
                if distance <= act_dist: # Rover is in observation range
                    observer_count += 1

            if observer_count >= coupling:  # If observers meet coupling req, calculate reward
                for rv in range(coupling):
                    summed_distances += min(observer_distances)
                    od_index = observer_distances.index(min(observer_distances))
                    observer_distances[od_index] = inf
                temp_reward = poi_values[poi_id]/summed_distances
            else:
                temp_reward = 0.0

            if temp_reward > current_poi_reward:
                current_poi_reward = temp_reward

        g_reward += current_poi_reward

    return g_reward


# DIFFERENCE REWARDS -------------------------------------------------------------------------------------------------
cpdef calc_difference(rover_path, poi_values, poi_positions):
    cdef int nrovers = int(p.num_rovers*p.num_types)
    cdef int npois = int(p.num_pois)
    cdef int coupling = int(p.coupling)
    cdef int poi_id, step_number, rover_id, rv, observer_count, od_index, other_rover_id
    cdef double min_dist = p.min_distance
    cdef double act_dist = p.activation_dist
    cdef double rover_x_dist, rover_y_dist, distance, summed_distances, current_poi_reward, temp_reward, g_without_self
    cdef int num_steps = int(p.num_steps + 1)
    cdef double inf = 1000.00
    cdef double g_reward = 0.0
    cdef double[:] difference_rewards = np.zeros(nrovers)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_global(rover_path, poi_values, poi_positions)

    # CALCULATE DIFFERENCE REWARD
    for rover_id in range(nrovers):
        g_without_self = 0.0

        for poi_id in range(npois):
            current_poi_reward = 0.0

            for step_number in range(num_steps):
                observer_count = 0  # Track number of POI observers at time step
                observer_distances = [0.0 for _ in range(nrovers)]
                summed_distances = 0.0  # Denominator of reward function

                # Calculate distance between poi and agent
                for other_rover_id in range(nrovers):
                    if rover_id != other_rover_id:  # Only do for other rovers
                        rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_number, other_rover_id, 0]
                        rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_number, other_rover_id, 1]
                        distance = math.sqrt((rover_x_dist**2) + (rover_y_dist**2))

                        if distance <= min_dist:
                            distance = min_dist

                        observer_distances[other_rover_id] = distance

                        if distance <= act_dist:
                            observer_count += 1

                    if rover_id == other_rover_id:  # Ignore self
                        observer_distances[rover_id] = inf

                if observer_count >= coupling:  # If coupling satisfied, compute reward
                    for rv in range(coupling):
                        summed_distances += min(observer_distances)
                        od_index = observer_distances.index(min(observer_distances))
                        observer_distances[od_index] = inf
                    temp_reward = poi_values[poi_id]/summed_distances
                else:
                    temp_reward = 0.0

                if temp_reward > current_poi_reward:
                    current_poi_reward = temp_reward

            g_without_self += current_poi_reward
        difference_rewards[rover_id] = g_reward - g_without_self

    return difference_rewards


# D++ REWARD ----------------------------------------------------------------------------------------------------------
cpdef calc_dpp(rover_path, poi_values, poi_positions):
    cdef int nrovers = int(p.num_rovers*p.num_types)
    cdef int npois = int(p.num_pois)
    cdef int coupling = int(p.coupling)
    cdef int poi_id, step_number, rover_id, rv, observer_count, od_index, other_rover_id, c
    cdef double min_dist = p.min_distance
    cdef double act_dist = p.activation_dist
    cdef double rover_x_dist, rover_y_dist, distance, summed_distances, current_poi_reward, temp_reward, g_without_self
    cdef double self_x, self_y, self_dist
    cdef int num_steps = int(p.num_steps + 1)
    cdef double inf = 1000.00
    cdef double g_reward = 0.0
    cdef double[:] difference_rewards = np.zeros(nrovers)
    cdef double[:] dplusplus_reward = np.zeros(nrovers)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_global(rover_path, poi_values, poi_positions)

    # CALCULATE DIFFERENCE REWARD
    dplusplus_reward = calc_difference(rover_path, poi_values, poi_positions)

    # CALCULATE DPP REWARD
    for c_count in range(coupling):

        # Calculate Difference with Extra Me Reward
        for rover_id in range(nrovers):
            g_with_counterfactuals = 0.0

            for poi_id in range(npois):
                current_poi_reward = 0.0

                for step_number in range(num_steps):
                    observer_count = 0  # Track number of POI observers at time step
                    observer_distances = []
                    summed_distances = 0.0 # Denominator of reward function
                    self_x = poi_positions[poi_id, 0] - rover_path[step_number, rover_id, 0]
                    self_y = poi_positions[poi_id, 1] - rover_path[step_number, rover_id, 1]
                    self_dist = math.sqrt((self_x**2) + (self_y**2))

                    # Calculate distance between poi and agent
                    for other_rover_id in range(nrovers):
                        rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_number, other_rover_id, 0]
                        rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_number, other_rover_id, 1]
                        distance = math.sqrt((rover_x_dist**2) + (rover_y_dist**2))

                        if distance <= min_dist:
                            distance = min_dist
                        observer_distances.append(distance)

                        # Update observer count
                        if distance <= act_dist:
                            observer_count += 1

                    if self_dist <= act_dist:  # Another me only works if self in range
                        for c in range(c_count):
                            observer_distances.append(self_dist)
                        observer_count += c_count

                    if observer_count >= coupling:  # If coupling satisfied, compute reward
                        for rv in range(coupling):
                            summed_distances += min(observer_distances)
                            od_index = observer_distances.index(min(observer_distances))
                            observer_distances[od_index] = inf
                        temp_reward = poi_values[poi_id]/summed_distances
                    else:
                        temp_reward = 0.0

                    if temp_reward > current_poi_reward:
                        current_poi_reward = temp_reward

                g_with_counterfactuals += current_poi_reward

            temp_dpp_reward = (g_with_counterfactuals - g_reward)/(1 + c_count)
            if temp_dpp_reward > dplusplus_reward[rover_id]:
                dplusplus_reward[rover_id] = temp_dpp_reward

    return dplusplus_reward
