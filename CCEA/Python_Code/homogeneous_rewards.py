import numpy as np
from parameters import Parameters as p
import math



# GLOBAL REWARDS ------------------------------------------------------------------------------------------------------
def calc_global(rover_path, poi_values, poi_positions):
    num_steps = p.num_steps + 1
    inf = 1000.00
    g_reward = 0.0
    boundary = 5.0

    # For all POIs
    poi_status = [False for _ in range(p.num_pois)]
    poi_observed = False
    for poi_id in range(p.num_pois):

        current_poi_reward = 0.0

        # For all timesteps (rover steps)
        for step_number in range(num_steps):
            observer_count = 0  # Track number of observers for given POI
            observer_distances = [0.0 for _ in range(p.num_rovers)]
            summed_distances = 0.0  # Denominator of reward function

            # For all agents
            # Calculate distance between poi and agent
            for rover_id in range(p.num_rovers):
                rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_number, rover_id, 0]
                rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_number, rover_id, 1]
                distance = math.sqrt((rover_x_dist**2) + (rover_y_dist**2))

                if distance <= p.min_distance:
                    distance = p.min_distance  # Clip distance

                x_center = p.x_dim/2; y_center = p.y_dim/2

                wall_penalty = 0
                if (rover_path[step_number, rover_id, 0] < boundary):
                    wall_penalty = wall_penalty-0.5*(abs(rover_path[step_number, rover_id, 0] - 0))
                if(rover_path[step_number, rover_id, 0] > p.x_dim-boundary):
                    wall_penalty = wall_penalty-0.5*(abs(rover_path[step_number, rover_id, 0] - p.x_dim))
                if (rover_path[step_number, rover_id, 1] < boundary):
                    wall_penalty = wall_penalty - 0.5 * (abs(rover_path[step_number, rover_id, 1] - 0))
                if (rover_path[step_number, rover_id, 1] > p.y_dim-boundary):
                    wall_penalty = wall_penalty - 0.5 * (abs(rover_path[step_number, rover_id, 1] - p.y_dim))



                observer_distances[rover_id] = distance
                 # Check if agent observes poi
                if distance <= p.activation_dist: # Rover is in observation range
                    observer_count += 1

            if observer_count >= p.coupling:  # If observers meet coupling req, calculate reward
                for rv in range(p.coupling):
                    summed_distances += min(observer_distances)
                    od_index = observer_distances.index(min(observer_distances))
                    observer_distances[od_index] = inf
                temp_reward = poi_values[poi_id]/summed_distances
                poi_status[poi_id] = True # Ignore POIs that have been already been observed in the previous time steps
                poi_observed = True


            else:
                temp_reward = 0.0

            if temp_reward > current_poi_reward:
                current_poi_reward = temp_reward

            current_poi_reward = current_poi_reward + wall_penalty



            if poi_observed == True:
                break # none of the agents will get any more reward in any future steps now.



        g_reward += current_poi_reward

    return g_reward, poi_status


# DIFFERENCE REWARDS -------------------------------------------------------------------------------------------------
def calc_difference(rover_path, poi_values, poi_positions):
    num_steps = p.num_steps + 1
    inf = 1000.00
    difference_rewards = np.zeros(p.num_rovers * p.num_types)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_global(rover_path, poi_values, poi_positions)

    # CALCULATE DIFFERENCE REWARD
    for rover_id in range(p.num_rovers):
        g_without_self = 0.0

        for poi_id in range(p.num_pois):
            current_poi_reward = 0.0

            for step_number in range(num_steps):
                observer_count = 0  # Track number of POI observers at time step
                observer_distances = [0.0 for _ in range(p.num_rovers)]
                summed_distances = 0.0  # Denominator of reward function

                # Calculate distance between poi and agent
                for other_rover_id in range(p.num_rovers):
                    if rover_id != other_rover_id:  # Only do for other rovers
                        rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_number, other_rover_id, 0]
                        rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_number, other_rover_id, 1]
                        distance = math.sqrt((rover_x_dist**2) + (rover_y_dist**2))

                        if distance <= p.min_distance:
                            distance = p.min_distance

                        observer_distances[other_rover_id] = distance

                        if distance <= p.activation_dist:
                            observer_count += 1

                    if rover_id == other_rover_id:  # Ignore self
                        observer_distances[rover_id] = inf

                if observer_count >= p.coupling:  # If coupling satisfied, compute reward
                    for rv in range(p.coupling):
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
def calc_dpp(rover_path, poi_values, poi_positions):
    num_steps = p.num_steps + 1
    inf = 1000.00
    dplusplus_reward = np.zeros(p.num_rovers * p.num_types)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_global(rover_path, poi_values, poi_positions)

    # CALCULATE DIFFERENCE REWARD
    difference_reward = calc_difference(rover_path, poi_values, poi_positions)

    # CALCULATE DPP REWARD
    for c_count in range(p.coupling-1):

        # Calculate Difference with Extra Me Reward
        for rover_id in range(p.num_rovers):
            g_with_counterfactuals = 0.0

            for poi_id in range(p.num_pois):
                current_poi_reward = 0.0

                for step_number in range(num_steps):
                    observer_count = 0  # Track number of POI observers at time step
                    observer_distances = []
                    summed_distances = 0.0 # Denominator of reward function
                    self_x = poi_positions[poi_id, 0] - rover_path[step_number, rover_id, 0]
                    self_y = poi_positions[poi_id, 1] - rover_path[step_number, rover_id, 1]
                    self_dist = math.sqrt((self_x**2) + (self_y**2))

                    # Calculate distance between poi and agent
                    for other_rover_id in range(p.num_rovers):
                        rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_number, other_rover_id, 0]
                        rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_number, other_rover_id, 1]
                        distance = math.sqrt((rover_x_dist**2) + (rover_y_dist**2))

                        if distance <= p.min_distance:
                            distance = p.min_distance
                        observer_distances.append(distance)

                        # Update observer count
                        if distance <= p.activation_dist:
                            observer_count += 1

                    if self_dist <= p.activation_dist:  # Another me only works if self in range
                        for c in range(c_count):
                            observer_distances.append(self_dist)
                        observer_count += c_count

                    if observer_count >= p.coupling:  # If coupling satisfied, compute reward
                        for rv in range(p.coupling):
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

    for rov_id in range(p.num_rovers):
        if difference_reward[rov_id] > dplusplus_reward[rov_id]:  # Use difference reward, if it is better
            dplusplus_reward[rov_id] = difference_reward[rov_id]

    return dplusplus_reward
