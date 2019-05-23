import numpy as np
from parameters import Parameters as p
import math
from supervisor import one_of_each_type

# GLOBAL REWARDS ------------------------------------------------------------------------------------------------------
cpdef calc_global_reward(rover_history, poi_vals, poi_pos):
    cdef int n_rovers = p.num_rovers  # Number of rovers of each type
    cdef int n_types = p.num_types  # Number of rover types in play
    cdef int n_pois = p.num_pois
    cdef double min_dist = p.min_distance  # Minimum distance used for reward calculation
    cdef int num_steps = p.num_steps + 1
    cdef int coupling = p.coupling  # Coupling requirement for POI observation
    cdef double activation_dist = p.activation_dist  # Distance at which observations of POI can be made
    cdef double[:, :, :] rov_history = rover_history
    cdef double[:] poi_values = poi_vals
    cdef double[:, :] poi_positions = poi_pos
    cdef int poi_id, step_id, agent_id, observer_count, rtype, rov_id, t
    cdef double rover_x_dist, rover_y_dist, distance
    cdef double inf = 1000.00
    cdef double g_reward = 0.0 # Global reward
    cdef double current_poi_reward = 0.0 #Tracks current highest reward from observing a specific POI
    cdef double temp_reward = 0.0
    cdef double summed_distances = 0.0 # Denominator for reward calculation for a given POI

    # For all POIs
    for poi_id in range(n_pois):
        current_poi_reward = 0.0  # Tracks highest reward received from a POI across all timesteps

        for step_id in range(num_steps):  # Calculate rewards at each time step
            observer_count = 0  # Count observers at given time step
            observer_distances = np.zeros((n_types, n_rovers))  # Track distances between rovers and POI
            summed_distances = 0.0
            temp_reward = 0.0
            types_in_range = []

            # Calculate distance between poi and agent
            for rtype in range(n_types):
                for agent_id in range(n_rovers):
                    rov_id = int(n_rovers*rtype + agent_id) # Converts identifier to be compatible with base code
                    rover_x_dist = poi_positions[poi_id, 0] - rov_history[step_id, rov_id, 0]
                    rover_y_dist = poi_positions[poi_id, 1] - rov_history[step_id, rov_id, 1]
                    distance = math.sqrt((rover_x_dist * rover_x_dist) + (rover_y_dist * rover_y_dist))

                    if distance < min_dist:
                        distance = min_dist

                    observer_distances[rtype, agent_id] = distance

                    # Check if agent observes poi
                    if distance <= activation_dist: # Rover is in observation range
                        types_in_range.append(rtype)

            for t in range(n_types):
                if t in types_in_range:  # If a rover of a given type is in range, count increases
                    observer_count += 1

            # Update closest distance only if poi is observed
            if observer_count >= coupling:
                for rv in range(coupling):  # Coupling requirement is one of each type
                    summed_distances += min(observer_distances[rv, :])  # Take distance from closest observer
                temp_reward = poi_values[poi_id]/summed_distances
            else:
                temp_reward = 0.0

            if temp_reward > current_poi_reward:
                current_poi_reward = temp_reward  # Track best observation from given POI

        g_reward += current_poi_reward

    return g_reward


# DIFFERENCE REWARDS -------------------------------------------------------------------------------------------------
cpdef calc_difference_reward(rover_history, poi_vals, poi_pos):
    cdef int n_rovers = p.num_rovers  # Number of rovers for each type
    cdef int n_types = p.num_types  # Number of types of rovers in play
    cdef int n_pois = p.num_pois
    cdef double min_dist = p.min_distance  # Clipped distance for reward calculation
    cdef int num_steps = p.num_steps + 1
    cdef int coupling = p.coupling
    cdef double activation_dist = p.activation_dist
    cdef double[:, :, :] rov_history = rover_history
    cdef double[:] poi_values = poi_vals
    cdef double[:, :] poi_positions = poi_pos
    cdef int poi_id, step_id, agent_id, observer_count, other_agent_id, rtype, other_type, rov_id, t
    cdef double rover_x_dist, rover_y_dist, distance
    cdef double inf = 1000.00
    cdef double g_reward = 0.0
    cdef double g_without_self = 0.0
    cdef double current_poi_reward = 0.0 #Tracks current highest reward from observing a specific POI
    cdef double temp_reward = 0.0
    cdef double summed_distances = 0.0
    cdef double[:] difference_reward = np.zeros(n_rovers*n_types)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_global_reward(rover_history, poi_vals, poi_pos)  # Get true global reward

    # CALCULATE DIFFERENCE REWARD
    for rtype in range(n_types):
        for agent_id in range(n_rovers):
            g_without_self = 0.0

            for poi_id in range(n_pois):
                current_poi_reward = 0.0

                for step_id in range(num_steps):
                    observer_count = 0
                    observer_distances = np.zeros((n_types, n_rovers))  # Track distances between POI and rovers
                    summed_distances = 0.0
                    temp_reward = 0.0
                    types_in_range = []

                    # Calculate distance between poi and other agents
                    for other_type in range(n_types):
                        for other_agent_id in range(n_rovers):
                            rov_id = int(n_rovers*other_type + other_agent_id)  # Convert rover id to AADI base format
                            if agent_id != other_agent_id or rtype != other_type:
                                rover_x_dist = poi_positions[poi_id, 0] - rov_history[step_id, rov_id, 0]
                                rover_y_dist = poi_positions[poi_id, 1] - rov_history[step_id, rov_id, 1]
                                distance = math.sqrt((rover_x_dist * rover_x_dist) + (rover_y_dist * rover_y_dist))
                                if distance < min_dist:
                                    distance = min_dist
                                observer_distances[other_type, other_agent_id] = distance

                                if distance <= activation_dist:  # Track what rover types are observing
                                    types_in_range.append(other_type)
                            else:
                                observer_distances[rtype, agent_id] = inf  # Ignore self

                    for t in range(n_types):
                        if t in types_in_range:
                            observer_count += 1

                    # update closest distance only if poi is observed
                    if observer_count >= coupling:
                        for rv in range(coupling):  # Coupling requirement is one of each type
                            summed_distances += min(observer_distances[rv, :])
                        temp_reward = poi_values[poi_id]/summed_distances
                    else:
                        temp_reward = 0.0

                    if temp_reward > current_poi_reward:
                        current_poi_reward = temp_reward  # Track current best reward for given POI

                g_without_self += current_poi_reward

            rov_id = int(n_rovers*rtype + agent_id)  # Convert to AADI compatible rover identifier
            difference_reward[rov_id] = g_reward - g_without_self

    return difference_reward


# D++ REWARD ----------------------------------------------------------------------------------------------------------
cpdef calc_dpp_reward(rover_history, poi_vals, poi_pos):
    cdef int n_rovers = p.num_rovers  # Number of rovers of each type
    cdef int n_types = p.num_types  # Number of rover types in play
    cdef int n_pois = p.num_pois
    cdef int current_rtype = 0
    cdef double min_dist = p.min_distance
    cdef int num_steps = p.num_steps + 1
    cdef int coupling = p.coupling
    cdef double activation_dist = p.activation_dist
    cdef double[:, :, :] rov_history = rover_history
    cdef double[:] poi_values = poi_vals
    cdef double[:, :] poi_positions = poi_pos
    cdef int poi_id, step_id, agent_id, observer_count, other_agent_id, c_count, id, other_type, rtype, rov_id, t, c
    cdef double rover_x_dist, rover_y_dist, distance
    cdef double inf = 1000.00
    cdef double g_reward = 0.0
    cdef double g_without_self = 0.0
    cdef double g_with_counterfactuals = 0.0 # Reward with n counterfactual partners added
    cdef double current_poi_reward = 0.0 #Tracks current highest reward from observing a specific POI
    cdef double temp_reward = 0.0
    cdef double temp_dpp_reward = 0.0
    cdef double summed_distances = 0.0
    cdef double[:] dplusplus_reward = np.zeros(n_rovers * n_types)
    cdef double[:] difference_reward = np.zeros(n_rovers * n_types)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_global_reward(rover_history, poi_vals, poi_pos)

    # CALCULATE DIFFERENCE REWARD
    difference_reward = calc_difference_reward(rover_history, poi_vals, poi_pos)

    # CALCULATE DPP REWARD
    for c_count in range(coupling):  # Increment counterfactuals up to coupling requirement

        # Calculate Difference with Extra Me Reward
        for rtype in range(n_types):
            for agent_id in range(n_rovers):
                g_with_counterfactuals = 0.0
                self_dist = 0.0

                for poi_id in range(n_pois):
                    current_poi_reward = 0.0

                    for step_id in range(num_steps):
                        # Count how many agents observe poi, update closest distance if necessary
                        observer_count = 0
                        observer_distances = [[] for _ in range(n_types)]
                        summed_distances = 0.0
                        temp_reward = 0.0
                        types_in_range = []

                        # Calculate distance between poi and agent
                        for other_type in range(n_types):
                            for other_agent_id in range(n_rovers):
                                rov_id = int(n_rovers*other_type + other_agent_id)  # Make rover ID AADI compatible
                                rover_x_dist = poi_positions[poi_id, 0] - rov_history[step_id, rov_id, 0]
                                rover_y_dist = poi_positions[poi_id, 1] - rov_history[step_id, rov_id, 1]
                                distance = math.sqrt((rover_x_dist * rover_x_dist) + (rover_y_dist * rover_y_dist))
                                if distance < min_dist:
                                    distance = min_dist
                                observer_distances[other_type].append(distance)

                                if other_agent_id == agent_id and other_type == rtype:
                                    self_dist = distance  # Track distance from self for counterfactuals

                                if distance <= activation_dist:  # Track rover types currently observing
                                    types_in_range.append(other_type)

                        if self_dist <= activation_dist:  # Add counterfactual partners if in range (more of me)
                            for c in range(c_count):
                                observer_distances[rtype].append(self_dist)
                                types_in_range.append(rtype)

                        for t in range(n_types):
                            if t in types_in_range:
                                observer_count += 1

                        # update closest distance only if poi is observed
                        if observer_count >= coupling:
                            for rv in range(coupling):  # Coupling is one of each type
                                summed_distances += min(observer_distances[rv][:])
                            temp_reward = poi_values[poi_id]/summed_distances
                        else:
                            temp_reward = 0.0

                        if temp_reward > current_poi_reward:  # Track current best from POI
                            current_poi_reward = temp_reward

                    g_with_counterfactuals += current_poi_reward

                temp_dpp_reward = (g_with_counterfactuals - g_reward)/(1 + c_count)
                rov_id = int(n_rovers*rtype + agent_id)  # Convert rover identifier to AADI format
                if temp_dpp_reward > dplusplus_reward[rov_id]:
                    dplusplus_reward[rov_id] = temp_dpp_reward

    for rov_id in range(n_rovers*n_types):
        if difference_reward[rov_id] > dplusplus_reward[rov_id]:
            dplusplus_reward[rov_id] = difference_reward[rov_id]

    return dplusplus_reward

# S-D++ REWARD --------------------------------------------------------------------------------------------------------
cpdef calc_sdpp_reward(rover_history, poi_vals, poi_pos):
    cdef int n_rovers = p.num_rovers  # Number of rovers of each type
    cdef int n_types = p.num_types  # Number of rover types in play
    cdef int n_pois = p.num_pois
    cdef double min_dist = p.min_distance
    cdef int num_steps = p.num_steps + 1
    cdef int coupling = p.coupling
    cdef double activation_dist = p.activation_dist
    cdef double[:, :, :] rov_history = rover_history
    cdef double[:] poi_values = poi_vals
    cdef double[:, :] poi_positions = poi_pos
    cdef int poi_id, step_id, agent_id, observer_count, other_agent_id, c_count, rov_id, rtype, other_type, rv
    cdef double rover_x_dist, rover_y_dist, distance
    cdef double inf = 1000.00
    cdef double g_reward = 0.0
    cdef double g_without_self = 0.0
    cdef double g_with_counterfactuals = 0.0 # Reward with n counterfactual partners added
    cdef double current_poi_reward = 0.0 #Tracks current highest reward from observing a specific POI
    cdef double temp_reward = 0.0
    cdef double temp_dpp_reward = 0.0
    cdef double summed_distances = 0.0
    cdef double[:] difference_reward = np.zeros(n_rovers*n_types)
    cdef double[:] dplusplus_reward = np.zeros(n_rovers*n_types)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_global_reward(rover_history, poi_vals, poi_pos)

    # CALCULATE DIFFERENCE REWARD
    difference_reward = calc_difference_reward(rover_history, poi_vals, poi_pos)

    # CALCULATE S-DPP REWARD
    for c_count in range(coupling):

        # Calculate reward with suggested counterfacual partners
        for rtype in range(n_types):
            for agent_id in range(n_rovers):
                g_with_counterfactuals = 0.0
                self_dist = 0.0

                for poi_id in range(n_pois):
                    current_poi_reward = 0.0

                    for step_id in range(num_steps):
                        # Count how many agents observe poi, update closest distance if necessary
                        observer_count = 0
                        observer_distances = [[] for _ in range(n_types)]
                        summed_distances = 0.0
                        temp_reward = 0.0
                        types_in_range = []

                        # Calculate distance between poi and agent
                        for other_type in range(n_types):
                            for other_agent_id in range(n_rovers):
                                rov_id = int(n_rovers*other_type + other_agent_id)
                                rover_x_dist = poi_positions[poi_id, 0] - rov_history[step_id, rov_id, 0]
                                rover_y_dist = poi_positions[poi_id, 1] - rov_history[step_id, rov_id, 1]
                                distance = math.sqrt((rover_x_dist * rover_x_dist) + (rover_y_dist * rover_y_dist))
                                if distance < min_dist:
                                    distance = min_dist
                                observer_distances[other_type].append(distance)

                                if other_agent_id == agent_id and rtype == other_type:
                                    self_dist = distance # Track distance from self for counterfactuals

                                # Check if agent observes poi, update closest step distance
                                if distance <= activation_dist:
                                    types_in_range.append(other_type)

                        #  Add counterfactuals
                        rov_partners = one_of_each_type(c_count)

                        for rv in range(c_count):
                            observer_distances[rv].append(rov_partners[rv, 0])
                            if rov_partners[rv, 1] != rtype:
                                observer_count += 1


                        # update closest distance only if poi is observed
                        if observer_count >= coupling:
                            for rv in range(coupling):  # Coupling is one of each type
                                summed_distances += min(observer_distances[rv][:])
                            temp_reward = poi_values[poi_id]/summed_distances
                        else:
                            temp_reward = 0.0

                        if temp_reward > current_poi_reward:
                            current_poi_reward = temp_reward

                    g_with_counterfactuals += current_poi_reward

                temp_dpp_reward = (g_with_counterfactuals - g_reward)/(1 + c_count)
                rov_id = int(n_rovers*rtype + agent_id)
                if temp_dpp_reward > dplusplus_reward[rov_id]:
                    dplusplus_reward[rov_id] = temp_dpp_reward

    for rov_id in range(n_rovers*n_types):
        if difference_reward[rov_id] > dplusplus_reward[rov_id]:
            dplusplus_reward[rov_id] = difference_reward[rov_id]

    return dplusplus_reward
