import numpy as np
from parameters import Parameters as p
import math
from supervisor import one_of_each_type


# GLOBAL REWARDS ------------------------------------------------------------------------------------------------------
cpdef calc_hetero_global(rover_path, poi_values, poi_positions):
    cdef int npois = int(p.num_pois)
    cdef int nrovers = int(p.num_rovers)
    cdef int ntypes = int(p.num_types)
    cdef int num_steps = int(p.num_steps + 1)
    cdef int coupling = int(p.coupling)
    cdef double min_dist = p.min_distance
    cdef double act_dist = p.activation_dist
    cdef double g_reward = 0.0
    cdef int poi_id, step_id, rtype, rover_id, rv_id, observer_count, t
    cdef double summed_distances, rover_x_dist, rover_y_dist, distance, current_poi_reward, temp_reward
    cdef double[:, :] observer_distances

    # For all POIs
    for poi_id in range(npois):
        current_poi_reward = 0.0  # Tracks highest reward received from a POI across all timesteps

        for step_id in range(num_steps):  # Calculate rewards at each time step
            observer_count = 0  # Count observers at given time step
            observer_distances = np.zeros((ntypes, nrovers))  # Track distances between rovers and POI
            summed_distances = 0.0
            types_in_range = []

            # Calculate distance between poi and agent
            for rtype in range(ntypes):
                for rover_id in range(nrovers):
                    rv_id = int(nrovers*rtype + rover_id)  # Converts identifier to be compatible with base code
                    rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_id, rv_id, 0]
                    rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_id, rv_id, 1]
                    distance = math.sqrt((rover_x_dist**2) + (rover_y_dist**2))

                    if distance < min_dist:  # Clip distance to avoid excessively large rewards
                        distance = min_dist
                    observer_distances[rtype, rover_id] = distance

                    # Check if agent observes poi
                    if distance <= act_dist: # Rover is in observation range
                        types_in_range.append(rtype)

            for t in range(ntypes):  # Assumes coupling is one of each type
                if t in types_in_range:  # If a rover of a given type is in range, count increases
                    observer_count += 1

            if observer_count >= coupling:  # If coupling satisfied, compute reward
                for t in range(coupling):
                    summed_distances += min(observer_distances[t, :])
                temp_reward = poi_values[poi_id]/summed_distances
            else:
                temp_reward = 0.0

            if temp_reward > current_poi_reward:
                current_poi_reward = temp_reward  # Track best observation from given POI

        g_reward += current_poi_reward

    return g_reward


# DIFFERENCE REWARDS -------------------------------------------------------------------------------------------------
cpdef calc_hetero_difference(rover_path, poi_values, poi_positions):
    cdef int npois = int(p.num_pois)
    cdef int nrovers = int(p.num_rovers)
    cdef int ntypes = int(p.num_types)
    cdef int num_steps = int(p.num_steps + 1)
    cdef int coupling = int(p.coupling)
    cdef double min_dist = p.min_distance
    cdef double act_dist = p.activation_dist
    cdef double inf = 1000.0
    cdef int poi_id, step_id, rtype, other_type, other_rov, rv_id, observer_count, t, current_type, current_rov
    cdef double summed_distances, rover_x_dist, rover_y_dist, distance, current_poi_reward, temp_reward
    cdef double g_without_self
    cdef double[:, :] observer_distances
    cdef double g_reward = 0.0
    cdef double[:] difference_reward = np.zeros(nrovers * ntypes)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_hetero_global(rover_path, poi_values, poi_positions)  # Get true global reward

    # CALCULATE DIFFERENCE REWARD
    for current_type in range(ntypes):
        for current_rov in range(nrovers):
            g_without_self = 0.0

            for poi_id in range(npois):
                current_poi_reward = 0.0

                for step_id in range(num_steps):
                    observer_count = 0
                    observer_distances = np.zeros((ntypes, nrovers))  # Track distances between POI and rovers
                    summed_distances = 0.0
                    types_in_range = []

                    # Calculate distance between poi and other agents
                    for other_type in range(ntypes):
                        for other_rov in range(nrovers):
                            rv_id = int(nrovers*other_type + other_rov)  # Convert rover id to AADI base format

                            if current_rov != other_rov or current_type != other_type:
                                rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_id, rv_id, 0]
                                rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_id, rv_id, 1]
                                distance = math.sqrt((rover_x_dist**2) + (rover_y_dist**2))

                                if distance < min_dist:  # Clip distance to avoid excessively large rewards
                                    distance = min_dist
                                observer_distances[other_type, other_rov] = distance

                                if distance <= act_dist:  # Track what rover types are observing
                                    types_in_range.append(other_type)
                            else:
                                observer_distances[current_type, current_rov] = inf  # Ignore self

                    for t in range(ntypes):
                        if t in types_in_range:  # If a rover of a given type is in range, count increases
                            observer_count += 1

                    if observer_count >= coupling:  # If coupling satisfied, compute reward
                        for t in range(coupling):
                            summed_distances += min(observer_distances[t, :])
                        temp_reward = poi_values[poi_id]/summed_distances
                    else:
                        temp_reward = 0.0

                    if temp_reward > current_poi_reward:
                        current_poi_reward = temp_reward  # Track current best reward for given POI

                g_without_self += current_poi_reward

            rv_id = int(nrovers*current_type + current_rov)  # Convert to AADI compatible rover identifier
            difference_reward[rv_id] = g_reward - g_without_self

    return difference_reward


# D++ REWARD ----------------------------------------------------------------------------------------------------------
cpdef calc_hetero_dpp(rover_path, poi_values, poi_positions):
    cdef int npois = int(p.num_pois)
    cdef int nrovers = int(p.num_rovers)
    cdef int ntypes = int(p.num_types)
    cdef int num_steps = int(p.num_steps + 1)
    cdef int coupling = int(p.coupling)
    cdef double min_dist = p.min_distance
    cdef double act_dist = p.activation_dist
    cdef double inf = 1000.0
    cdef int poi_id, step_id, rtype, other_type, other_rov, rv_id, observer_count, t, current_type, current_rov
    cdef int self_id, c_count
    cdef double summed_distances, rover_x_dist, rover_y_dist, distance, current_poi_reward, temp_reward, self_dist
    cdef double g_without_self, g_with_counterfactuals, self_x, self_y
    cdef double g_reward = 0.0
    cdef double[:] dplusplus_reward = np.zeros(nrovers * ntypes)

    # CALCULATE GLOBAL REWARD
    g_reward = calc_hetero_global(rover_path, poi_values, poi_positions)

    # CALCULATE DIFFERENCE REWARD
    dplusplus_reward = calc_hetero_difference(rover_path, poi_values, poi_positions)

    # CALCULATE DPP REWARD
    for c_count in range(coupling):  # Increment counterfactuals up to coupling requirement

        # Calculate Difference with Extra Me Reward
        for rtype in range(ntypes):
            for current_rov in range(nrovers):
                g_with_counterfactuals = 0.0; self_id = int(nrovers*rtype + current_rov)

                for poi_id in range(npois):
                    current_poi_reward = 0.0

                    for step_id in range(num_steps):
                        # Count how many agents observe poi, update closest distance if necessary
                        observer_count = 0; summed_distances = 0.0
                        observer_distances = [[] for _ in range(ntypes)]
                        types_in_range = []
                        self_x = poi_positions[poi_id, 0] - rover_path[step_id, self_id, 0]
                        self_y = poi_positions[poi_id, 1] - rover_path[step_id, self_id, 1]
                        self_dist = math.sqrt((self_x**2) + (self_y**2))

                        # Calculate distance between poi and agent
                        for other_type in range(ntypes):
                            for other_rov in range(nrovers):
                                rv_id = int(nrovers*other_type + other_rov)  # Make rover ID AADI compatible
                                rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_id, rv_id, 0]
                                rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_id, rv_id, 1]
                                distance = math.sqrt((rover_x_dist**2) + (rover_y_dist**2))

                                if distance < min_dist:  # Clip distance to avoid excessively large rewards
                                    distance = min_dist
                                observer_distances[other_type].append(distance)

                                if distance <= act_dist:  # Track rover types currently observing
                                    types_in_range.append(other_type)

                        if self_dist <= act_dist:  # Add counterfactual partners if in range (more of me)
                            for c in range(c_count):
                                observer_distances[rtype].append(self_dist)
                                types_in_range.append(rtype)

                        for t in range(ntypes):
                            if t in types_in_range:  # If a rover of a given type is in range, count increases
                                observer_count += 1

                        if observer_count >= coupling:  # If coupling satisfied, compute reward
                            for t in range(coupling):
                                summed_distances += min(observer_distances[t][:])
                            temp_reward = poi_values[poi_id]/summed_distances
                        else:
                            temp_reward = 0.0

                        if temp_reward > current_poi_reward:  # Track current best from POI
                            current_poi_reward = temp_reward

                    g_with_counterfactuals += current_poi_reward

                temp_dpp_reward = (g_with_counterfactuals - g_reward)/(1 + c_count)
                rv_id = int(nrovers*rtype + current_rov)  # Convert rover identifier to AADI format
                if temp_dpp_reward > dplusplus_reward[rv_id]:
                    dplusplus_reward[rv_id] = temp_dpp_reward


    return dplusplus_reward

# S-D++ REWARD --------------------------------------------------------------------------------------------------------
cpdef calc_sdpp(rover_path, poi_values, poi_positions):
    cdef int npois = int(p.num_pois)
    cdef int nrovers = int(p.num_rovers)
    cdef int ntypes = int(p.num_types)
    cdef int num_steps = int(p.num_steps + 1)
    cdef int coupling = int(p.coupling)
    cdef double min_dist = p.min_distance
    cdef double act_dist = p.activation_dist
    cdef double inf = 1000.0
    cdef int poi_id, step_id, rtype, other_type, other_rov, rv_id, observer_count, t, current_type, current_rov
    cdef int self_id, c_count
    cdef double summed_distances, rover_x_dist, rover_y_dist, distance, current_poi_reward, temp_reward, self_dist
    cdef double g_without_self, g_with_counterfactuals, self_x, self_y
    cdef double g_reward = 0.0
    cdef double[:] s_dplusplus_reward = np.zeros(nrovers * ntypes)
    cdef double[:, :] rov_partners

    # CALCULATE GLOBAL REWARD
    g_reward = calc_hetero_global(rover_path, poi_values, poi_positions)

    # CALCULATE DIFFERENCE REWARD
    s_dplusplus_reward = calc_hetero_difference(rover_path, poi_values, poi_positions)

    # CALCULATE S-DPP REWARD
    for c_count in range(coupling):
        rov_partners = np.zeros((c_count, 2))

        # Calculate reward with suggested counterfacual partners
        for rtype in range(ntypes):
            for current_rov in range(nrovers):
                g_with_counterfactuals = 0.0; self_id = int(nrovers*rtype + current_rov)

                for poi_id in range(npois):
                    current_poi_reward = 0.0

                    for step_id in range(num_steps):
                        # Count how many agents observe poi, update closest distance if necessary
                        observer_count = 0; summed_distances = 0.0
                        observer_distances = [[] for _ in range(ntypes)]
                        types_in_range = []
                        self_x = poi_positions[poi_id, 0] - rover_path[step_id, self_id, 0]
                        self_y = poi_positions[poi_id, 1] - rover_path[step_id, self_id, 1]
                        self_dist = math.sqrt((self_x**2) + (self_y**2))  # Distance between self and POI

                        # Calculate distance between all poi and rovers
                        for other_type in range(ntypes):
                            for other_rov in range(nrovers):
                                rv_id = int(nrovers*other_type + other_rov)
                                rover_x_dist = poi_positions[poi_id, 0] - rover_path[step_id, rv_id, 0]
                                rover_y_dist = poi_positions[poi_id, 1] - rover_path[step_id, rv_id, 1]
                                distance = math.sqrt((rover_x_dist**2) + (rover_y_dist**2))

                                if distance < min_dist:  # Clip distance to avoid excessively large rewards
                                    distance = min_dist
                                observer_distances[other_type].append(distance)

                                if distance <= act_dist:  # Rover type appended to observer list
                                    types_in_range.append(other_type)

                        #  Add counterfactuals
                        if self_dist <= act_dist:  # Don't add partners unless self in range
                            rov_partners = one_of_each_type(c_count, rtype, self_dist)  # Get counterfactual partners

                            for rv in range(c_count):
                                observer_distances[rv].append(rov_partners[rv, 0])
                                observer_count += 1

                        if observer_count >= coupling:  # If coupling satisfied, compute reward
                            for t in range(coupling):
                                summed_distances += min(observer_distances[t][:])
                            temp_reward = poi_values[poi_id]/summed_distances
                        else:
                            temp_reward = 0.0

                        if temp_reward > current_poi_reward:
                            current_poi_reward = temp_reward

                    g_with_counterfactuals += current_poi_reward

                temp_dpp_reward = (g_with_counterfactuals - g_reward)/(1 + c_count)
                rv_id = int(nrovers*rtype + current_rov)
                if temp_dpp_reward > s_dplusplus_reward[rv_id]:
                    s_dplusplus_reward[rv_id] = temp_dpp_reward

    return s_dplusplus_reward
