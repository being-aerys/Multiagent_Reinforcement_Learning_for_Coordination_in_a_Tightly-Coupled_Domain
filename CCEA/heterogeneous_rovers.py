from parameters import Parameters as p
import numpy as np
import random


def init_rover_positions_fixed():  # Set rovers to fixed starting position
    nrovers = p.num_rovers * p.num_types  # Total number of rovers (of all types)
    rover_positions = np.zeros((nrovers, 3))

    for t in range(p.num_types):
        for rov_id in range(p.num_rovers):
            r_id = (p.num_rovers*t + rov_id)
            rover_positions[r_id, 0] = 0.5*p.x_dim  # Rover X-Coordinate
            rover_positions[r_id, 1] = 0.5*p.y_dim  # Rover Y-Coordinate
            rover_positions[r_id, 2] = t  # Rover type

    return rover_positions

def init_rover_positions_random():  # Randomly set rovers on map
    nrovers = p.num_rovers * p.num_types  # Total number of rovers (of all types)
    rover_positions = np.zeros((nrovers, 3))

    for t in range(p.num_types):
        for rov_id in range(p.num_rovers):
            r_id = (p.num_rovers * t + rov_id)
            rover_positions[r_id, 0] = random.uniform(0, p.x_dim-1)  # Rover X-Coordinate
            rover_positions[r_id, 1] = random.uniform(0, p.y_dim-1)  # Rover Y-Coordinate
            rover_positions[r_id, 2] = t  # Rover type


def init_poi_positions_random():  # Randomly set POI on the map
    poi_positions = np.zeros((p.num_pois, 2))

    for poi_id in range(p.num_pois):
        poi_positions[poi_id, 0] = random.uniform(0, p.x_dim-1)
        poi_positions[poi_id, 1] = random.uniform(0, p.y_dim-1)

    return poi_positions



def init_poi_positions_four_corners():  # Statically set 4 POI (one in each corner)
    assert(p.num_pois == 4)  # There must only be 4 POI for this initialization

    poi_positions = np.zeros((p.num_pois, 2))

    poi_positions[0, 0] = 0.0; poi_positions[0, 1] = 0.0  # Bottom left
    poi_positions[1, 0] = 0.0; poi_positions[1, 1] = (p.y_dim - 1.0)  # Top left
    poi_positions[2, 0] = (p.x_dim - 1.0); poi_positions[2, 1] = 0.0  # Bottom right
    poi_positions[3, 0] = (p.x_dim - 1.0); poi_positions[3, 1] = (p.y_dim - 1.0)  # Top right

    return poi_positions


def init_poi_values_random():  # POI values randomly assigned 1-10
    poi_vals = [0.0 for _ in range(p.num_pois)]

    for poi_id in range(p.num_pois):
        poi_vals[poi_id] = random.randint(1, 10)

    return poi_vals


def init_poi_values_fixed():  # POI values set to fixed value
    poi_vals = [1.0 for _ in range(p.num_pois)]

    for poi_id in range(p.num_pois):
        poi_vals[poi_id] = poi_vals[poi_id] * 5

    return poi_vals
