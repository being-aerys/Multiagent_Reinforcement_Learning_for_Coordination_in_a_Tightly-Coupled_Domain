"""
This file contains all constant test parameters which may be altered from this single
location for convenience.
"""


class Parameters:

    # Run Parameters
    stat_runs = 1
    generations = 500# Number of generations for CCEA in each stat run
    visualizer_on = True  # Turn visualizer on or off (TURN OFF FOR MULTIPLE STAT RUNS)

    # Domain parameters
    #team_types = 'heterogeneous'  # Use 'homogeneous' for uniform rovers, and 'heterogeneous' for non-uniform rovers
    team_types = 'homogeneous'
    reward_type = 0  # 0 for global, 1 for difference, 2 for d++, 3 for s-d++
    num_rovers = 2 # Number of rovers on map (GETS MULTIPLIED BY NUMBER OF TYPES)
    num_types = 1  # How many types of rovers are on the map
    coupling = 2 # Number of rovers required to view a POI for credit
    num_pois = 4  # Number of POIs on map
    num_steps = 100 # Number of steps rovers take each episode
    min_distance = 1.0  # Minimum distance which may appear in the denominator of credit eval functions
    x_dim = 30  # X-Dimension of the rover map
    y_dim = 30  # Y-Dimension of the rover map
    activation_dist = 4.0  # Minimum distance rovers must be to observe POIs
    observation_radius = 20.0
    angle_resolution = 90  # Resolution of sensors (determines number of sectors)
    sensor_model = "closest"  # Should either be density or closest (Use closest for evolutionary domain)

    # Neural network parameters
    num_inputs = int(720/angle_resolution)+4 # for 4 wall information
    num_nodes = 100
    num_outputs = 2

    # CCEA parameters
    mutation_rate = 0.1
    epsilon = 0.1
    pop_size = 50
