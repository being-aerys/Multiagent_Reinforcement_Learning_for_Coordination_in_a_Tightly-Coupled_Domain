"""
This file contains all constant test parameters which may be altered from this single
location for convenience.
"""
import argparse, os


class Parameters:

    # Run Parameters
    algo = 'DDPG'  # DDPG | NAF
    gamma = 0.99
    tau = 0.001
    noise_scale = 0.5
    final_noise_scale = 0.3
    seed = 4
    num_hnodes = num_mem = 512

    is_dpp = True
    batch_size = 156
    autoencoder_output_length = 10 #todo: AE output length

    ##### episodes #####
    num_episodes = 2000
    updates_per_step = 1
    render = 'False'
    unit_test = 0 #0: None
                  #1: Single Agent
                  #2: Multiagent 2-coupled

    exploration_end = 0.9*num_episodes  # Num of episodes with noise



    visualization = True
    stat_runs = 1
    visualizer_on = True  # Turn visualizer on or off (TURN OFF FOR MULTIPLE STAT RUNS)

    # Domain parameters
    #team_types = 'heterogeneous'  # Use 'homogeneous' for uniform rovers, and 'heterogeneous' for non-uniform rovers
    poi_rand = True # True for random initialization of rovers and POIs
    team_types = 'homogeneous'
    reward_type = 0  # 0 for global, 1 for difference, 2 for d++, 3 for s-d++
    num_rovers = 1  # Number of rovers on map (GETS MULTIPLIED BY NUMBER OF TYPES)
    num_types = 1  # How many types of rovers are on the map
    coupling = 1  # Number of rovers required to view a POI for credit
    num_pois = 1 # Number of POIs on map
    num_timesteps = 50 # Number of steps rovers take each episode
    min_distance = 1.0  # Minimum distance which may appear in the denominator of credit eval functions
    dim_x = 30  # X-Dimension of the rover map
    dim_y = 30  # Y-Dimension of the rover map
    activation_dist = 4.0  # Minimum distance rovers must be to observe POIs
    observation_radius = 20.0
    angle_resolution = 20  # Resolution of sensors (determines number of sectors)
    rover_speed = 1
    is_homogeneous = True  # False --> Heterogenenous Actors
    sensor_model = 2 #2: Closest Sensor
    sensor_model = "closest"  # Should either be density or closest (Use closest for evolutionary domain)

    state_dim = 2 * 360 / angle_resolution + 4

    action_dim = 2
    test_frequency = 10

    #Replay Buffer
    buffer_size = 10000000

    save_foldername = 'R_D++/'
