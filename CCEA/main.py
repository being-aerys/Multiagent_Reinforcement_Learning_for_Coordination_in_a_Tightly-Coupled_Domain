from Python_Code import ccea
from Python_Code import neural_network
from parameters import Parameters as p
from rover_domain_python import RoverDomain
from Python_Code import homogeneous_rewards as homr
#from Python_Code import heterogeneous_rewards as hetr
import csv; import os; import sys
from visualizer import visualize
import numpy


def save_reward_history(reward_history, file_name):
    dir_name = 'Output_Data/'  # Intended directory for output files
    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)


def save_world_configuration(rover_positions, poi_positions, poi_vals):
    dir_name = 'Output_Data/'  # Intended directory for output files
    nrovers = p.num_rovers * p.num_types

    if not os.path.exists(dir_name):  # If Data directory does not exist, create it
        os.makedirs(dir_name)

    rcoords_name = os.path.join(dir_name, 'Rover_Positions.txt')
    pcoords_name = os.path.join(dir_name, 'POI_Positions.txt')
    pvals_name = os.path.join(dir_name, 'POI_Values.txt')

    rov_coords = open(rcoords_name, 'a')
    for r_id in range(nrovers):  # Record initial rover positions to txt file
        rov_coords.write('%f' % rover_positions[r_id, 0])
        rov_coords.write('\t')
        rov_coords.write('%f' % rover_positions[r_id, 1])
        rov_coords.write('\t')
    rov_coords.write('\n')
    rov_coords.close()

    poi_coords = open(pcoords_name, 'a')
    poi_values = open(pvals_name, 'a')
    for p_id in range(p.num_pois):  # Record POI positions and values
        poi_coords.write('%f' % poi_positions[p_id, 0])
        poi_coords.write('\t')
        poi_coords.write('%f' % poi_positions[p_id, 1])
        poi_coords.write('\t')
        poi_values.write('%f' % poi_vals[p_id])
        poi_values.write('\t')
    poi_coords.write('\n')
    poi_values.write('\n')
    poi_coords.close()
    poi_values.close()


def save_rover_path(rover_path):  # Save path rovers take using best policy found
    dir_name = 'Output_Data/'  # Intended directory for output files
    nrovers = p.num_rovers * p.num_types

    rpath_name = os.path.join(dir_name, 'Rover_Paths.txt')

    rpath = open(rpath_name, 'a')
    for rov_id in range(nrovers):
        for t in range(p.num_steps+1):
            rpath.write('%f' % rover_path[t, rov_id, 0])
            rpath.write('\t')
            rpath.write('%f' % rover_path[t, rov_id, 1])
            rpath.write('\t')
        rpath.write('\n')
    rpath.write('\n')
    rpath.close()


''''

# HETEROGENEOUS ROVER TEAMS -------------------------------------------------------------------------------------------
def run_heterogeneous_rovers():
    cc = ccea.Ccea()
    nn = neural_network.NeuralNetwork()
    rd = RoverDomain()

    rtype = p.reward_type

    for srun in range(p.stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)
        reward_history = []

        # Reset CCEA, NN, and world for new stat run
        cc.reset_populations()  # Randomly initialize ccea populations
        nn.reset_nn()  # Initialize NN architecture
        rd.reset()  # Re-initialize world

        save_world_configuration(rd.rover_initial_pos, rd.poi_pos, rd.poi_value)

        for gen in range(p.generations):
            # print("Gen: %i" % gen)
            cc.select_policy_teams()  # Selects which policies will be grouped into which teams
            for team_number in range(cc.population_size):  # Each policy in CCEA is tested in teams
                rd.reset_to_init()  # Resets rovers to initial configuration
                done = False; rd.istep = 0
                joint_state = rd.get_joint_state()

                while not done:
                    for rover_id in range(rd.num_agents):
                        policy_id = int(cc.team_selection[rover_id, team_number])  # Select policy from CCEA pop
                        nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, policy_id], rover_id)
                    joint_state, done = rd.step(nn.out_layer)

                # Update fitness of policies using reward information
                if rtype == 0:
                    reward = hetr.calc_hetero_global(rd.rover_path, rd.poi_value, rd.poi_pos)
                    for pop_id in range(rd.num_agents):
                        policy_id = int(cc.team_selection[pop_id, team_number])
                        cc.fitness[pop_id, policy_id] = reward
                elif rtype == 1:
                    reward = hetr.calc_hetero_difference(rd.rover_path, rd.poi_value, rd.poi_pos)
                    for pop_id in range(rd.num_agents):
                        policy_id = int(cc.team_selection[pop_id, team_number])
                        cc.fitness[pop_id, policy_id] = reward[pop_id]
                elif rtype == 2:
                    reward = hetr.calc_hetero_dpp(rd.rover_path, rd.poi_value, rd.poi_pos)
                    for pop_id in range(rd.num_agents):
                        policy_id = int(cc.team_selection[pop_id, team_number])
                        cc.fitness[pop_id, policy_id] = reward[pop_id]
                elif rtype == 3:
                    reward = hetr.calc_sdpp(rd.rover_path, rd.poi_value, rd.poi_pos)
                    for pop_id in range(rd.num_agents):
                        policy_id = int(cc.team_selection[pop_id, team_number])
                        cc.fitness[pop_id, policy_id] = reward[pop_id]
                else:
                    sys.exit('Incorrect Reward Type for Heterogeneous Teams')

            cc.down_select()  # Perform down_selection after each policy has been evaluated

            # Testing Phase
            rd.reset_to_init()  # Reset rovers to initial positions
            done = False; rd.istep = 0
            joint_state = rd.get_joint_state()
            while not done:
                for rover_id in range(rd.num_agents):
                    nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, 0], rover_id)
                joint_state, done = rd.step(nn.out_layer)

            reward = hetr.calc_hetero_global(rd.rover_path, rd.poi_value, rd.poi_pos)
            reward_history.append(reward)

            if gen == (p.generations-1):  # Save path at end of final generation
                save_rover_path(rd.rover_path)
                if p.visualizer_on:
                    visualize(rd, reward)

        if rtype == 0:
            save_reward_history(reward_history, "Global_Reward.csv")
        if rtype == 1:
            save_reward_history(reward_history, "Difference_Reward.csv")
        if rtype == 2:
            save_reward_history(reward_history, "DPP_Reward.csv")
        if rtype == 3:
            save_reward_history(reward_history, 'SDPP_Reward.csv')

'''

# HOMOGENEOUS ROVER TEAMS ---------------------------------------------------------------------------------------------
def run_homogeneous_rovers():
    cc = ccea.Ccea()
    nn = neural_network.NeuralNetwork()
    rd = RoverDomain()

    rtype = p.reward_type

    for srun in range(p.stat_runs):  # Perform statistical runs
        print("Run: %i" % srun)
        reward_history = []

        # Reset CCEA, NN, and world for new stat run
        cc.reset_populations()  # Randomly initialize ccea populations
        nn.reset_nn()  # Initialize NN architecture
        rd.reset()  # Re-initialize world

        save_world_configuration(rd.rover_initial_pos, rd.poi_pos, rd.poi_value)

        for gen in range(p.generations):
            # print("Gen: %i" % gen)
            cc.select_policy_teams()  # Selects which policies will be grouped into which teams
            for team_number in range(cc.population_size):  # Each policy in CCEA is tested in teams
                rd.reset_to_init()  # Resets rovers to initial configuration

                done = False; rd.istep = 0
                joint_state = rd.get_joint_state()
                while not done:
                    for rover_id in range(rd.num_agents):
                        policy_id = int(cc.team_selection[rover_id][team_number]) # get policy selected from previous generation
                        nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, policy_id], rover_id)
                    joint_state, done = rd.step(nn.out_layer)

                    temp_state = joint_state

                    # print("#############", np.shape(temp_state), "##################")
                    temp_state = numpy.array(temp_state)
                    #print("####", temp_state, "#####")

                    file = open("CCEA_data_%d_%d_%d.txt" % (p.num_rovers, p.num_pois, p.angle_resolution), "a")
                    file.write(str(temp_state) + "\n")

                    file.close()

                # Update fitness of policies using reward information
                if rtype == 0:
                    reward, poi_status = homr.calc_global(rd.rover_path, rd.poi_value, rd.poi_pos)
                    rd.poi_status = poi_status

                    for pop_id in range(rd.num_agents):
                        policy_id = int(cc.team_selection[pop_id][team_number])
                        cc.fitness[pop_id, policy_id] = reward



                elif rtype == 1:
                    reward = homr.calc_difference(rd.rover_path, rd.poi_value, rd.poi_pos)
                    for pop_id in range(rd.num_agents):
                        policy_id = int(cc.team_selection[pop_id][team_number])
                        cc.fitness[pop_id, policy_id] = reward[pop_id]
                    print("Generation", gen, "out of", p.generations, "Difference Reward:", reward)

                elif rtype == 2:
                    reward = homr.calc_dpp(rd.rover_path, rd.poi_value, rd.poi_pos)
                    for pop_id in range(rd.num_agents):
                        policy_id = int(cc.team_selection[pop_id][team_number])
                        cc.fitness[pop_id, policy_id] = reward[pop_id]
                    print("Generation", gen, "out of", p.generations, "D++ Reward:", reward)

                else:
                    sys.exit('Incorrect Reward Type for Homogeneous Teams')

            cc.down_select()  # Perform down_selection after each policy has been evaluated

            if rtype == 0:
                print("Generation", gen, "out of", p.generations, "Global Reward:", reward)
            elif rtype == 1:
                print("Generation", gen, "out of", p.generations, "Difference Reward:", reward)
            elif rtype == 2:
                print("Generation", gen, "out of", p.generations, "D++ Reward:", reward)
            else:
                sys.exit('Incorrect Reward Type for Homogeneous Teams')

            # Testing Phase
            rd.reset_to_init()  # Reset rovers to initial positions
            done = False; rd.istep = 0
            joint_state = rd.get_joint_state()
            while not done:
                for rover_id in range(rd.num_agents):
                    nn.run_neural_network(joint_state[rover_id], cc.pops[rover_id, 0], rover_id)
                joint_state, done = rd.step(nn.out_layer)

            reward, poi_status = homr.calc_global(rd.rover_path, rd.poi_value, rd.poi_pos)
            reward_history.append(reward)
            print("Global Reward", reward)

            if gen%1==0:  # Save path at end of final generation
                save_rover_path(rd.rover_path)
                if p.visualizer_on:
                    visualize(rd, reward)



        if rtype == 0:
            save_reward_history(reward_history, "Global_Reward.csv")
        if rtype == 1:
            save_reward_history(reward_history, "Difference_Reward.csv")
        if rtype == 2:
            save_reward_history(reward_history, "DPP_Reward.csv")


def main():
    if p.team_types == 'homogeneous':
        run_homogeneous_rovers()
    #elif p.team_types == 'heterogeneous':
    #    run_heterogeneous_rovers()
    else:
        print('ERROR')


main()  # Run the program
