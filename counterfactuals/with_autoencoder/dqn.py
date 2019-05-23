import os, random, math
import mod_dqn as mod
from random import randint
import numpy as np, torch


class Parameters:
    def __init__(self):

        #NN specifics
        self.num_hnodes = self.num_mem = 100
        self.is_dpp = True
        self.target_synch = 500

        # Train data
        self.batch_size = 10000
        self.num_episodes = 500000
        self.actor_epoch = 1; self.actor_lr = 0.005
        self.critic_epoch = 1; self.critic_lr = 0.005
        self.update_frequency = 20

        #Rover domain
        self.dim_x = self.dim_y = 20; self.obs_radius = 10; self.act_dist = 2.5; self.angle_res = 20
        self.num_poi = 10; self.num_rover = 6; self.num_timestep = 20
        self.poi_rand = 1; self.coupling = 5; self.rover_speed = 1
        self.is_homogeneous = True  #False --> Heterogenenous Actors
        self.sensor_model = 2 #1: Density Sensor
                              #2: Closest Sensor


        #Dependents
        self.state_dim = 2*360 / self.angle_res + 5
        self.action_dim = 5
        self.epsilon = 0.2; self.alpha = 0.9; self.gamma = 0.99
        self.reward_crush = 0.2 #Crush reward to prevent numerical issues

        #Replay Buffer
        self.buffer_size = 1000000

        self.save_foldername = 'R_Block/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)

        #Unit tests (Simply changes the rover/poi init locations)
        self.unit_test = 0 #0: None
                           #1: Single Agent
                           #2: Multiagent 2-coupled

class Tracker(): #Tracker
    def __init__(self, parameters, vars_string, project_string):
        self.vars_string = vars_string; self.project_string = project_string
        self.foldername = parameters.save_foldername
        self.all_tracker = [[[],0.0,[]] for _ in vars_string] #[Id of var tracked][fitnesses, avg_fitness, csv_fitnesses]
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)

    def update(self, updates, generation):
        for update, var in zip(updates, self.all_tracker):
            var[0].append(update)

        #Constrain size of convolution
        if len(self.all_tracker[0][0]) > 100: #Assume all variable are updated uniformly
            for var in self.all_tracker:
                var[0].pop(0)

        #Update new average
        for var in self.all_tracker:
            var[1] = sum(var[0])/float(len(var[0]))

        if generation % 10 == 0:  # Save to csv file
            for i, var in enumerate(self.all_tracker):
                var[2].append(np.array([generation, var[1]]))
                filename = self.foldername + self.vars_string[i] + self.project_string
                np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')

if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    env = mod.Task_Rovers(parameters); tracker = Tracker(parameters, ['rewards'], '')


    #Create agent teams
    all_agents = []
    if parameters.is_homogeneous: #Homogeneous team (all agents pointing to the same agent)
        agent = mod.A2C_Discrete(parameters, 0)
        for i in range(parameters.num_rover):
            all_agents.append(agent) # Create a homogeneous multiagent team (all reference point to the same agent)
    else: #Heterogeneous team
        for i in range(parameters.num_rover): all_agents.append(mod.A2C_Discrete(parameters, i)) #Create a hetero multiagent team

    #LEARNING LOOP
    for episode in range(1, parameters.num_episodes, 1): #Each episode
        episode_reward = 0.0; env.reset() #Reset environment
        for agent in all_agents: agent.ledger.reset() #Reset ledger
        macro_experience = [[None, None, None, None, rover_id] for rover_id in range(parameters.num_rover)] #Bucket to store macro actions (time extended) experiences.

        #ONE EPISODE OF LEARNING
        for timestep in range(1, parameters.num_timestep+1): #Each timestep

            # Get current state from environment
            if timestep == 1: #For first timestep, explicitly get current state from the env
                joint_state = []
                for rover_id in range(parameters.num_rover): joint_state.append(mod.to_tensor(env.get_state(rover_id, agent.ledger)))
            else: joint_state = joint_next_state #For timestep >1, simply use the next joint state computed at the end of last timestep

            # Get action
            joint_action = torch.cat([agent.ac.actor_forward(joint_state[i]) for i, agent in enumerate(all_agents)], 1)
            joint_action_prob = mod.to_numpy(joint_action) #[probs, batch]

            greedy_actions = [] #Greedy actions breaking ties
            for i in range(len(joint_action_prob[0])):
                max = np.max(joint_action_prob[:,i])
                greedy_actions.append(np.random.choice(np.where(max == joint_action_prob[:,i])[0]))
            greedy_actions = np.array(greedy_actions)

            #Epsilon greedy exploration through action choice perturbation
            rand = np.random.uniform(0,1,parameters.num_rover)
            is_perturb = rand < parameters.epsilon
            if episode % parameters.update_frequency == 0: is_perturb = np.zeros(parameters.num_rover).astype(bool) #Greedy for these test episodes
            actions = np.multiply(greedy_actions, (np.invert(is_perturb))) + np.multiply(np.random.randint(0, parameters.action_dim, parameters.num_rover), (is_perturb))

            #TODO
            #if timestep < 5: actions = mod.oracle2(env, timestep)

            #Macro action to macro (time-extended macro action)
            for rover_id, entry in enumerate(env.util_macro):
                 if actions[rover_id] == 5 and entry[0] == False: #Macro action first state
                     env.util_macro[rover_id][0] = True #Turn on is_currently_active?
                     env.util_macro[rover_id][1] = True #Turn on is_activated_now?
                     macro_experience[rover_id][0] = joint_state[rover_id]
                     macro_experience[rover_id][2] = actions[rover_id]

                 if entry[0]: actions[rover_id] = 5 #Macro action continuing

            # Run enviornment one step up and get reward
            env.step(actions, agent.ledger)
            joint_rewards = env.get_reward()
            episode_reward += (sum(joint_rewards)/parameters.coupling)/parameters.reward_crush

            #Get new state
            joint_next_state = []
            for rover_id in range(parameters.num_rover): joint_next_state.append(mod.to_tensor(env.get_state(rover_id, agent.ledger)))

            #Add new state and reward to macro-action (time extended) considerations
            for rover_id, entry in enumerate(env.util_macro):
                if entry[2]: #If reached destination
                    macro_experience[rover_id][1] = joint_next_state[rover_id]
                    macro_experience[rover_id][3] = joint_rewards[rover_id]

            #Add to memory
            for agent_id, (state, new_state, action, reward) in enumerate(zip(joint_state, joint_next_state, actions, joint_rewards)):
                if action == 5: continue #Skip the ones currently executing a macro action (not the one who just chose it).
                mod.add_experience(state, new_state, action, reward, all_agents[agent_id])

            #Process macro experiences and add to memory
            for id, exp in enumerate(macro_experience): #Id is common for both agent_id and rover_id
                if env.util_macro[id][2]: #If reached destination
                    env.util_macro[id][2] = False
                    mod.add_experience(macro_experience[id][0], macro_experience[id][1], macro_experience[id][2], macro_experience[id][3], all_agents[id])
                    macro_experience[id] = [None, None, None, None, id]


        #Gradient update periodically
        if episode % parameters.update_frequency == 0:
            tracker.update([episode_reward], episode)
            for agent in all_agents:
                agent.update_critic(episode)
                agent.update_actor(episode, is_dpp = parameters.is_dpp)
                if parameters.is_homogeneous: break #Break after one round of training
            print('Gen', episode, 'Reward', episode_reward, 'Aggregrate', "%0.2f" % tracker.all_tracker[0][1], 'Epsilon', "%0.2f" %parameters.epsilon)#, 'Mem_size', agent.replay_buffer.size() 'Exp_Success:', "%0.2f" % (explore_success/episode),
            #env.trace_viz()
        if episode % parameters.target_synch == 0:
            agent.synchronize()

        #if episode % 200 == 0: visualize_episode(env, agent, parameters)
        #if episode % parameters.update_frequency == 0: trace_viz(env, agent, parameters)
        #if episode % 50 == 0: v_check(env, agent.ac, parameters)
        #if episode % 50 == 0: actor_check(env, agent.ac, parameters)


