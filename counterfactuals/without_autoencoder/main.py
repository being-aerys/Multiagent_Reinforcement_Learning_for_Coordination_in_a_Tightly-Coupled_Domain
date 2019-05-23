import argparse, os
import math
from collections import namedtuple
from itertools import count

import gym
import numpy as np
from gym import wrappers

import torch
from ddpg import DDPG
#from naf import NAF
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from replay_memory import ReplayMemory, Transition
from rover_domain import Task_Rovers
import utils as utils
from torch.autograd import Variable
#import calculate_rewards
from visualizer import visualize
from D_VAE_Parameters import Parameters
from calculate_rewards import *

args = Parameters()
tracker = utils.Tracker(args, ['rewards'], '')
env = Task_Rovers(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

poi_vals = env.set_poi_values()
if not os.path.exists(args.save_foldername): os.makedirs(args.save_foldername)


if args.algo == "NAF":
    agent = NAF(args.gamma, args.tau, args.num_hnodes, env.observation_space.shape[0], env.action_space, args)
else:
    agent = DDPG(args.gamma, args.tau, args.num_hnodes, env.observation_space.shape[0], env.action_space, args)

memory = ReplayMemory(args.buffer_size)
ounoise = OUNoise(env.action_space.shape[0])

for i_episode in range(args.num_episodes):
    joint_state = utils.to_tensor(np.array(env.reset())) # reset the environment

    if i_episode % args.test_frequency != 0:
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end - i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()

    episode_reward = 0.0
    global_joint_reward = 0.0


    # Here the joint reward is being given to each rover if it is observing some POI along with other rovers. So each
    # rover gets a reward if the POI gets observed.'
    # The task is considered done, if all POIs are being observed
    for t in range(args.num_timesteps): # TODO: insert some break point when all POIs are observed
        if i_episode % args.test_frequency == 0:
            joint_action = agent.select_action(joint_state, ounoise)
        else:
            joint_action = agent.select_action(joint_state, None)  # doing exploratory task once in a while (in test_frequency interval)

        #print("action:", joint_action)

        joint_next_state, joint_reward = env.step(joint_action.cpu().numpy())

        joint_next_state = utils.to_tensor(np.array(joint_next_state), volatile=True)

        rover_current_pos = env.rover_pos # get the current pos of the rover

        #dpp_joint_reward = calc_dpp(rover_current_pos, poi_vals, env.poi_pos)  # calculate D++ reward

        global_joint_reward = calc_global(rover_current_pos, poi_vals, env.poi_pos)  # calculate global reward

        #if (global_joint_reward>0):
        #    print(i_episode, global_joint_reward)

        joint_reward = global_joint_reward


        #print("Joint Next State", joint_next_state)
        #if (joint_reward[0] > 0):
        #    print(joint_reward[0])


        temp_state = joint_next_state
        temp_state = np.array(temp_state.cpu())
        #print("#############", np.shape(temp_state), "##################")

        file = open("data_%d_%d_%d.txt" % (args.num_rovers, args.num_pois, args.angle_resolution), "a")
        file.write(str(temp_state) + "\n")

        file.close()

        done = t == args.num_timesteps - 1
        #episode_reward += np.sum(joint_reward)
        episode_reward += joint_reward[0]
        #episode_reward = episode_reward + global_joint_reward
        #print(episode_reward)

        #Add to memory
        for i in range(args.num_rovers):
            action = Variable(joint_action[i].unsqueeze(0))
            state = joint_state[i,:].unsqueeze(0)
            next_state = joint_next_state[i, :].unsqueeze(0)
            reward = utils.to_tensor(np.array([joint_reward[i]])).unsqueeze(0) #todo: what reward it signifies
            #reward = utils.to_tensor(np.array([global_joint_reward[i]])).unsqueeze(0)  # take this as the DPP reward
            #reward = global_joint_reward
            memory.push(state, action, next_state, reward)

        joint_state = joint_next_state

        #with open("array.txt", "wb") as f:
        #    f.write("%s\n" % np.array(temp_state.cpu())) # %
            #f.write('\n')

        if len(memory) > args.batch_size * 5:
            for _ in range(args.updates_per_step):
                transitions = memory.sample(args.batch_size)
                batch = Transition(*zip(*transitions))
                #agent.update_parameters_dpp(batch) # todo: need to see how update_parameters_dpp is different from other
                agent.update_parameters(batch)
    #path_reward = calc_dpp_path(env.rover_path, poi_vals, env.poi_pos) # calculate D++ reward for the entire trajectory



    #if i_episode % args.test_frequency == 0:
    if i_episode % 10== 0:
        #env.render()
        tracker.update([episode_reward], i_episode)
        #print(i_episode, episode_reward/args.num_timesteps)
        print('Episode: {}, noise: {:.5f}, reward: {:.5f}, average reward: {:.5f}'.format(i_episode, ounoise.scale,
        float(episode_reward), float(episode_reward/args.num_timesteps)))


###### once the training is over, test the policies ###############
if args.visualization:
    visualize(env, episode_reward)

input("Press Enter to continue...")