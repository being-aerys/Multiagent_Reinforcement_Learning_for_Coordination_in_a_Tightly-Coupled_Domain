import argparse, os
import math
from collections import namedtuple
from itertools import count
import torchvision
import gym
import numpy as np
from gym import wrappers
import torch.utils.data
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
import torch
from matplotlib.ticker import MaxNLocator
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plot
import keyboard
import time
import numpy as np
import math

import pickle

args = Parameters()
tracker = utils.Tracker(args, ['rewards'], '')
env = Task_Rovers(args)

device = torch.device("cuda")
torch.manual_seed(args.seed)
np.random.seed(args.seed)

with_ae_rewards = "with_ae_%d_rovers_%d_POI_rewards.pkl" % (args.num_rovers, args.num_pois)
with_ae_path = "with_ae_%d_rovers_%d_POI_path.pkl" % (args.num_rovers, args.num_pois)

class AutoEncoder(nn.Module):

    # The class constructor
    def __init__(self, rover_state_size):
        # We call the class constructor of the parent class Module
        super(AutoEncoder, self).__init__()
        # The encoder part of the auto-encoder. nn.Sequential joins several layers end to end
        self.encoder = nn.Sequential(nn.Linear(rover_state_size, 1024),
                                     nn.LeakyReLU(),
                                     nn.Linear(1024, 256),
                                     nn.LeakyReLU(),
                                     nn.Linear(256, 32),
                                     nn.LeakyReLU(),
                                     nn.Linear(32, 10),
                                     nn.LeakyReLU())

        # The decoder part of the auto-encoder. nn.Sequential joins several layers end to end
        self.decoder = nn.Sequential(nn.Linear(10, 32),
                                     nn.LeakyReLU(),
                                     nn.Linear(32, 256),
                                     nn.LeakyReLU(),
                                     nn.Linear(256, 1024),
                                     nn.LeakyReLU(),
                                     nn.Linear(1024, rover_state_size))

    # This function is the forward pass of the auto-encoder neural network
    def forward(self, x):
        # This is the encoder section
        x1 = self.encoder(x)
        # This is the decoder section
        x2 = self.decoder(x1)
        # Returns the decoded result
        return x1, x2

    # This function can be used to load a model
    def load_model(self, model_location):
        # Loads the pretrained dictionary
        pretrained_dict = torch.load(model_location)
        # The dictionary for the current neural network
        current_net_dict = self.state_dict()
        # Iterating through the key-value pairs in the dictionary
        for key, value in pretrained_dict.items():
            # Copies the values from the pretrained dictionary to the current model
            current_net_dict[key] = pretrained_dict[key]
        # Loads the data in the current network dictionary
        self.load_state_dict(current_net_dict)

    # This is a function to evaluate the neural network
    def evaluate_network(self, data_loader, loss_criterion):
        # initializing the total evaluation loss to be zero
        total_evaluation_loss = 0
        #count the number of total samples in the whole dataloader
        total_samples = 0
        # Making the network to be in evaluation mode so that losses won't be accumulated, thereby saving memory
        self.eval()


poi_vals = env.set_poi_values()
if not os.path.exists(args.save_foldername): os.makedirs(args.save_foldername)



if args.algo == "NAF":
    agent = NAF(args.gamma, args.tau, args.num_hnodes, args.autoencoder_output_length, env.action_space, args)
else:
    agent = DDPG(args.gamma, args.tau, args.num_hnodes, args.autoencoder_output_length, env.action_space, args)

memory = ReplayMemory(args.buffer_size)
ounoise = OUNoise(env.action_space.shape[0])



model1 = AutoEncoder(40)
model1.load_state_dict(torch.load('DDDPG_4_10_20_LS_20.pth'))
model1 = model1.to(device)
model1.eval()


episode_rewards_list = []
rover_path_list = []
poi_pos_list = []
poi_status_list = []


for i_episode in range(args.num_episodes):
    joint_state = utils.to_tensor(np.array(env.reset())) # reset the environment
    joint_state, _ = model1.forward(joint_state)

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

        #joint_reward = global_joint_reward



        #print("Joint Next State", joint_next_state)
        #if (joint_reward[0] > 0):
        #    print(joint_reward[0])


        temp_state = joint_next_state
        temp_state = temp_state.detach() #todo: not required for without AE
        temp_state = np.array(temp_state.cpu())
        #print("#############", np.shape(temp_state), "##################")

        file = open("data_%d_%d_%d.txt" % (
            args.num_rovers, args.num_pois, args.angle_resolution), "a")
        file.write(str(temp_state) + "\n")

        file.close()

        done = t == args.num_timesteps - 1
        #episode_reward += np.sum(joint_reward)
        episode_reward += np.sum(joint_reward)
        #episode_reward = episode_reward + global_joint_reward
        #print(episode_reward)



        #Add to memory
        joint_next_state, _ = model1.forward(joint_next_state)

        for i in range(args.num_rovers):
            action = Variable(joint_action[i].unsqueeze(0))
            state = joint_state[i,:].unsqueeze(0)
            #next_state = joint_next_state[i, :].unsqueeze(0)
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



    episode_rewards_list.append(episode_reward)
    #if i_episode % args.test_frequency == 0:
    if i_episode % 10== 0:
        #env.render()
        tracker.update([episode_reward], i_episode)
        #print(i_episode, episode_reward/args.num_timesteps)
        print('Episode: {}, noise: {:.5f}, reward: {:.5f}, average reward: {:.5f}'.format(i_episode, ounoise.scale,
        float(episode_reward), float(episode_reward/args.num_timesteps)))




    if i_episode % 100 ==0:
        if args.visualization:
            visualize(env, episode_reward)


    # saves reward to plot for each episode
    if i_episode % (args.num_episodes - 1) == 0:
        # Writes the reward list for without AE rewards
        with open(with_ae_rewards, "wb") as file1:
            pickle.dump(episode_rewards_list, file1)
            #print(episode_rewards_list)
        # writes the reward list for without
        with open(with_ae_path, "wb") as file2:
            pickle.dump([env.rover_path, poi_pos_list, poi_status_list], file2)
            #print([env.rover_path, poi_pos_list, poi_status_list])



