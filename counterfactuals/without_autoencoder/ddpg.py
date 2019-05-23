import sys

import torch, random
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import utils as utils

MSELoss = nn.MSELoss()

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Actor(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        #self.bn0 = nn.BatchNorm1d(num_inputs)
        #self.bn0.weight.data.fill_(1)
        #self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        #self.bn1 = nn.BatchNorm1d(hidden_size)
        #self.bn1.weight.data.fill_(1)
        #self.bn1.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        #self.bn2 = nn.BatchNorm1d(hidden_size)
        #self.bn2.weight.data.fill_(1)
        #self.bn2.bias.data.fill_(0)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)
        self.cuda()


    def forward(self, inputs):
        x = inputs
        #x = self.bn0(x)
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))

        mu = F.tanh(self.mu(x))
        return mu


class Critic(nn.Module):

    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]
        #self.bn0 = nn.BatchNorm1d(num_inputs)
        #self.bn0.weight.data.fill_(1)
        #self.bn0.bias.data.fill_(0)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        #self.bn1 = nn.BatchNorm1d(hidden_size)
        #self.bn1.weight.data.fill_(1)
        #self.bn1.bias.data.fill_(0)

        self.linear_action = nn.Linear(num_outputs, hidden_size)
        #self.bn_a = nn.BatchNorm1d(hidden_size)
        #self.bn_a.weight.data.fill_(1)
        #self.bn_a.bias.data.fill_(0)

        self.linear2 = nn.Linear(hidden_size + hidden_size, hidden_size)
        #self.bn2 = nn.BatchNorm1d(hidden_size)
        #self.bn2.weight.data.fill_(1)
        #self.bn2.bias.data.fill_(0)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        self.cuda()

    def forward(self, inputs, actions):
        x = inputs
        #x = self.bn0(x)
        x = F.tanh(self.linear1(x))
        a = F.tanh(self.linear_action(actions))
        x = torch.cat((x, a), 1)
        x = F.tanh(self.linear2(x))

        V = self.V(x)
        return V


class DDPG(object):
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space, args):

        self.args = args
        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-4)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)


    #def select_action(self, state, exploration=None):
    def select_action(self, state, exploration=None):

        self.actor.eval()
        mu = self.actor(state)
        self.actor.train()
        mu = mu.data
        if exploration is not None:
            mu += torch.Tensor(exploration.noise()).cuda()

        return mu.clamp(-1, 1)


    def update_parameters(self, batch):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        expected_state_action_batch = reward_batch + (self.gamma * next_state_action_values)

        self.critic_optim.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = MSELoss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic((state_batch),self.actor((state_batch)))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)


    def update_parameters_dpp(self, batch):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        expected_state_action_batch = reward_batch + (self.gamma * next_state_action_values)

        self.critic_optim.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = MSELoss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        #Actor
        self.actor_optim.zero_grad()

        policy_loss = -self.dpp(self.critic, self.actor, state_batch)


        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)


    def dpp(self, critic, actor, state):
        all_q = [critic((state),actor((state)))]

        state = utils.to_numpy(state)
        mid_index = 180 / self.args.angle_resolution
        coupling = self.args.coupling
        # dpp_sweep = [mid_index + i for i in range(int(-coupling/2), int(-coupling/2) + coupling, 1)]

        dpp_sweep = random.sample(range(360 / self.args.angle_resolution), coupling + 1)

        for i, add_index in enumerate(dpp_sweep):
            state[:, add_index] += 2.0
            shaped_state = utils.to_tensor(state)
            all_q.append(critic((shaped_state), actor((shaped_state)))/(i+2))

        all_q = torch.cat(all_q, 1)
        return torch.max(all_q, 1)[0].unsqueeze(1)
