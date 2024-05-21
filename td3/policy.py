import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import constants as C
from torch.optim import Adam
from core.data.data_TD3 import SequenceRolloutBuffer, MlflowEpisodeRepository

device = torch.device(C.cuda if torch.cuda.is_available() else "cpu")


class TD3Agent(torch.nn.Module):
    def __init__(self, input_dir):
        super(TD3Agent, self).__init__()

        self.buffer: SequenceRolloutBuffer = SequenceRolloutBuffer(
            MlflowEpisodeRepository(input_dir),
            C.batch_size,
            C.buffer_update_rate,
            C.observation_shape,
            C.action_dim)

        self.actor = Actor(C.max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=C.lr)

        self.critic = Critic().to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=C.lr)

        self.total_it = 0

    def select_action(self, state, data_size):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action_yaw = self.actor(state).detach().cpu().data.numpy().flatten()
        '''
        action type #1: <class 'numpy.ndarray'>
        action shape #1: (1,)
        '''


        # add noise
        with torch.no_grad():
            action_yaw = torch.from_numpy(np.array(action_yaw))
            epsilon = max(1 - data_size / 400_000, 0.04)
            # epsilon = 0
            rand_action_yaw = torch.rand(action_yaw.shape)
            rand_action_yaw = rand_action_yaw * 2 - 1
            if random.uniform(0, 1) < epsilon:
                action_yaw = rand_action_yaw

            action_yaw = action_yaw.cpu().data.numpy().flatten()

        action = np.concatenate((np.array([1.0]), action_yaw), axis=0)
        '''
        action type #2: <class 'numpy.ndarray'>
        action shape #2: (2,)
        '''
        return action, {}  # action: np array

    def store_transition(self, state, action, reward, next_state, done):
    # step --> buffer
        self.buffer.add(state, action, reward, next_state, done)

    def learn(self, samples):
        gradients = {}
        if len(self.buffer) > C.batch_size:
            self.total_it += 1
            # Get samples from buffer and Convert to PyTorch tensors
            states = torch.FloatTensor(samples['states']).to(device)
            actions = torch.FloatTensor(samples['actions']).to(device)
            rewards = torch.FloatTensor(samples['rewards']).to(device)
            next_states = torch.FloatTensor(samples['next_states']).to(device)
            dones = torch.FloatTensor(samples['dones']).to(device)
            # print(f'dones: {dones}')
            # print(f'shape of dones: {dones.shape}')
            # print(f'type of action: {type(actions)}')

            not_dones = 1. - dones
            # print(f'not_dones: {not_dones}')
            # print(f'shape of not_dones: {not_dones.shape}')
            # print(f'type of not_dones: {type(not_dones)}')

            # print(f'actions: {actions}')
            # print(f'shape of action: {actions.shape}')
            # print(f'type of action: {type(actions)}')
            '''
            shape of actions: torch.Size([40, 2])
            type of actions: <class 'torch.Tensor'>
            actions: tensor([[ 7.6000e-02, -7.9606e-02],
                             [ 7.6000e-02,  8.2937e-06],
            '''

            with torch.no_grad():
                noise_yaw = (
                        torch.randn_like(actions[:, 0].unsqueeze(1)) * 0.10
                ).clamp(-0.25, 0.25).to(device)
                # print(f'shape of noise_yaw: {noise_yaw.shape}')

                next_actions_yaw = (
                        self.actor_target(next_states) + noise_yaw
                ).clamp(-0.5, 0.5).to(device)
                # print(f'shape of next_actions_yaw: {next_actions_yaw.shape}')
                # print(f'shape of next_action_yaw: {next_actions_yaw.shape}')

                next_actions = torch.cat([actions[:, 0].unsqueeze(1), next_actions_yaw], 1).to(device)
                # print(f'shape of next_action: {next_actions.shape}')


                # action = np.concatenate((np.array([C.action_v]), action), axis=0)
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards.unsqueeze(1) + not_dones.unsqueeze(1) * C.discount * target_Q.to(device)
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(states, actions)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            # Gradient clipping
            gradients["critic"] = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=100.0)

            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_it % C.policy_freq == 0:

                actions_v = torch.full((C.batch_size, 1), C.action_v).to(device)
                actions_yaw = self.actor(states).to(device)
                actions = torch.cat((actions_v.detach(), actions_yaw), dim=1).to(device)

                # Compute actor loss
                actor_loss = -self.critic.Q1(states, actions).mean().to(device)
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()

                # Gradient clipping
                gradients["actor"] = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=100.0)

                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(C.tau * param.data + (1 - C.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(C.tau * param.data + (1 - C.tau) * target_param.data)
                return (actor_loss.detach().cpu(), critic_loss.detach().cpu(), gradients)
            else:
                return (None, critic_loss.detach().cpu(), gradients)
        else:
            return (None, None, None)


# 1 hidden layer
class Actor(torch.nn.Module):
    def __init__(self, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(C.state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        self.max_action = max_action

    def forward(self, state):
        state = state.to(device)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        action = self.max_action * torch.tanh(self.l3(a))

        return action

class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(C.state_dim + C.action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(C.state_dim + C.action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        state = state.to(device)
        action = action.to(device)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        state = state.to(device)
        action = action.to(device)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


'''
# 2
class Actor(torch.nn.Module):
    def __init__(self, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(C.state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 1)

        self.max_action = max_action

    def forward(self, state):
        state = state.to(device)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        action = self.max_action * torch.tanh(self.l4(a))

        return action

class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(C.state_dim + C.action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 1)

        # Q2 architecture
        self.l5 = nn.Linear(C.state_dim + C.action_dim, 256)
        self.l6 = nn.Linear(256, 256)
        self.l7 = nn.Linear(256, 256)
        self.l8 = nn.Linear(256, 1)

    def forward(self, state, action):
        state = state.to(device)
        action = action.to(device)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)

        q2 = F.relu(self.l5(sa))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)
        return q1, q2

    def Q1(self, state, action):
        state = state.to(device)
        action = action.to(device)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        return q1

'''
'''
# 3
class Actor(torch.nn.Module):
    def __init__(self, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(C.state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 256)
        self.l5 = nn.Linear(256, 1)

        self.max_action = max_action

    def forward(self, state):
        state = state.to(device)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        a = F.relu(self.l4(a))
        action = self.max_action * torch.tanh(self.l5(a))

        return action

class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(C.state_dim + C.action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 256)
        self.l5 = nn.Linear(256, 1)

        # Q2 architecture
        self.l6 = nn.Linear(C.state_dim + C.action_dim, 256)
        self.l7 = nn.Linear(256, 256)
        self.l8 = nn.Linear(256, 256)
        self.l9 = nn.Linear(256, 256)
        self.l10 = nn.Linear(256, 1)

    def forward(self, state, action):
        state = state.to(device)
        action = action.to(device)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = F.relu(self.l4(q1))
        q1 = self.l5(q1)

        q2 = F.relu(self.l6(sa))
        q2 = F.relu(self.l7(q2))
        q2 = F.relu(self.l8(q2))
        q2 = F.relu(self.l9(q2))
        q2 = self.l10(q2)
        return q1, q2


    def Q1(self, state, action):
        state = state.to(device)
        action = action.to(device)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = F.relu(self.l4(q1))
        q1 = self.l5(q1)
        return q1

'''
'''
# 4
class Actor(torch.nn.Module):
    def __init__(self, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(C.state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        self.max_action = max_action

    def forward(self, state):
        state = state.to(device)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l3(a))
        a = F.relu(self.l4(a))
        a = F.relu(self.l5(a))
        action = self.max_action * torch.tanh(self.l6(a))

        return action

class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(C.state_dim + C.action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        # Q2 architecture
        self.l7 = nn.Linear(C.state_dim + C.action_dim, 256)
        self.l8 = nn.Linear(256, 256)
        self.l9 = nn.Linear(256, 256)
        self.l10 = nn.Linear(256, 256)
        self.l11 = nn.Linear(256, 256)
        self.l12 = nn.Linear(256, 1)

    def forward(self, state, action):
        state = state.to(device)
        action = action.to(device)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = F.relu(self.l4(q1))
        q1 = F.relu(self.l5(q1))
        q1 = self.l6(q1)

        q2 = F.relu(self.l7(sa))
        q2 = F.relu(self.l8(q2))
        q2 = F.relu(self.l9(q2))
        q2 = F.relu(self.l10(q2))
        q2 = F.relu(self.l11(q2))
        q2 = self.l12(q2)
        return q1, q2


    def Q1(self, state, action):
        state = state.to(device)
        action = action.to(device)
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = F.relu(self.l4(q1))
        q1 = F.relu(self.l5(q1))
        q1 = self.l6(q1)
        return q1
'''