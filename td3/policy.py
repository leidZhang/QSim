import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import constants as C
from torch.optim import Adam
from core.data.data_TD3 import SequenceRolloutBuffer, MlflowEpisodeRepository

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        action = self.actor(state).detach().cpu().data.numpy().flatten()
        '''
        action type #1: <class 'numpy.ndarray'>
        action shape #1: (1,)
        '''


        # add noise
        with torch.no_grad():
            action = torch.from_numpy(np.array(action))
            epsilon = max(1 - data_size / 200_000, 0.05)
            rand_action = torch.rand(action.shape)
            rand_action = rand_action * 2 - 1
            if random.uniform(0, 1) < epsilon:
                action = rand_action
            action = action.cpu().data.numpy().flatten()

        action = np.concatenate((np.array([C.action_v]), action), axis=0)
        '''
        action type #2: <class 'numpy.ndarray'>
        action shape #2: (2,)
        '''
        return action, {}  # action: np array

    def store_transition(self, state, action, reward, next_state, done):
    # step --> buffer
        self.buffer.add(state, action, reward, next_state, done)

    def learn(self, samples):
        if len(self.buffer) > C.batch_size:
            self.total_it += 1
            # Get samples from buffer and Convert to PyTorch tensors
            states = torch.FloatTensor(samples['states']).to(device)
            actions = torch.FloatTensor(samples['actions']).to(device)
            rewards = torch.FloatTensor(samples['rewards']).to(device)
            next_states = torch.FloatTensor(samples['next_states']).to(device)
            dones = torch.FloatTensor(samples['dones']).to(device)

            # print(f'actions: {actions}')
            # print(f'shape of action: {actions.shape}')
            # print(f'type of action: {type(actions)}')
            '''
            shape of actions: torch.Size([40, 2])
            type of actions: <class 'torch.Tensor'>
            actions: tensor([[ 7.6000e-02, -7.9606e-02],
                             [ 7.6000e-02,  8.2937e-06],
            '''

            # with torch.no_grad():
            #     noise = (
            #             torch.randn_like(actions) * 0.02
            #     ).clamp(-0.05, 0.05).to(device)
            #     print(f'noise: {noise}')
            #     print(f'shape of noise: {noise.shape}')
            #
            #     next_action = (
            #             self.actor_target(next_states) + noise
            #     ).clamp(-C.max_action, C.max_action)
            #     print(f'next_action: {next_action}')
            #     print(f'shape of next_action: {next_action.shape}')

            with torch.no_grad():
                noise_yaw = (
                        torch.randn_like(actions[:, 0].unsqueeze(1)) * 0.1
                ).clamp(-0.25, 0.25).to(device)
                # print(f'shape of noise_yaw: {noise_yaw.shape}')

                next_actions_yaw = (
                        self.actor_target(next_states) + noise_yaw
                ).clamp(-0.5, 0.5)
                # print(f'shape of next_actions_yaw: {next_actions_yaw.shape}')
                # print(f'shape of next_action_yaw: {next_actions_yaw.shape}')

                next_actions = torch.cat([actions[:, 0].unsqueeze(1), next_actions_yaw], 1)
                # print(f'shape of next_action: {next_actions.shape}')


                # action = np.concatenate((np.array([C.action_v]), action), axis=0)
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards.unsqueeze(1) + dones.unsqueeze(1) * C.discount * target_Q
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(states, actions)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_it % C.policy_freq == 0:

                actions_v = torch.full((C.batch_size, 1), C.action_v).to(device)
                actions_yaw = self.actor(states).to(device)
                actions = torch.cat((actions_v, actions_yaw), dim=1)

                # Compute actor loss
                actor_loss = -self.critic.Q1(states, actions).mean()
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(C.tau * param.data + (1 - C.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(C.tau * param.data + (1 - C.tau) * target_param.data)
                return (actor_loss.detach().cpu(), critic_loss.detach().cpu())
            else:
                return (None, critic_loss.detach().cpu())
        else:
            return (None, None)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


class Actor(torch.nn.Module):
    def __init__(self, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(8, 256) # (6, 256)
        self.l2 = nn.Linear(256, 256)
        # self.l3 = nn.Linear(256, C.action_dim)
        self.l3 = nn.Linear(256, 1)

        self.max_action = max_action

    def forward(self, state):
        state = state.to(device)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        action = self.max_action * torch.tanh(self.l3(a))
        '''
        action: tensor([[-0.0273]], grad_fn=<MulBackward0>)
        action type #0: <class 'torch.Tensor'>
        action shape #0: torch.Size([1, 1])
        '''

        # print(f'action: {action}')
        # print(f'action type #0: {type(action)}')
        # print(f'action shape #0: {action.shape}')
        return action

class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(C.state_dim + C.action_dim, 256)
        # self.l1 = nn.Linear(C.state_dim + 1, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(C.state_dim + C.action_dim, 256)
        # self.l4 = nn.Linear(C.state_dim + 1, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1