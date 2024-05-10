import time
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
    def __init__(self, env, input_dir):
        super(TD3Agent, self).__init__()
        self.env = env

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

        # add noise
        with torch.no_grad():
            action = torch.from_numpy(np.array(action)).to(device)

            noise_rate = min(1, 1 - data_size / 100_000)
            rand_action = torch.rand(action.shape).to(device)
            rand_action[1] = rand_action[1] * 2 - 1
            noise = (rand_action * noise_rate).to(device)
            # print(f"E Before: {action[0]}")
            action = (
                (action * (1 - noise_rate)) + noise * noise_rate
            ).clamp(-C.max_action, C.max_action)
            # print(f"E After: {action[0]}")
            action = action.cpu().data.numpy().flatten()


        return action, {}

    def store_transition(self, state, action, reward, next_state, done):
    # 将一个step的s a r s' done保存到 buffer
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

            # print(f'ACTION TB: {actions[0]}')

            with torch.no_grad():
                noise = (
                        torch.randn_like(actions) * 0.02
                ).clamp(-0.05, 0.05).to(device)
                # print(f"NOISE: {noise[0]}")
                # print(f"T Before: {self.actor_target(next_states[0])}")

                next_action = (
                        self.actor_target(next_states) + noise
                ).clamp(-C.max_action, C.max_action)

                # print(f"T After: {next_action[0]}")
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_states, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards.unsqueeze(1) + dones.unsqueeze(1) * C.discount * target_Q
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(states, actions)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # print("C-LOSS SHAPE")
            # print(critic_loss.shape)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_it % C.policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
                # print("A-LOSS SHAPE")
                # print(actor_loss.shape)
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
        self.l3 = nn.Linear(256, C.action_dim)

        self.max_action = max_action

    def forward(self, state):
        state = state.to(device)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        action = self.max_action * torch.tanh(self.l3(a))
        action[:,0] = (action[:,0] + 1) / 2
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