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
            C.resolution,
            C.state_info_dim,  # 新增，用于存储额外的状态信息维度
            C.action_dim)

        self.actor = Actor(C.max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=C.lr)

        self.critic = Critic().to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=C.lr)

        self.total_it = 0

    def select_action(self, image, state_info, data_size):
        image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device)
        state_info = torch.from_numpy(state_info).float().to(device)
        action = self.actor(image, state_info).detach().cpu().data.numpy().flatten()

        # add noise
        with torch.no_grad():
            action = torch.from_numpy(np.array(action))
            epsilon = max(1 - data_size / 200_000, 0.04)

            rand_action_v = torch.rand(action[0].shape).to(device)
            rand_action_yaw = (torch.rand(action[1].shape) * 2 - 1).to(device)

            if random.uniform(0, 1) < epsilon:
                action[0] = rand_action_v
                action[1] = rand_action_yaw

            action_v = action[0].cpu().data.numpy().flatten()
            action_yaw = action[1].cpu().data.numpy().flatten()

        action = np.concatenate((action_v, action_yaw), axis=0)

        return action, {}  # action: np array

    def store_transition(self, image, state_info, action, reward, next_image, next_state_info, done):
    # step --> buffer
        self.buffer.add(image, state_info, action, reward, next_image, next_state_info, done)

    def learn(self, samples):
        gradients = {}
        if len(self.buffer) > C.batch_size:
            self.total_it += 1
            # Get samples from buffer and Convert to PyTorch tensors
            images = torch.from_numpy(samples['images']).float().permute(0, 3, 1, 2).to(device)
            states_info = torch.from_numpy(samples['states_info']).to(device)
            actions = torch.from_numpy(samples['actions']).to(device)
            rewards = torch.from_numpy(samples['rewards']).to(device)
            next_images = torch.from_numpy(samples['next_images']).float().permute(0, 3, 1, 2).to(device)
            next_states_info = torch.from_numpy(samples['next_states_info']).to(device)
            dones = torch.from_numpy(samples['dones']).to(device)

            not_dones = ~dones

            with torch.no_grad():
                noise_yaw = (
                        torch.randn_like(actions[:, 0].unsqueeze(1)) * 0.10
                ).clamp(-0.25, 0.25).to(device)
                # print(f'noise_yaw: {noise_yaw}')

                next_actions_yaw = (
                        self.actor_target(next_images, next_states_info)[:, 1].unsqueeze(1) + noise_yaw
                ).clamp(-0.5, 0.5).to(device)
                # print(f'a_t: {self.actor_target(next_images, next_states_info)[:, 1].unsqueeze(1)}')
                # print(f'next_actions_yaw: {next_actions_yaw}')
                next_actions = torch.cat([actions[:, 0].unsqueeze(1), next_actions_yaw], 1).to(device)
                # print(f'next_actions: {next_actions}')
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_images, next_states_info, next_actions)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = rewards.unsqueeze(1) + not_dones.unsqueeze(1) * C.discount * target_Q.to(device)
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(images, states_info, actions)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            # Gradient clipping
            gradients["critic"] = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)

            self.critic_optimizer.step()

            # Delayed policy updates
            if self.total_it % C.policy_freq == 0:

                # actions_v = torch.full((C.batch_size, 1), C.action_v).to(device)
                actions = self.actor(images, states_info).to(device)
                # actions = torch.cat((actions_v.detach(), actions_yaw), dim=1).to(device)

                # Compute actor loss
                actor_loss = -self.critic.Q1(images, states_info, actions).mean().to(device)
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()

                # Gradient clipping
                gradients["actor"] = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                # torch.nn.utils.clip_grad_value_(self.actor.parameters(), clip_value=10.0)

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



# image and state as input
class Actor(nn.Module):
    def __init__(self, max_action):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_output_size = 64 * 7 * 7  # 3136
        self.fc1 = nn.Linear(conv_output_size + 6, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, C.action_dim)

        self.max_action = max_action

    def forward(self, image, state_info):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(x.size(0), -1)
        if state_info.dim() == 1:
            state_info = state_info.unsqueeze(0)
        x = torch.cat([x, state_info], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.max_action * torch.tanh(self.fc4(x))

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_output_size = 64 * 7 * 7  # 3136
        self.fc1 = nn.Linear(conv_output_size + 6 + C.action_dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)

        # Q2 architecture
        self.fc5 = nn.Linear(conv_output_size + 6 + C.action_dim, 2048)
        self.fc6 = nn.Linear(2048, 1024)
        self.fc7 = nn.Linear(1024, 512)
        self.fc8 = nn.Linear(512, 1)

    def forward(self, image, state_info, action):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(x.size(0), -1)
        x = torch.cat([x, state_info, action], 1)
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.fc3(q1))
        q1 = self.fc4(q1)

        q2 = F.relu(self.fc5(x))
        q2 = F.relu(self.fc6(q2))
        q2 = F.relu(self.fc7(q2))
        q2 = self.fc8(q2)
        return q1, q2

    def Q1(self, image, state_info, action):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(x.size(0), -1)
        # print(f'shape of x: {x.shape}')
        x = torch.cat([x, state_info, action], 1)
        # print(f'action: {action}')
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.fc3(q1))
        q1 = self.fc4(q1)
        return q1