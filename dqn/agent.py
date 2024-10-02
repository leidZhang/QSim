import random
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from .net import DuelingDQN
from .memory import PrioritizedReplayMemory, Transition


class Agent:
    """
    Agent interacting with the environment.
    """
    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = PrioritizedReplayMemory(100000)
        self.policy_net = DuelingDQN(action_size).to('cuda')
        self.target_net = DuelingDQN(action_size).to('cuda')
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.loss_func = torch.nn.MSELoss(reduction='none')
        self.steps_done = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 5000
        self.gamma = 0.95
        self.batch_size = 64
        self.sequence_length = 4  # For LSTM sequence length
        self.target_update = 10  # Frequency of target network update
        self.hidden = (
            torch.zeros(1, self.batch_size, 256).to('cuda'),
            torch.zeros(1, self.batch_size, 256).to('cuda')
        )  # Initialize hidden state

        # Initialize target network
        self.update_target_network()

    def process_observation(self, observation):
        """
        Process the observation from the environment.
        """
        # Extract images and state information from observation
        images = self.preprocess_images(observation['images'])
        state_info = torch.tensor(observation['state_info'], dtype=torch.float32)
        state = {
            'images': images,
            'state_info': state_info
        }
        return state

    def preprocess_images(self, images):
        """
        Preprocess images for input into the network: cancatenate, normalize, reorder
        No resize, should done resize before here
        """
        images: np.ndarray = np.concatenate(images, axis=-1)
        print(images.shape)
        images = torch.tensor(images, dtype=torch.float32) / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
        images = images.permute(2, 0, 1)  # Change to (channels, height, width)
        return images

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        """
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        images = state['images'].unsqueeze(0).to('cuda')
        state_info = state['state_info'].unsqueeze(0).to('cuda')

        if sample > eps_threshold:
            with torch.no_grad():
                q_values, self.hidden = self.policy_net(images, state_info, self.hidden)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_size)

        return action

    def optimize_model(self):
        """
        Perform a single optimization step.
        """
        if len(self.memory) < self.batch_size:
            return
        transitions, idxs, is_weights = self.memory.sample(self.batch_size, self.sequence_length)

        # Prepare batches
        batch = Transition(*zip(*transitions))
        images_batch = torch.stack([b['images'] for b in batch.state]).to('cuda')
        state_info_batch = torch.stack([b['state_info'] for b in batch.state]).to('cuda')

        actions_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to('cuda')
        rewards_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to('cuda')
        dones_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to('cuda')

        next_images_batch = torch.stack([b['images'] for b in batch.next_state]).to('cuda')
        next_state_info_batch = torch.stack([b['state_info'] for b in batch.next_state]).to('cuda')

        # Compute current Q values

        q_values, _ = self.policy_net(images_batch, state_info_batch, self.hidden)
        state_action_values = q_values.gather(1, actions_batch)

        # Compute target Q values
        with torch.no_grad():
            next_q_values_policy, _ = self.policy_net(next_images_batch, next_state_info_batch, self.hidden)
            next_actions = next_q_values_policy.max(1)[1].unsqueeze(1)
            next_q_values_target, _ = self.target_net(next_images_batch, next_state_info_batch, self.hidden)
            next_state_values = next_q_values_target.gather(1, next_actions)

        expected_state_action_values = (next_state_values * self.gamma * (1 - dones_batch)) + rewards_batch

        # Compute loss
        is_weights = torch.tensor(is_weights, dtype=torch.float32).unsqueeze(1).to('cuda')
        loss = self.loss_func(state_action_values, expected_state_action_values)
        loss = loss * is_weights
        prios = loss + 1e-5
        loss = loss.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        for idx, prio in zip(idxs, prios.detach().cpu().numpy()):
            self.memory.update(idx, prio)

    def update_target_network(self):
        """
        Update the target network with the policy network's weights.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

DQNPolicy = Agent