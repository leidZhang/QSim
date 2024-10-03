import os
import time
import datetime
from typing import *
from multiprocessing import Queue

import wandb
import torch

from settings import BATCH_SIZE, MAX_TRAIN_STEPS
from settings import DQN_MODEL_PATH, UPDATE_FREQ
from .agent import Agent
from generator.environment import CrossRoadEnvironment


# TODO: Trainer should not use the online-training environment, but instead use a replay buffer to store transitions and sample from it.
def train():
    env = CrossRoadEnvironment()
    agent = Agent(action_size=5)  # Assuming 5 discrete actions
    num_episodes = 1000
    max_steps = 1000

    for episode in range(num_episodes):
        observation, reward, done, info = env.reset()
        state = agent.process_observation(observation)
        batch_size = agent.batch_size
        hidden = (torch.zeros(1, batch_size, 256),
                  torch.zeros(1, batch_size, 256))  # Initialize LSTM hidden state
        total_reward = 0

        for t in range(max_steps):
            action, hidden = agent.select_action(state, hidden)
            next_observation, reward, done, info = env.step(action)
            next_state = agent.process_observation(next_observation)
            total_reward += reward

            # Store transition in memory
            agent.memory.push(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state

            # Perform optimization
            agent.optimize_model()

            if done:
                print(f"Episode {episode}, Total Reward: {total_reward}")
                break

        # Update the target network
        if episode % agent.target_update == 0:
            agent.update_target_network()

    # Save the trained model
    torch.save(agent.policy_net.state_dict(), 'd3qn_per_lstm_model.pth')


# TODO: Implement your trainer here
class DQNTrainer:
    def __init__(self, data_queue: Queue) -> None:
        self.train_step: int = 0
        self.done: bool = False
        self.data_queue: Queue = data_queue

        self.replay_buffer: Any = None
        self.agent: Agent = Agent(action_size=1)
        self._resume_from_last_checkpoint()

        self.save_timer: float = time.time()
        self.update_timer: float = time.time()

    def _resume_from_last_checkpoint(self) -> None:
        if os.path.exists(DQN_MODEL_PATH):
            checkpoint: dict = torch.load(DQN_MODEL_PATH, map_location="cpu")
            self.train_step = checkpoint["epoch"]
            self.agent.load_state_dict(checkpoint["model_state_dict"])
            print(f"Resuming from step {self.train_step}")

    def _check_buffer_update(self) -> None:
        if time.time() - self.update_timer > UPDATE_FREQ:
            # Directly read the data from the queue
            while not self.data_queue.empty():
                data: dict = self.data_queue.get()
                self.replay_buffer.add(data)
            self.update_timer = time.time()

    def _save_weights(self) -> None:
        if time.time() - self.save_timer > 60:
            # TODO: Implement saving the weights here
            checkpoint: dict = {
                "epoch": self.train_step,
                "model_state_dict": self.agent.state_dict(),
                "optimizer_actor_state_dict": self.agent.actor_optimizer.state_dict(),
                "optimizer_critic_state_dict": self.agent.critic_optimizer.state_dict(),
            }
            torch.save(checkpoint, DQN_MODEL_PATH)
            print(f"Saved checkpoint {self.train_step}")
            self.save_timer = time.time()

    def _learn(self, metric: Dict[str, float]) -> float:
        start: float = time.time()
        critic_loss, actor_loss = self.agent.train(self.replay_buffer, BATCH_SIZE)
        if actor_loss is not None:
            metric["actor_loss"] = actor_loss.item()
        metric["critic_loss"] = critic_loss.item()
        metric["train/steps"] = self.train_step
        return start

    def _log_metric(self, start: float, metric: Dict[str, float]) -> None:
        time_diff: float = time.time() - start
        metric['_timestamp'] = datetime.now().timestamp()
        metric['train/fps'] = 1 / time_diff
        if "actor_loss" in metric.keys() and metric["actor_loss"] is not None:
            wandb.log(data={
                "train/actor_loss": metric["actor_loss"]
            }, step=self.train_step)
        wandb.log(data={
            "train/critic_loss": metric['critic_loss'],
            "train/fps": metric['train/fps'],
        }, step=self.train_step)

    def train(self) -> None:
        metric: Dict[str, float] = {}

        self._check_buffer_update()
        start: float = self._learn(metric)
        self._save_weights()
        self._log_metric(start, metric)

        self.train_step += 1
        if self.train_step >= MAX_TRAIN_STEPS:
            self.done = True


if __name__ == "__main__":
    train()
