import random
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
from gym import Env

from core.policies.pt_policy import PTPolicy
from core.roadmap.dispatcher import TaskDispacher
from core.policies.pure_persuit import PurePursuiteAdaptor, PolicyAdapter
from .environment import ReinformerQLabEnv
from .vehicle import ReinformerPolicy
from .settings import ACT_DIM, STATE_DIM


# TODO: Solve ong parameter list in the this function
def reinformer_car_eval(
    model: nn.Module,
    model_path: str,
    device: str,
    context_len: int,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    num_eval_ep: int = 10,
    max_test_ep_len: int = 1000,
) -> Tuple[float, float, float, float]:
    # initialize the environment
    env: Env = ReinformerQLabEnv()
    # initialize the agent and the expert
    agent: PTPolicy = ReinformerPolicy(model=model, model_path=model_path)
    expert: PolicyAdapter = PurePursuiteAdaptor()

    # initialize returns
    returns, lengths = [], []
    for _ in range(num_eval_ep):
        # initialize the task assigner
        start_index: int = random.randint(0, 24)
        task_assigner: TaskDispacher = TaskDispacher(start_node=start_index)
        task, waypoints = task_assigner.get_one_task()
        # Reinitialize the environment
        state, reward, done, _ = env.reset(task, waypoints)
        episode_return: float = 0
        episode_length: float = 0
        # setup the agent
        agent.setup(
            eval_batch_size=1,
            max_test_ep_len=max_test_ep_len,
            context_len=context_len,
            state_mean=state_mean,
            state_std=state_std,
            state_dim=STATE_DIM,
            act_dim=ACT_DIM,
            device=device
        )

        # execute the steps
        for _ in range(max_test_ep_len):
            expert_action, _ = expert.execute(state)
            agent_action, _ = agent.execute(state)
            state, reward, done, _ = env.step(agent_action, expert_action)
            episode_return += reward
            episode_length += 1
            if done:
                returns.append(episode_return)
                lengths.append(episode_length)
                break

    return (
        np.array(returns).mean(),
        np.array(returns).std(),
        np.array(lengths).mean(),
        np.array(lengths).std()
    )


def Reinformer_eval(
    model,
    device,
    context_len,
    env,
    state_mean,
    state_std,
    num_eval_ep=10,
    max_test_ep_len=1000,
):
    eval_batch_size = 1
    returns = []
    lengths = []

    state_dim = 10
    act_dim = 2

    state_mean = torch.from_numpy(state_mean).to(device)
    state_std = torch.from_numpy(state_std).to(device)

    timesteps = torch.arange(start=0, end=max_test_ep_len, step=1)
    timesteps = timesteps.repeat(eval_batch_size, 1).to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(num_eval_ep):
            # zeros place holders
            actions = torch.zeros(
                (eval_batch_size, max_test_ep_len, act_dim),
                dtype=torch.float32,
                device=device,
            )
            states = torch.zeros(
                (eval_batch_size, max_test_ep_len, state_dim),
                dtype=torch.float32,
                device=device,
            )
            returns_to_go = torch.zeros(
                (eval_batch_size, max_test_ep_len, 1),
                dtype=torch.float32,
                device=device,
            )

            # init episode
            running_state = env.reset()
            episode_return = 0
            episode_length = 0

            for t in range(max_test_ep_len):
                # add state in placeholder and normalize
                states[0, t] = torch.from_numpy(running_state).to(device)
                states[0, t] = (states[0, t] - state_mean) / state_std
                # predict rtg by model
                if t < context_len:
                    rtg_preds, _, _ = model.forward(
                        timesteps[:, :context_len],
                        states[:, :context_len],
                        actions[:, :context_len],
                        returns_to_go[:, :context_len],
                    )
                    rtg = rtg_preds[0, t].detach()
                else:
                    rtg_preds, _, _ = model.forward(
                        timesteps[:, t - context_len + 1 : t + 1],
                        states[:, t - context_len + 1 : t + 1],
                        actions[:, t - context_len + 1 : t + 1],
                        returns_to_go[:, t - context_len + 1 : t + 1],
                    )
                    rtg = rtg_preds[0, -1].detach()
                # add rtg in placeholder
                returns_to_go[0, t] = rtg
                # take action by model
                if t < context_len:
                    _, act_dist_preds, _ = model.forward(
                        timesteps[:, :context_len],
                        states[:, :context_len],
                        actions[:, :context_len],
                        returns_to_go[:, :context_len],
                    )
                    act = act_dist_preds.mean.reshape(eval_batch_size, -1, act_dim)[0, t].detach()
                else:
                    _, act_dist_preds, _ = model.forward(
                        timesteps[:, t - context_len + 1 : t + 1],
                        states[:, t - context_len + 1 : t + 1],
                        actions[:, t - context_len + 1 : t + 1],
                        returns_to_go[:, t - context_len + 1 : t + 1],
                    )
                    act = act_dist_preds.mean.reshape(eval_batch_size, -1, act_dim)[0, -1].detach()
                # env step
                running_state, running_reward, done, _ = env.step(
                    act.cpu().numpy()
                )
                # add action in placeholder
                actions[0, t] = act
                # calculate return and episode length
                episode_return += running_reward
                episode_length += 1
                # terminate
                if done:
                    returns.append(episode_return)
                    lengths.append(episode_length)
                    break

    return np.array(returns).mean(), np.array(returns).std(), np.array(lengths).mean(), np.array(lengths).mean()
