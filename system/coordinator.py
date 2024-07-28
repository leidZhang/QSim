import os
import time
import pickle
from datetime import datetime
from typing import List, Dict

import torch
import numpy as np
import gym
import wandb
from torch.utils.data import DataLoader

from system.settings import STATE_DIM, ACT_DIM, CONTEXT_LEN, BATCH_SIZE
from system.trainer import ReinFormerTrainer
from reinformer.dataset import D4RLTrajectoryDataset
from reinformer.eval import Reinformer_eval

from core.utils.simulation import destroy_map
from core.utils.simulation import prepare_test_environment
from control.executions import run_reinformer_car
from .converter import DataConverter
from .settings import TASK


def control() -> None:
    destroy_map()
    waypoints: np.ndarray = prepare_test_environment(TASK)
    run_reinformer_car(waypoints)


def convert():
    project_path: str = os.getcwd()
    print(f"Current working directory: {project_path[:-11]}")
    local_path: str = r"mlruns\0\e4eef53e8c3a49a0b2967fa6be338fd2\artifacts\episodes_train\0"
    npz_folder_path: str = os.path.join(project_path[:-11], local_path)
    data_converter: DataConverter = DataConverter(local_path)
    trajectories: List[Dict[str, np.ndarray]] = data_converter.execute()

    with open("assets/trajectories.pkl", "wb") as f:
        pickle.dump(trajectories, f)
    # with open("assets/trajectories.pkl", "rb") as f:
    #     data = pickle.load(f)
    # print(len(data))


def experiment(variant):
    # seeding
    # seed = variant["seed"]
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    # env = variant["env"]
    # dataset = variant["dataset"]
    
    # if dataset == "complete":
    #     variant["batch_size"] = 16
    # if env == "kitchen":
    #     d4rl_env = f"{env}-{dataset}-v0"
    # elif env in ["pen", "door", "hammer", "relocate", "maze2d"]:
    #     d4rl_env = f"{env}-{dataset}-v1"
    # elif env in ["halfcheetah", "hopper", "walker2d", "antmaze"]:
    #     d4rl_env = f"{env}-{dataset}-v2"
    # if env in ["kitchen", "maze2d", "antmaze"]:
    #     variant["num_eval_ep"] = 100
    # if env == "hopper":
    #     if dataset == "medium" or dataset == "meidum-replay":
    #         variant["batch_size"] = 256
    
    # dataset_path = os.path.join(variant["dataset_dir"], f"trajectories.pkl")
    dataset_path = r'C:\Users\SDCNLab_P720\PycharmProjects\qsim\assets\trajectories.pkl'
    device = torch.device(variant["device"])

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    traj_dataset = D4RLTrajectoryDataset(
        dataset_path, variant["context_len"], device
    )

    traj_data_loader = DataLoader(
        traj_dataset,
        batch_size=variant["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    data_iter = iter(traj_data_loader)
    state_mean, state_std = traj_dataset.get_state_stats()
    formatted_state_mean = ", ".join(f"{num:.8e}" for num in state_mean)
    formatted_state_std = ", ".join(f"{num:.8e}" for num in state_std)
    print(f"{type(state_mean)} {type(state_std)}")
    print("state mean: ", state_mean, "state std: ", state_std)
    print(f'{type(formatted_state_mean)} {type(formatted_state_std)}')
    print(f'formatted_state_mean: {formatted_state_mean}, formatted_state_std: {formatted_state_std}')
    # env = gym.make(d4rl_env)
    # env.seed(seed)

    model_type = variant["model_type"]

    if model_type == "Reinformer":
        Trainer = ReinFormerTrainer(
            state_dim=STATE_DIM,
            act_dim=ACT_DIM,
            device=device,
            variant=variant
        )
        # def evaluator(model):
        #     return_mean, _, _, _ = Reinformer_eval(
        #         model=model,
        #         device=device,
        #         context_len=variant["context_len"],
        #         env = env,
        #         state_mean=state_mean,
        #         state_std=state_std,
        #         num_eval_ep=variant["num_eval_ep"],
        #         max_test_ep_len=variant["max_eval_ep_len"]
        #     )
        #     return env.get_normalized_score(
        #         return_mean
        #     ) * 100

    max_train_iters = variant["max_train_iters"]
    num_updates_per_iter = variant["num_updates_per_iter"]
    # normalized_d4rl_score_list = []
    for _ in range(1, max_train_iters+1):
        t1 = time.time()
        for epoch in range(num_updates_per_iter):
            try:
                (
                    timesteps,
                    states,
                    next_states,
                    actions,
                    returns_to_go,
                    rewards,
                    traj_mask,
                ) = next(data_iter)
            except StopIteration:
                data_iter = iter(traj_data_loader)
                (
                    timesteps,
                    states,
                    next_states,
                    actions,
                    returns_to_go,
                    rewards,
                    traj_mask,
                ) = next(data_iter)

            loss = Trainer.train_step(
                timesteps=timesteps,
                states=states,
                next_states=next_states,
                actions=actions,
                returns_to_go=returns_to_go,
                rewards=rewards,
                traj_mask=traj_mask
            )
            if variant["use_wandb"]:
                wandb.log(
                    data={
                        "training/loss" : loss,
                    }
                )
        t2 = time.time()
        # normalized_d4rl_score = evaluator(
        #     model=Trainer.model
        # )
        t3 = time.time()
        # normalized_d4rl_score_list.append(normalized_d4rl_score)
        if variant["use_wandb"]:
            wandb.log(
                data={
                        "training/time" : t2 - t1,
                        # "evaluation/score" : normalized_d4rl_score,
                        # "evaluation/time": t3 - t2
                    }
            )

    # if args.use_wandb:
    #     wandb.log(
    #         data={
    #             "evaluation/max_score" : max(normalized_d4rl_score_list),
    #             "evaluation/last_score" : normalized_d4rl_score_list[-1]
    #         }
    #     )
    # print(normalized_d4rl_score_list)
    print("=" * 60)
    print("finished training!")
    end_time = datetime.now().replace(microsecond=0)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("finished training at: " + end_time_str)
    print("=" * 60)
