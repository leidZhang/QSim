import argparse
import logging
import os
import json
from typing import List 
from datetime import datetime
import time
import os

# import d4rl
import gym
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from ..settings import *
from .dataset import D4RLTrajectoryDataset, CustomDataSet
from .trainer import ReinFormerTrainer
from .eval import Reinformer_eval
# from td3.environment import WaypointEnvironment



def experiment(variant):
    use_wandb = True
    dataset_path = r"assets/train"
    device = torch.device(variant["device"])

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    traj_dataset = CustomDataSet(
        dataset_path, variant["context_len"], device, RESUME
    )

    traj_data_loader = DataLoader(
        traj_dataset,
        batch_size=variant["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=WORKERS
    )

    data_iter = iter(traj_data_loader)
    state_mean, state_std = traj_dataset.get_state_stats()
    # formatted_state_mean = ", ".join(f"{num:.8e}" for num in state_mean)
    # formatted_state_std = ", ".join(f"{num:.8e}" for num in state_std)
    # print(f"{type(state_mean)} {type(state_std)}")
    # print("state mean: ", state_mean, "state std: ", state_std)
    # print(f'{type(formatted_state_mean)} {type(formatted_state_std)}')
    # print(f'formatted_state_mean: {formatted_state_mean}, formatted_state_std: {formatted_state_std}')

    print("Saving state statistics...")
    state_mean_list: List[float] = [num for num in state_mean]
    state_std_list: List[float] = [num for num in state_std]
    stat: dict = {
        "state_mean": state_mean_list,
        "state_std": state_std_list,
        "return_stats": traj_dataset.return_stats
    }
    with open('state_stat.json', 'w') as f:
        json.dump(stat, f)
    print("State statistics saved.")

    # env = gym.make(d4rl_env)
    # env.seed(seed)

    model_type = variant["model_type"]

    print("Initializing trainer...")
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
    print("Start training...")
    for _ in range(1, max_train_iters+1):
        t1 = time.time()
        for epoch in range(num_updates_per_iter):
            try:
                (
                    timesteps,
                    states,
                    images,
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
                    images,
                    next_states,
                    actions,
                    returns_to_go,
                    rewards,
                    traj_mask,
                ) = next(data_iter)

            loss = Trainer.train_step(
                timesteps=timesteps,
                states=states,
                images=images,
                next_states=next_states,
                actions=actions,
                returns_to_go=returns_to_go,
                rewards=rewards,
                traj_mask=traj_mask
            )
            if use_wandb:
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
        if use_wandb:
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


def run_reinformer_main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=[ "Reinformer"], default="Reinformer")
    # parser.add_argument("--env", type=str, default="QLab")
    # parser.add_argument("--dataset", type=str, default="medium")
    # parser.add_argument("--num_eval_ep", type=int, default=10)
    # parser.add_argument("--max_eval_ep_len", type=int, default=1000)
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--context_len", type=int, default=CONTEXT_LEN)
    parser.add_argument("--n_blocks", type=int, default=N_BLOCKS)
    parser.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    parser.add_argument("--n_heads", type=int, default=N_HEADS)
    parser.add_argument("--dropout_p", type=float, default=DROPOUT_P)
    parser.add_argument("--grad_norm", type=float, default=GRAD_NORM)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--wd", type=float, default=WD)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--max_train_iters", type=int, default=MAX_TRAIN_ITERS)
    parser.add_argument("--num_updates_per_iter", type=int, default=NUM_UPDATES_PER_ITER) # 5000
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--init_temperature", type=float, default=INIT_TEMPERATURE)
    # use_wandb = False
    parser.add_argument("--use_wandb", action='store_true', default=True)
    args = parser.parse_args()

    resume: bool = os.path.exists("assets/models/latest_checkpoint.pt")

    if args.use_wandb:
        wandb.init(
            name="QLab", # + "-" + args.dataset,
            project="Reinformer",
            config=vars(args),
            resume=RESUME
        )

    experiment(vars(args))
