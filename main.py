import argparse
import logging
import os

# import d4rl
import wandb

from system.settings import CONTEXT_LEN, BATCH_SIZE
from system.coordinator import experiment, control, convert


if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=[ "trainer", "converter", "car"], default="trainer")
    parser.add_argument("--model_type", choices=[ "Reinformer"], default="Reinformer")
    # parser.add_argument("--env", type=str, default="QLab")
    # parser.add_argument("--dataset", type=str, default="medium")
    # parser.add_argument("--num_eval_ep", type=int, default=10)
    # parser.add_argument("--max_eval_ep_len", type=int, default=1000)
    parser.add_argument("--dataset_dir", type=str, default="assets")
    parser.add_argument("--context_len", type=int, default=CONTEXT_LEN)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--grad_norm", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--max_train_iters", type=int, default=10)
    parser.add_argument("--num_updates_per_iter", type=int, default=5000) # 5000
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--init_temperature", type=float, default=0.1)
    # use_wandb = False
    parser.add_argument("--use_wandb", action='store_true', default=True)
    args = parser.parse_args()

    if args.run == "trainer":
        if args.use_wandb:
            wandb.init(
                name="QLab", # + "-" + args.dataset,
                project="Reinformer",
                config=vars(args)
            )

        experiment(vars(args))
    elif args.run == "converter":
        convert()
    elif args.run == "car":
        control()
