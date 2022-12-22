import argparse
from collections import deque
import pomdp_spaceship_env
import visdom
import numpy as np
import os
import time
from RL.ppo import *
from RL.policies import *
import torch
import time
import string
import torch.nn as nn
import torch.optim as optim
from colour import Color
import torch.nn.functional as F
from torch.distributions import Categorical
import cProfile, pstats, io
from pstats import SortKey
from line_profiler import LineProfiler
PROFILE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

min_in = torch.Tensor([0, 0, -np.pi/3, -np.pi/3]).to(DEVICE)
max_in = torch.Tensor([30, 30, np.pi/3, np.pi/3]).to(DEVICE)

def main():
    parser = argparse.ArgumentParser(description="Using A2C for solving Space Ship Sim")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_episodes", type=int, default=10000)
    parser.add_argument("--n_sims", type=int, default=64)  # 16
    parser.add_argument("--max_steps", type=int, default=400)  # 400
    parser.add_argument("--recurrent_seq_len",  default=50)
    parser.add_argument("--recurrent_layers",  default=2)
    parser.add_argument("--minibatch_size",  default=16)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--state_dim", type=int, default=521)
    parser.add_argument("--act_dim", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--plot_interval", type=int, default=2)
    parser.add_argument("--checkpoint", default="10000")
    # parser.add_argument("--checkpoint", default="5000")
    parser.add_argument("--live_plot", type=bool, default=True)
    parser.add_argument("--n_updates", type=int, default=8)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--entropy_coeff", type=float, default=0.005)
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--model_prefix",  default="models_rec/")

    args = parser.parse_args()
    sim_conf = pomdp_spaceship_env.Config()
    sim_conf.Viz = False
    sim_conf.PrintLevel = 0
    sim_conf.AutoReset = True
    sim_conf.DynamicGoals = False
    sim_conf.ShareEnvs = False
    sim_conf.NumObs = 60
    rewardF = pomdp_spaceship_env.RewardFunction()
    print("saving to" + os.path.abspath(args.model_prefix))
    rewardF.DeltaThrustAngle = 1.0
    rewardF.DeltaThrust = 1.0
    ac = LSTMActorCritic1DConv(args.state_dim, 32, args.act_dim, args.recurrent_layers)
    env = pomdp_spaceship_env.Env(sim_conf, args.n_sims, rewardF)
    ppo = PPO(ac, env, args, sim_conf)
    if PROFILE:
        prof = LineProfiler()
        lp_wrap = prof(ppo.train)
        lp_wrap()
        prof.print_stats()
    ppo.train()
    # ppo.test_multiple(["10000"])#, "1000", "500"])

    # ppo.test()<

if __name__ == "__main__":
    main()
