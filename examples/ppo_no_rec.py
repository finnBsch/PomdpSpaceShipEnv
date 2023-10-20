import argparse
from collections import deque
import pomdp_spaceship_env
import visdom
import numpy as np

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(description="Using A2C for solving Space Ship Sim")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_episodes", type=int, default=7500)
    parser.add_argument("--n_sims", type=int, default=16) # 128
    parser.add_argument("--max_steps", type=int, default=400) # 400
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--state_dim", type=int, default=521)
    parser.add_argument("--act_dim", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--plot_interval", type=int, default=5)
    parser.add_argument("--checkpoint", default=None)
    # parser.add_argument("--checkpoint", default="5000")
    parser.add_argument("--live_plot", type=bool, default=True)
    parser.add_argument("--n_updates", type=int, default=8)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--entropy_coeff", type=float, default=0.005)
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--model_prefix",  default="models_no_rec/")
    parser.add_argument("--minibatch_size",  default=512)
    args = parser.parse_args()
    sim_conf = pomdp_spaceship_env.Config()
    sim_conf.Viz = False
    sim_conf.PrintLevel = 0
    sim_conf.AutoReset = True
    sim_conf.DynamicGoals = False
    sim_conf.ShareEnvs = False
    sim_conf.NumObs = 0
    ac = ActorCritic1DConv(args.state_dim, args.act_dim)
    env = pomdp_spaceship_env.Env(sim_conf, args.n_sims)
    ppo = PPO(ac, env, args, sim_conf)
    ppo.train()
    # ppo.test_multiple(["15000"])#, "1000", "500"])
    # ppo.test()

if __name__ == "__main__":
    main()
