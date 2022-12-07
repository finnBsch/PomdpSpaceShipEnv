import argparse
from collections import deque
import visdom
import numpy as np
import time
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

profile_ = True
#if profile_:
#    pr = cProfile.Profile()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

min_in = torch.Tensor([0, 0, -np.pi/3, -np.pi/3]).to(DEVICE)
max_in = torch.Tensor([30, 30, np.pi/3, np.pi/3]).to(DEVICE)

class ActorCritic(nn.Module):
    """
implements both actor and critic in one model
"""

    def __init__(self, state_dim=11, action_dim=4):
        super(ActorCritic, self).__init__()
        # self.affine_0 = nn.Linear(state_dim, 128)

        self.conv = nn.Conv1d(1, 1, 30, 5)
        self.conv_c = nn.Conv1d(1, 1, 30, 5)
        # self.max_pool = nn.MaxPool1d(5)
        self.conv2 = nn.Conv1d(1, 1, 10, 2)
        self.conv2_c = nn.Conv1d(1, 1, 10, 2)
        # self.conv3 = nn.Conv1d(1, 1, 5, 2)
        self.affine1_a = nn.Linear(44 + 9, 64)
        self.affine2_a = nn.Linear(64, 64)
        self.action_head = nn.Linear(64, action_dim)

        self.affine1_c = nn.Linear(44 + 9, 100)
        self.affine2_c = nn.Linear(100, 50)
        self.affine3_c = nn.Linear(50, 25)
        self.value_head = nn.Linear(25, 1)
        self.log_std = nn.Parameter(torch.FloatTensor([0.5, 0.5, 0.5, 0.5]).log(), requires_grad=True)
        # self.affine_s3 = nn.Linear(16, 16)

        # actor's head
        # self.affine_a1 = nn.Linear(16, 16)


        # critic's head
        # self.affine_c1 = nn.Linear(16, 16)


        # action & reward buffer
        self.saved_actions = []
        self.rewards = []


    def forward(self, x):
        """
        forward of both actor and critic
        """
        # x = F.tanh(self.affine_1(x))
        x_state, x_conv = torch.split(x, [9, 512], dim=x.dim()-1)
        x_conv = x_conv.unsqueeze(x.dim()-1)
        if(x.dim() == 3):
            x_conv = torch.reshape(x_conv, (x.shape[0]*x.shape[1], 1, 512))

        x_conv_a = F.leaky_relu_(self.conv2(F.leaky_relu_(self.conv(x_conv*2 - 1))))
        x_conv_c = F.leaky_relu_(self.conv2_c(F.leaky_relu_(self.conv_c(x_conv*2 - 1))))
        # x_conv = F.leaky_relu_(self.conv2(x_conv))
        # x_conv = F.relu(self.conv3(x_conv))
        if(x.dim() == 3):
            x_conv_a = torch.reshape(x_conv_a, (x.shape[0], x.shape[1], 1, 44))
            x_conv_c = torch.reshape(x_conv_c, (x.shape[0], x.shape[1], 1, 44))
        if(x_state.shape[0] == 1):
            x_in_a = torch.concat((x_state, x_conv_a.squeeze().unsqueeze(0)), x.dim() - 1)
            x_in_c = torch.concat((x_state, x_conv_c.squeeze().unsqueeze(0)), x.dim() - 1)
        else:
            x_in_a = torch.concat((x_state, x_conv_a.squeeze()), x.dim() - 1)
            x_in_c = torch.concat((x_state, x_conv_c.squeeze()), x.dim() - 1)
        # x_in = F.relu(self.affine0(x_in))
        x_a = F.tanh(self.affine1_a(x_in_a))
        x_a = F.tanh(self.affine2_a(x_a))
        x_c = F.tanh(self.affine1_c(x_in_c))
        x_c = F.tanh(self.affine2_c(x_c))
        x_c = F.tanh(self.affine3_c(x_c))

        action_prob = F.tanh(self.action_head(x_a))
        state_values = self.value_head(x_c)
        return action_prob, state_values

class RolloutBuffer(object):
    def __init__(self, log_probs, observations, actions, n_envs, max_steps, advantages, returns, state_dim, act_dim):
        self.n_samples = n_envs * max_steps
        self.observations = torch.zeros((self.n_samples, state_dim)).to(DEVICE)
        self.actions = torch.zeros((self.n_samples, act_dim)).to(DEVICE)
        self.log_probs = torch.zeros((self.n_samples, )).to(DEVICE)  # TODO Check
        self.advantages = torch.zeros((self.n_samples, )).to(DEVICE)
        self.returns = torch.zeros((self.n_samples, )).to(DEVICE)
        fil_id = 0
        for i in range(log_probs.shape[0]):
            last_id = max_steps
            self.observations[fil_id:fil_id + last_id] = observations[i][:last_id]
            self.actions[fil_id:fil_id + last_id] = actions[i][:last_id]
            self.log_probs[fil_id:fil_id + last_id] = log_probs[i][:last_id]
            self.advantages[fil_id:fil_id + last_id] = advantages[i][:last_id]
            self.returns[fil_id:fil_id + last_id] = returns[i][:last_id]
            fil_id += last_id

    def get(self, batch_size=None):
        indices = np.random.permutation(self.n_samples)
        if batch_size is None:
            batch_size = self.n_samples

        start_idx = 0
        while start_idx < self.n_samples:
            batch_inds = indices[start_idx: start_idx + batch_size]
            data = (
                self.observations[batch_inds],
                self.actions[batch_inds],
                self.log_probs[batch_inds],
                self.advantages[batch_inds],
                self.returns[batch_inds]
            )
            yield data
            start_idx += batch_size

class PPO(nn.Module):
    def __init__(self, args):
        super(PPO, self).__init__()
        self.meansub = torch.FloatTensor([1, 1, 0, 0]).to(DEVICE)
        self.scale = torch.FloatTensor([15, 15, np.pi/3, np.pi/3]).to(DEVICE)
        self.gamma = args.gamma
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.plot_interval = args.plot_interval
        self.n_episodes = args.n_episodes
        self.n_sims = args.n_sims
        self.n_updates = args.n_updates
        self.clip = args.clip
        self.n_trajs = self.n_sims
        self.max_steps = args.max_steps
        self.dt = args.dt
        self.gae_lambda = args.gae_lambda
        self.max_grad_norm = args.max_grad_norm
        self.state_dim = args.state_dim
        self.act_dim = args.act_dim
        self.entropy_coeff = args.entropy_coeff
        self.live_plot = args.live_plot
        self.sim_conf = librlsimpy.Config()
        self.sim_conf.Viz = False
        self.sim_conf.PrintLevel = 0
        self.sim_conf.AutoReset = True
        self.sim_conf.DynamicGoals = False
        self.sim_conf.ShareEnvs = False

        self.simpool = librlsimpy.SpaceShip(self.sim_conf, self.n_sims)
        self.ac = ActorCritic(self.state_dim, self.act_dim).to(DEVICE)
        if args.checkpoint is not None:
            self.ac.load_state_dict(torch.load("models_rec/" + args.checkpoint))
        max_w = 0
        for name, param in self.ac.named_parameters():
            max_w = max(max_w, param.abs().max())
        print("Max weight: ", max_w)
        # self.cov_mat = torch.nn.Parameter(self.cov_mat_)
        self.optim = optim.Adam(self.ac.parameters(), lr=self.lr)
        self.alive_time = np.zeros((self.n_trajs, ))
        self.running_rew = np.zeros((self.n_trajs, ))
        self.running_steps = np.zeros((self.n_trajs, ))
        self.current_rew = np.zeros((self.n_trajs, ))
        self.entropies = []
        self.value_losses = []
        if self.live_plot:
            self.viz = visdom.Visdom()
            self.rew_window = self.viz.line(
                Y=np.zeros((1)),
                X=np.zeros((1)),
                opts=dict(xlabel='epoch',ylabel='Avg Cum. Rew',title='Reward over episodes',legend=['Reward']))
            self.step_window = self.viz.line(
                Y=np.zeros((1)),
                X=np.zeros((1)),
                opts=dict(xlabel='epoch',ylabel='Avg. Survive Time',title='Survive Time',legend=['Time in s']))
            self.loss_window = self.viz.line(
                Y=np.zeros((1)),
                X=np.zeros((1)),
                opts=dict(xlabel='epoch',ylabel='Avg. Value Loss',title='Value loss',legend=['Value Loss']))
            self.entropy_window = self.viz.line(
                Y=np.zeros((1)),
                X=np.zeros((1)),
                opts=dict(xlabel='epoch',ylabel='Avg. Entropy',title='Entropy',legend=['Entropy']))

    def select_action(self, acts_mean):
        action_std = torch.diag(self.ac.log_std.exp())
        dist = torch.distributions.MultivariateNormal(acts_mean, action_std) #scale_tril=self.scale_tril)
        action = dist.rsample()
        log_probs = dist.log_prob(action)
        # action = self.clip_actions(action)
        return action.detach(), log_probs.detach()


    """
    Runs simulations to collect data
    """
    def rollout(self):
        state = self.simpool.GetState().T
        terminal_id = np.full((self.n_trajs,), np.inf)
        rewards = np.zeros((self.n_trajs, self.max_steps))
        log_probs = torch.zeros((self.n_trajs, self.max_steps))
        entropy = torch.zeros((self.n_trajs,))
        values = torch.zeros((self.n_trajs, self.max_steps))
        all_obs = np.zeros((self.n_trajs, self.max_steps, self.state_dim))
        all_acts = np.zeros((self.n_trajs, self.max_steps, self.act_dim))
        episode_starts = np.zeros((self.n_trajs, self.max_steps))
        for step in range(self.max_steps):
            self.alive_time += 1
            action_probs, t_values = self.ac(torch.from_numpy(state).float().to(DEVICE))  # TODO instead of copying, make the state array writeabale
            actions, t_log_probs = self.select_action(action_probs)
            control_ins = actions
            all_obs[:, step, :] = state
            all_acts[:, step, :] = actions.cpu()
            self.simpool.SetControl(self.clip_actions((control_ins + self.meansub)*self.scale).cpu().detach().numpy())
            done = self.simpool.Step(self.dt)
            t_done = self.simpool.GetAgentDone().T
            episode_starts[:, step] = t_done[:, 0]
            t_rewards = self.simpool.GetReward().T
            self.current_rew += t_rewards[:, 0]
            self.running_rew = np.where(t_done, self.running_rew*0.8 + 0.2*self.current_rew, self.running_rew)
            self.current_rew = np.where(t_done, 0, self.current_rew)
            self.running_steps = np.where(t_done, self.running_steps*0.8 + 0.2*self.alive_time, self.running_steps)
            self.alive_time = np.where(t_done, 0, self.alive_time)

            log_probs[:, step] = t_log_probs
            rewards[:, step] = t_rewards[:, 0]

            values[:, step] = t_values[:, 0]
            done_ids = np.where(t_done == True, step, np.inf)
            terminal_id = np.minimum(terminal_id, done_ids[:, 0])
            state = self.simpool.GetState().T
        dones = t_done
        _, last_values = self.ac(torch.from_numpy(state.copy()).float().to(DEVICE))
        terminal_id = np.minimum(terminal_id, self.max_steps-1)
        all_obs = torch.FloatTensor(all_obs).to(DEVICE)
        all_acts = torch.FloatTensor(all_acts).to(DEVICE)
        return step, rewards, log_probs, entropy, terminal_id, all_obs, all_acts, values, dones, last_values, episode_starts

    def clip_actions(self, actions):
        return torch.clamp(actions, min_in, max_in)

    def evaluate(self, all_obs, all_acts):
        acts_mean, V = self.ac(all_obs)
        action_std = torch.diag(self.ac.log_std.exp())
        dist = torch.distributions.MultivariateNormal(acts_mean, action_std)# scale_tril=self.scale_tril)
        log_probs = dist.log_prob(all_acts)
        return V, log_probs, dist.entropy()

    def compute_qvals(self, rewards, terminal_ids):
        Qvals = np.zeros_like(rewards)
        for i in range(self.n_trajs):
            n_samples = int(terminal_ids[i]) + 1
            Qval = 0
            for t in reversed(range(n_samples)):
                Qval = rewards[i, t] + self.gamma*Qval
                Qvals[i, t] = Qval
        return Qvals

    def compute_returns(self, rewards, values, terminal_ids, dones, last_values, episode_starts):
        advantages = np.zeros_like(rewards)
        for i in range(self.n_trajs):
            n_samples = self.max_steps
            last_gae_lam = 0
            # TODO make vectorized with mask? (rewards = rewards*mask, torch.where intead of if t == n_samples-1)
            for t in reversed(range(n_samples)):
                if t == n_samples-1:
                    next_non_terminal = 1.0 - dones[i]
                    next_vals = last_values[i]
                else:
                    next_non_terminal = 1.0 - episode_starts[i, t]
                    next_vals = values[i, t+1]
                delta = rewards[i, t] + self.gamma * next_vals * next_non_terminal - values[i, t]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                advantages[i, t] = last_gae_lam
        returns = advantages + values
        return returns

    def train(self):
        avg_steps_alive = []
        avg_cum_reward = []
        total_num_steps = 0
        self.simpool.Reset()
        self.simpool.Step(0.001)  # TODO: Find easier way to init
        for e in range(self.n_episodes):
            # lp = LineProfiler()
            # lp_wrapper = lp(self.rollout)
            # lp_wrapper()
            # lp.print_stats()

            step, rewards, log_probs, entropy, terminal_id, all_obs, all_acts, values, dones, last_values, episode_starts = self.rollout()
            values = values.cpu().detach().numpy()
            last_values = last_values.cpu().detach().numpy()
            all_rtgs = self.compute_returns(rewards, values, terminal_id, dones, last_values, episode_starts)
            all_rtgs = torch.FloatTensor(all_rtgs).to(DEVICE)
            V, _, entropy = self.evaluate(all_obs, all_acts)
            V = V.squeeze()
            advantages = all_rtgs - V.detach()
            # for i in range(self.n_trajs):
            #     last_id = int(terminal_id[i]) + 1
            #     a_mean = advantages[i][:last_id].mean()
            #     a_std = advantages[i][:last_id].std()
            #     advantages[i] = (advantages[i] - a_mean)/(a_std + 1e-10)
            # mask = np.repeat(np.indices((rewards.shape[1], )), rewards.shape[0], 0) <= terminal_id[:, np.newaxis]
            rew_summed = np.sum(rewards, 1)
            m_rew = np.mean(rew_summed)
            avg_cum_reward.append(m_rew)
            avg_steps_alive.append(np.mean(terminal_id))
            rolloutbuffer = RolloutBuffer(log_probs, all_obs, all_acts, self.n_trajs, self.max_steps, advantages, all_rtgs, self.state_dim, self.act_dim)
            total_num_steps += rolloutbuffer.n_samples
            for _ in range(self.n_updates):
                for samples in rolloutbuffer.get(512):
                    observations, actions, log_probs, advantages, returns = samples
                    advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)
                    V, curr_log_probs, entropy = self.evaluate(observations, actions)
                    V = V.squeeze()
                    ratios = torch.exp(curr_log_probs - log_probs)
                    surr1 = ratios*advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    if V.dim() == 0:
                        V = V.unsqueeze(0)
                    critic_loss = nn.MSELoss()(V, returns)
                    self.value_losses.append(critic_loss.cpu().detach())
                    self.entropies.append(entropy.mean().cpu().detach())
                    loss = actor_loss + critic_loss - self.entropy_coeff * entropy.mean()
                    self.optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                    self.optim.step()
            if (e+1)%self.plot_interval == 0:
                if self.live_plot:
                    self.viz.line(X=[e+1], Y=[np.mean(self.running_rew)], win=self.rew_window, update='append')
                    self.viz.line(X=[e+1], Y=[np.mean(self.running_steps)*self.dt], win=self.step_window, update='append')
                    self.viz.line(X=[e+1], Y=[np.mean(self.entropies)], win=self.entropy_window, update='append')
                    self.viz.line(X=[e+1], Y=[np.mean(self.value_losses)], win=self.loss_window, update='append')
                    self.entropies.clear()
                    self.value_losses.clear()
            if (e+1)%self.log_interval == 0:
                max_w = 0
                print(self.ac.log_std.exp())
                for name, param in self.ac.named_parameters():
                    max_w = max(max_w, param.max())
                print("Max weight: ", max_w)

                torch.save(self.ac.state_dict(), "models_rec/" + str(e+1))

                print("Episode {}, Avg. Reward {}, Avg Steps {}, Total Steps {}, longest alive currently {}".format(e+1, np.mean(self.running_rew), np.mean(self.running_steps), total_num_steps, np.max(self.alive_time)))
    def test(self):
        sim_conf = librlsimpy.Config()
        sim_conf.Viz = True
        sim_conf.PrintLevel = 0
        sim_conf.AutoReset = True
        sim_conf.DynamicGoals = False
        sim_conf.ShareEnvs = False

        sim = librlsimpy.SpaceShip(sim_conf, 2)
        # sim = librlsimpy.SpaceShip(["#5000", "x"], True, True, 0, 2, False)
        # sim.AddShip("External")
        test_t = 10
        t = 0
        sim.Reset()
        t0 = time.time()
        # state = np.zeros((self.state_dim, self.state_dim))
        while True:
            state = sim.GetState().T
            action_probs, state_val = self.ac(torch.from_numpy(state).float().to(DEVICE))
            control_in = self.clip_actions(action_probs)
            # sim.SetControl(control_in.cpu().detach().numpy())
            done = sim.Step()
            # print(sim.GetAgentDone())
            if done:
                # sim.Reset()
                pass

    def test_multiple(self, versions):
        names = []
        nets = []
        for i in range(len(versions)):
            names.append("Ep. " + versions[i])
            nets.append(ActorCritic(self.state_dim, self.act_dim).to(DEVICE))
            nets[-1].load_state_dict(torch.load("models/" + versions[i], map_location=DEVICE))
        self.ac.log_std = nets[-1].log_std
        print(nets[-1].log_std.exp())
        sim_conf = librlsimpy.Config()
        sim_conf.Viz = True
        sim_conf.PrintLevel = 0
        sim_conf.AutoReset = True
        sim_conf.DynamicGoals = False
        sim_conf.ShareEnvs = False
        sim_conf.NumObs = 50

        sim = librlsimpy.SpaceShip(sim_conf, len(versions), names)
        sim.Reset()
        sim.Step(0.0)
        t0 = time.time()
        last_d = False
        while True:
            control_in = torch.zeros((len(versions), 4)).to(DEVICE)
            state = sim.GetState().T
            # print(state[0][:2])
            for i in range(len(versions)):
                action_probs, state_val = nets[i](torch.from_numpy(state).float().to(DEVICE))
                control_in[i] = self.clip_actions((action_probs + self.meansub)*self.scale)[i]
            # print(state)
            # control_in, t_log_probs = self.select_action(control_in)

            sim.SetControl(control_in.cpu().detach().numpy())
            done = sim.Step()
            rew = sim.GetReward()
            if(rew[0,0] == -100):
                print('ouch')
                if last_d:
                    print("#############")
                last_d = True
            else:
                last_d = False
            if done or time.time() - t0 > 10:
                # sim.Reset()
                pass
                # t0 = time.time()



# TODO CHECK ACTION CLIPPING!!, ADD ENTROPY


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
    parser.add_argument("--live_plot", type=bool, default=False)
    parser.add_argument("--n_updates", type=int, default=8)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--entropy_coeff", type=float, default=0.005)
    parser.add_argument("--max_grad_norm", type=float, default=1)
    args = parser.parse_args()
    ppo = PPO(args)
    # ppo.train()
    ppo.test_multiple(["7500"])#, "1000", "500"])
    # ppo.test()


if __name__ == "__main__":
    main()
