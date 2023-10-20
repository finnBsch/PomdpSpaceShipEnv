import pomdp_spaceship_env
import visdom
import numpy as np
from buffers import *
from policies import *

import torch
import time

import torch.nn as nn
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PPO():
    def __init__(self, ac, env, args, sim_conf):
        super(PPO, self).__init__()
        self.meansub = torch.FloatTensor([1, 1, 0, 0]).to(DEVICE)
        self.scale = torch.FloatTensor([15, 15, np.pi / 3, np.pi / 3]).to(DEVICE)
        self.gamma = args.gamma
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.plot_interval = args.plot_interval
        self.minibatch_size = args.minibatch_size
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
        self.model_prefix = args.model_prefix
        self.env = env
        self.min_in = torch.Tensor(self.env.GetMinIn()).to(DEVICE)
        self.max_in = torch.Tensor(self.env.GetMaxIn()).to(DEVICE)
        self.ac = ac
        self.actor_cell = None
        self.critic_cell = None
        self.sim_conf = sim_conf
        if self.ac.__class__.__name__ == "ActorCritic1DConv":
            self.is_recurrent = False
            self.buffer = RollOutBuffer(self.n_sims, self.max_steps, self.state_dim, self.act_dim,
                                        self.gamma, self.gae_lambda)
        else:
            self.recurrent_seq_len = args.recurrent_seq_len
            self.is_recurrent = True
            self.recurrent_layers = args.recurrent_layers
            self.buffer = RecurrentRolloutBuffer(self.n_sims, self.max_steps, self.state_dim, self.act_dim, self.gamma,
                                                 self.gae_lambda, self.ac.hs_dim, self.ac.cs_dim, self.recurrent_layers, self.recurrent_seq_len)
        self.ac.to(DEVICE)
        self.checkpoint = args.checkpoint
        if self.checkpoint is None:
            self.checkpoint = 0
        if args.checkpoint is not None:
            self.ac.load_state_dict(torch.load(self.model_prefix + args.checkpoint, map_location=DEVICE))
        # self.ac.log_std.detach()
        # self.ac.log_std.requires_grad = False
        # self.ac.log_std[-1] = self.ac.log_std[-2]
        # self.ac.log_std.requires_grad = True
        max_w = 0
        for name, param in self.ac.named_parameters():
            max_w = max(max_w, param.abs().max())
        print("Max weight: ", max_w)
        # self.cov_mat = torch.nn.Parameter(self.cov_mat_)
        self.optim = optim.Adam(self.ac.parameters(), lr=self.lr)
        self.alive_time = np.zeros((self.n_trajs,))
        self.running_rew = np.zeros((self.n_trajs,))
        self.running_steps = np.zeros((self.n_trajs,))
        self.current_rew = np.zeros((self.n_trajs,))
        self.entropies = []
        self.value_losses = []
        self.actor_losses = []
        self.kl_divs = []
        self.n_goals = []
        if self.live_plot:
            self.viz = visdom.Visdom()
            self.rew_window = self.viz.line(
                Y=np.zeros((1)),
                X=np.zeros((1)),
                opts=dict(xlabel='epoch', ylabel='Avg Cum. Rew', title='Reward over episodes', legend=['Reward']))
            self.step_window = self.viz.line(
                Y=np.zeros((1)),
                X=np.zeros((1)),
                opts=dict(xlabel='epoch', ylabel='Avg. Survive Time', title='Survive Time', legend=['Time in s']))
            self.loss_window = self.viz.line(
                Y=np.zeros((1)),
                X=np.zeros((1)),
                opts=dict(xlabel='epoch', ylabel='Avg. Value Loss', title='Value loss', legend=['Value Loss']))
            self.entropy_window = self.viz.line(
                Y=np.zeros((1)),
                X=np.zeros((1)),
                opts=dict(xlabel='epoch', ylabel='Avg. Entropy', title='Entropy', legend=['Entropy']))
            self.actor_loss_window = self.viz.line(
                Y=np.zeros((1)),
                X=np.zeros((1)),
                opts=dict(xlabel='epoch', ylabel='Avg. Actor Loss', title='Actor loss', legend=['Actor Loss']))
            self.n_goal_window = self.viz.line(
                Y=np.zeros((1)),
                X=np.zeros((1)),
                opts=dict(xlabel='epoch', ylabel='Number', title='Avg. times goal reached', legend=['n_goals']))
            self.kl_window = self.viz.line(
                Y=np.zeros((1)),
                X=np.zeros((1)),
                opts=dict(xlabel='epoch', ylabel='KL Div', title='Avg. KL Divs', legend=['kl_div']))

    def select_action(self, acts_mean):
        dist = torch.distributions.MultivariateNormal(acts_mean, scale_tril=self.scale_tril)
        action = dist.rsample()
        log_probs = dist.log_prob(action)
        # action = self.clip_actions(action)
        return action.detach(), log_probs.detach()

    """
    Runs simulations to collect data
    """

    def rollout(self):
        self.scale_tril = torch.linalg.cholesky(torch.diag(self.ac.log_std.exp()))
        state = self.env.GetState()
        terminal_id = np.full((self.n_trajs,), np.inf)
        last_t_done = torch.zeros((self.n_trajs, )).to(DEVICE)
        if self.is_recurrent:
            self.ac.get_init_state(self.n_sims)
        if self.is_recurrent and self.buffer.is_done:
            self.ac.warm_up_hidden_state(self.buffer.observations, self.buffer.last_of_episodes)
        self.buffer.reset()
        for step in range(self.max_steps):
            self.alive_time += 1
            if self.is_recurrent:
                a_hs1 = self.ac.hidden_cell_a[0].detach()
                a_hs2 = self.ac.hidden_cell_a[1].detach()
                c_hs1 = self.ac.hidden_cell_c[0].detach()
                c_hs2 = self.ac.hidden_cell_c[1].detach()
                action_probs, t_values = self.ac(
                    torch.from_numpy(state).float().to(DEVICE), last_t_done)
            else:
                action_probs, t_values = self.ac(
                    torch.from_numpy(state).float().to(DEVICE))
            actions, t_log_probs = self.select_action(action_probs)
            control_ins = actions
            self.env.SetControl(self.clip_actions((control_ins + self.meansub) * self.scale).cpu().detach().numpy())
            self.env.Step_dt(self.dt)
            t_done = self.env.GetAgentDone()
            t_rewards = self.env.GetReward()
            self.n_goals.append((t_rewards==100).sum())
            self.current_rew += t_rewards
            self.running_rew = np.where(t_done, self.running_rew * 0.8 + 0.2 * self.current_rew, self.running_rew)
            self.current_rew = np.where(t_done, 0, self.current_rew)
            self.running_steps = np.where(t_done, self.running_steps * 0.8 + 0.2 * self.alive_time, self.running_steps)
            self.alive_time = np.where(t_done, 0, self.alive_time)

            if self.is_recurrent:
                self.buffer.add_sample(t_log_probs, state, t_rewards, t_values.detach(), t_done, actions,
                                       a_hs1, a_hs2,
                                       c_hs1, c_hs2)
            else:
                self.buffer.add_sample(t_log_probs, state, t_rewards, t_values.detach(), t_done, actions)
            done_ids = np.where(t_done == True, step, np.inf)
            terminal_id = np.minimum(terminal_id, done_ids)
            state = self.env.GetState()
            last_t_done = torch.FloatTensor(t_done).to(DEVICE)
        dones = t_done
        if self.is_recurrent:
            _, last_values = self.ac(torch.from_numpy(state.copy()).float().to(DEVICE), last_t_done)
        else:
            _, last_values = self.ac(torch.from_numpy(state.copy()).float().to(DEVICE))
        terminal_id = np.minimum(terminal_id, self.max_steps - 1)
        return step, terminal_id, dones, last_values

    def clip_actions(self, actions):
        return torch.clamp(actions, self.min_in, self.max_in)

    def evaluate(self, all_obs, all_acts):
        acts_mean, V = self.ac(all_obs)
        action_std = torch.diag(self.ac.log_std.exp())
        dist = torch.distributions.MultivariateNormal(acts_mean, action_std)  # scale_tril=self.scale_tril)
        log_probs = dist.log_prob(all_acts)
        return V, log_probs, dist.entropy()

    def evaluate_rec(self, all_obs, all_acts, act_hxs, act_cxs, crit_hxs, crit_cxs, ep_starts):
        acts_mean, V = self.ac(all_obs, None, act_hxs, act_cxs, crit_hxs, crit_cxs, self.buffer.true_sequence_length)
        dist = torch.distributions.MultivariateNormal(acts_mean, scale_tril=self.scale_tril)
        log_probs = dist.log_prob(all_acts)
        return V, log_probs, dist.entropy()

    @staticmethod
    def _masked_mean(tensor:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        return (tensor * mask).sum() / torch.clamp(mask.float().sum(), min=1.0)


    def train(self):
        avg_steps_alive = []
        avg_cum_reward = []
        total_num_steps = 0
        self.env.Reset()
        self.env.Step_dt(0.001)  # TODO: Find easier way to init
        for e in range(self.n_episodes):
            print("train start")
            step, terminal_id, dones, last_values = self.rollout()
            self.buffer.finish_buffer(torch.FloatTensor(dones).to(DEVICE), last_values.detach())
            rew_summed = torch.sum(self.buffer.rewards, 1)
            m_rew = torch.mean(rew_summed)
            avg_cum_reward.append(m_rew)
            avg_steps_alive.append(np.mean(terminal_id))
            total_num_steps += self.n_sims * self.max_steps
            for _ in range(self.n_updates):
                for samples in self.buffer.get(self.minibatch_size):
                    self.scale_tril = torch.linalg.cholesky(torch.diag(self.ac.log_std.exp()))
                    if self.is_recurrent:
                        observations = samples["observations"]
                        actions = samples["actions"]
                        values = samples["values"]
                        returns = samples["returns"]
                        log_probs = samples["log_prob"]
                        advantages = samples["advantages"]
                        ep_starts = samples["last_of_episodes"]
                        mask = samples["loss_mask"]
                        a_hxs = samples["a_hxs"]
                        a_cxs = samples["a_cxs"]
                        c_hxs = samples["c_hxs"]
                        c_cxs = samples["c_cxs"]

                    else:
                        observations, actions, log_probs, advantages, returns = samples
                    if self.is_recurrent:
                        V, curr_log_probs, entropy = self.evaluate_rec(observations, actions, a_hxs, a_cxs, c_hxs, c_cxs, ep_starts)
                        b_mask = mask > 0
                        # TODO normalizing along sequence?
                        advantages = (advantages - advantages[b_mask].mean()) / (advantages[b_mask].std() + 1e-8)
                    else:
                        V, curr_log_probs, entropy = self.evaluate(observations, actions)
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    V = V.squeeze()
                    ratios = torch.exp(curr_log_probs - log_probs)
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
                    if V.dim() == 0:
                        V = V.unsqueeze(0)
                    if self.is_recurrent:
                        critic_loss = (V - returns)**2
                        critic_loss = self._masked_mean(critic_loss, mask)
                        entropy_loss = self._masked_mean(entropy, mask)
                        actor_loss = (-torch.min(surr1, surr2))
                        actor_loss = self._masked_mean(actor_loss, mask)
                    else:
                        critic_loss = nn.MSELoss()(V, returns)
                        entropy_loss = torch.mean(entropy)
                        actor_loss = (-torch.min(surr1, surr2)).mean()
                    self.value_losses.append(critic_loss.cpu().detach())
                    self.actor_losses.append(actor_loss.cpu().detach())
                    self.entropies.append(entropy_loss.mean().cpu().detach())
                    with torch.no_grad():
                        log_ratio = curr_log_probs - log_probs
                        if self.is_recurrent:
                            approx_kl_div = torch.mean(((torch.exp(log_ratio) - 1) - log_ratio)[b_mask]).cpu().numpy()
                        else:
                            approx_kl_div = torch.mean(((torch.exp(log_ratio) - 1) - log_ratio)).cpu().numpy()
                        self.kl_divs.append(approx_kl_div)
                    loss = actor_loss + critic_loss - self.entropy_coeff * entropy_loss
                    self.optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                    self.optim.step()
            print("train end")
            if (e + 1) % self.plot_interval == 0:
                print(self.ac.log_std.exp())
                if self.live_plot:
                    self.viz.line(X=[e + 1], Y=[np.mean(self.running_rew)], win=self.rew_window, update='append')
                    self.viz.line(X=[e + 1], Y=[np.mean(self.running_steps) * self.dt], win=self.step_window,
                                  update='append')
                    self.viz.line(X=[e + 1], Y=[np.mean(self.entropies)], win=self.entropy_window, update='append')
                    self.viz.line(X=[e + 1], Y=[np.mean(self.value_losses)], win=self.loss_window, update='append')
                    self.viz.line(X=[e + 1], Y=[np.mean(self.actor_losses)], win=self.actor_loss_window, update='append')
                    self.viz.line(X=[e + 1], Y=[np.sum(self.n_goals)/self.plot_interval], win=self.n_goal_window, update='append')
                    self.viz.line(X=[e + 1], Y=[np.mean(self.kl_divs)], win=self.kl_window, update='append')
                    self.n_goals.clear()
                    self.entropies.clear()
                    self.value_losses.clear()
                    self.actor_losses.clear()
                    self.kl_divs.clear()
            if (e + 1) % self.log_interval == 0:
                max_w = 0
                print(self.ac.log_std.exp())
                for name, param in self.ac.named_parameters():
                    max_w = max(max_w, param.max())
                print("Max weight: ", max_w)

                torch.save(self.ac.state_dict(), self.model_prefix + str(e + 1  + int(self.checkpoint)))

                print(
                    "Episode {}, Avg. Reward {}, Avg Steps {}, Total Steps {}, longest alive currently {}".format(e + 1,
                                                                                                                  np.mean(
                                                                                                                      self.running_rew),
                                                                                                                  np.mean(
                                                                                                                      self.running_steps),
                                                                                                                  total_num_steps,
                                                                                                                  np.max(
                                                                                                                      self.alive_time)))

    def test_multiple(self, versions):
        names = []
        nets = []
        for i in range(len(versions)):
            names.append("Ep. " + versions[i])
            if self.is_recurrent:
                nets.append(LSTMActorCritic1DConv(self.state_dim, 512, self.act_dim, self.recurrent_layers).to(DEVICE))
            else:
                nets.append(ActorCritic1DConv(self.state_dim, self.act_dim).to(DEVICE))
            nets[-1].load_state_dict(torch.load(self.model_prefix + versions[i], map_location=DEVICE))
        sim_conf = pomdp_spaceship_env.Config()
        sim_conf.Viz = True
        sim_conf.PrintLevel = 0
        sim_conf.AutoReset = True
        sim_conf.DynamicGoals = self.sim_conf.DynamicGoals
        sim_conf.ShareEnvs = False
        sim_conf.NumObs = self.sim_conf.NumObs
        sim_conf.ResX = int(1920/2)
        sim_conf.ResY = int(1080/2)

        sim = pomdp_spaceship_env.Env(sim_conf, len(versions), pomdp_spaceship_env.RewardFunction(), names)
        sim.Reset()
        sim.Step_dt(0.0)
        t0 = time.time()
        last_d = False
        if self.is_recurrent:
            last_t_done = torch.zeros((len(versions), )).to(DEVICE)
        while True:
            control_in = torch.zeros((len(versions), 4)).to(DEVICE)
            state = sim.GetState()
            # print(state[0][:2])
            for i in range(len(versions)):
                if self.is_recurrent:
                    action_probs, state_val = nets[i](torch.from_numpy(state).float().to(DEVICE), last_t_done)
                else:
                    action_probs, state_val = nets[i](torch.from_numpy(state).float().to(DEVICE))
                control_in[i] = self.clip_actions((action_probs + self.meansub) * self.scale)[i]
            # print(state)
            # control_in, t_log_probs = self.select_action(control_in)

            sim.SetControl(control_in.cpu().detach().numpy())
            done = sim.Step()
            rew = sim.GetReward()
            if self.is_recurrent:
                last_t_done = torch.FloatTensor(sim.GetAgentDone().T).to(DEVICE)

            if (rew[0] == -100):
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
