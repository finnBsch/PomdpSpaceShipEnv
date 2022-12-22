from typing import NamedTuple, Tuple
from functools import partial
from dataclasses import asdict
import torch.nn as nn
import numpy as np
import torch

# Collection for buffers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RollOutBuffer(object):
    def __init__(self, num_envs, num_steps, state_dim, act_dim, gamma, gae_lambda):
        self.is_done = False
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_samples = num_envs * num_steps
        self.sample_pointer = 0
        self.observations = torch.zeros((self.num_envs, self.num_steps, self.state_dim)).to(DEVICE)
        self.log_probs = torch.zeros((self.num_envs, self.num_steps)).to(DEVICE)
        self.advantages_ = torch.zeros((self.num_envs, self.num_steps)).to(DEVICE)
        self.actions = torch.zeros((self.num_envs, self.num_steps, self.act_dim)).to(DEVICE)
        self.rewards = torch.zeros((self.num_envs, self.num_steps)).to(DEVICE)
        self.values = torch.zeros((self.num_envs, self.num_steps)).to(DEVICE)
        self.returns = torch.zeros((self.num_envs, self.num_steps)).to(DEVICE)
        self.last_of_episodes = torch.zeros((self.num_envs, self.num_steps)).to(DEVICE)

    def add_sample(self, log_prob, observation, rewards, value, last_of_episode, action):
        self.observations[:, self.sample_pointer] = torch.FloatTensor(observation)
        self.log_probs[:, self.sample_pointer] = log_prob.to(DEVICE)
        self.actions[:, self.sample_pointer] = action.to(DEVICE)
        self.rewards[:, self.sample_pointer] = torch.FloatTensor(rewards)
        self.values[:, self.sample_pointer] = value[:, 0].to(DEVICE)
        self.last_of_episodes[:, self.sample_pointer] = torch.FloatTensor(last_of_episode)
        self.sample_pointer += 1

    def finish_buffer(self, dones, last_values):
        self.is_done = True
        n_samples = self.num_steps
        last_gae_lam = torch.zeros((self.num_envs, )).to(DEVICE)
        # TODO make vectorized with mask? (rewards = rewards*mask, torch.where intead of if t == n_samples-1)
        for t in reversed(range(n_samples)):
            if t == n_samples - 1:
                next_non_terminal = 1.0 - dones
                next_vals = last_values.squeeze()
            else:
                next_non_terminal = 1.0 - self.last_of_episodes[:, t]
                next_vals = self.values[:, t + 1]
            delta = self.rewards[:, t] + self.gamma * next_vals * next_non_terminal - self.values[:, t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages_[:, t] = last_gae_lam
        self.returns = self.advantages_ + self.values
        self.last_of_episodes[:, -1] = 1

    def reset(self):
        self.is_done = False
        self.sample_pointer = 0
        self.observations = torch.zeros_like(self.observations).to(DEVICE)
        self.log_probs = torch.zeros_like(self.log_probs).to(DEVICE)
        self.advantages_ = torch.zeros_like(self.advantages_).to(DEVICE)
        self.actions = torch.zeros_like(self.actions).to(DEVICE)
        self.rewards = torch.zeros_like(self.rewards).to(DEVICE)
        self.values = torch.zeros_like(self.values).to(DEVICE)
        self.returns = torch.zeros_like(self.returns).to(DEVICE)
        self.last_of_episodes = torch.zeros_like(self.last_of_episodes).to(DEVICE)

    def get(self, batch_size=None):
        indices = np.random.permutation(self.num_samples)
        if batch_size is None:
            batch_size = self.num_samples

        start_idx = 0
        observations = self.observations.view((self.num_samples, self.state_dim))
        actions = self.actions.view((self.num_samples, self.act_dim))
        log_probs = self.log_probs.view((self.num_samples,))
        advantages = self.advantages_.view((self.num_samples,))
        returns = self.returns.view((self.num_samples,))
        trajectory_ids = self.last_of_episodes.view((self.num_samples, )).nonzero()[:, 0] # Ids of the last state of a trajectory!
        while start_idx < self.num_samples:
            # sequences = indices[start_idx: start_idx + batch_size]
            # sequence_end_ids = trajectory_ids[np.digitize(sequences, bins=trajectory_ids)]
            batch_inds = indices[start_idx: start_idx + batch_size]
            data = (
                observations[batch_inds],
                actions[batch_inds],
                log_probs[batch_inds],
                advantages[batch_inds],
                returns[batch_inds]
            )
            yield data
            start_idx += batch_size


class RecurrentRolloutBuffer(RollOutBuffer):
    '''
    Extends the normal rollout buffer to account for recurrent networks. Requires different sampling and hidden state info
    '''
    def __init__(self, num_envs, num_steps, state_dim, act_dim, gamma, gae_lambda, hs_dim, cs_dim, lstm_layers, recurrent_seq_len):
        super().__init__(num_envs, num_steps, state_dim, act_dim, gamma, gae_lambda)
        self.recurrent_seq_len = recurrent_seq_len
        self.hs_dim = hs_dim
        self.cs_dim = cs_dim
        self.n_lstm_layers = lstm_layers
        self.actor_hxs = torch.zeros((self.num_envs, self.num_steps, self.n_lstm_layers, self.hs_dim)).to(DEVICE)
        self.actor_cxs = torch.zeros((self.num_envs, self.num_steps, self.n_lstm_layers, self.cs_dim)).to(DEVICE)
        self.critic_hxs = torch.zeros((self.num_envs, self.num_steps, self.n_lstm_layers, self.hs_dim)).to(DEVICE)
        self.critic_cxs = torch.zeros((self.num_envs, self.num_steps, self.n_lstm_layers, self.cs_dim)).to(DEVICE)

    def add_sample(self, log_prob, observation, rewards, value, last_of_episode, action, actor_hs, actor_cs, critic_hs, critic_cs):
        # Extend normal samples to contain hidden states
        self.actor_hxs[:, self.sample_pointer] = actor_hs.transpose(0, 1)
        self.actor_cxs[:, self.sample_pointer] = actor_cs.transpose(0, 1)
        self.critic_hxs[:, self.sample_pointer] = critic_hs.transpose(0, 1)
        self.critic_cxs[:, self.sample_pointer] = critic_cs.transpose(0, 1)
        super().add_sample(log_prob, observation, rewards, value, last_of_episode, action)

    def pad_sequence(self, sequence, target_length:int):
        delta_length = target_length - len(sequence)

        if delta_length <= 0:
            return sequence

        if len(sequence.shape) > 1:
            padding = torch.zeros(((delta_length,) + sequence.shape[1:]), dtype=sequence.dtype)
        else:
            padding = torch.zeros(delta_length, dtype=sequence.dtype)
        return torch.cat((sequence.to(DEVICE), padding.to(DEVICE)), axis=0)

    def finish_buffer(self, dones, last_values):
        super().finish_buffer(dones, last_values)
        samples = {
            "actions":self.actions,
            "values":self.values,
            "log_prob":self.log_probs,
            "observations": self.observations,
            "advantages": self.advantages_,
            "loss_mask":torch.ones((self.num_envs, self.num_steps), dtype=torch.float32).to(DEVICE),
            "returns":self.returns,
            "last_of_episodes":self.last_of_episodes
        }
        samples["a_hxs"] = self.actor_hxs
        samples["a_cxs"] = self.actor_cxs
        samples["c_hxs"] = self.critic_hxs
        samples["c_cxs"] = self.critic_cxs
        # List of Ids that end episodes
        episode_done_ids = []
        for e in range(self.num_envs):
            episode_done_ids.append(list(self.last_of_episodes[e].nonzero()[:, 0]))
            if len(episode_done_ids[e]) == 0 or episode_done_ids[e][-1] != self.num_steps - 1:
                episode_done_ids[e].append(self.num_steps - 1)
        max_seq_length = 1
        for key, value in samples.items():
            sequences = []
            for e in range(self.num_envs):
                start_index = 0
                for done_index in episode_done_ids[e]:
                    episode = value[e, start_index:done_index + 1]
                    start_index = done_index + 1
                    if self.recurrent_seq_len > 0 and not episode.shape[0] < self.recurrent_seq_len:
                        for start in range(0, len(episode), self.recurrent_seq_len):
                            end = start + self.recurrent_seq_len
                            sequences.append(episode[start:end])
                        max_seq_length = self.recurrent_seq_len
                    else:
                        sequences.append(episode)
                        max_seq_length = len(episode) if len(episode) > max_seq_length else max_seq_length
            for i, seq in enumerate(sequences):
                sequences[i] = self.pad_sequence(seq, max_seq_length)

            samples[key] = torch.stack(sequences, dim=0)
            if key == "a_hxs" or key == "a_cxs" or key == "c_hxs" or key == "c_cxs":
                samples[key] = samples[key][:, 0]

        self.true_sequence_length = max_seq_length
        self.samples_flat = {}
        for key, value in samples.items():
            if not key == "a_hxs" and not key == "a_cxs" and not key == "c_hxs" and not key == "c_cxs":
                value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            self.samples_flat[key] = value

    def get(self, batch_size=None):
        num_sequences = len(self.samples_flat["values"]) // self.true_sequence_length
        n_mini_batches = num_sequences // batch_size
        num_seq_per_batch = [batch_size] * n_mini_batches
        remainder = num_sequences % n_mini_batches
        for i in range(remainder):
            num_seq_per_batch[i] += 1
        indices = torch.arange(0, num_sequences * self.true_sequence_length).reshape(num_sequences, self.true_sequence_length)
        sequence_indices = torch.randperm(num_sequences)

        start = 0
        for n_seq in num_seq_per_batch:
            end = start + n_seq
            mini_batch_ids = indices[sequence_indices[start:end]].reshape(-1)
            mini_batch = {}
            for key, value in self.samples_flat.items():
                if key != "a_hxs" and key != "a_cxs" and key != "c_hxs" and key != "c_cxs":
                    mini_batch[key] = value[mini_batch_ids].to(DEVICE)
                else:
                    mini_batch[key] = value[sequence_indices[start:end]].to(DEVICE).transpose(0, 1)
            start = end
            yield mini_batch


    def reset(self):
        super().reset()
        self.actor_hxs = torch.zeros_like(self.actor_hxs).to(DEVICE)
        self.actor_cxs = torch.zeros_like(self.actor_cxs).to(DEVICE)
        self.critic_hxs = torch.zeros_like(self.critic_hxs).to(DEVICE)
        self.critic_cxs = torch.zeros_like(self.critic_cxs).to(DEVICE)

