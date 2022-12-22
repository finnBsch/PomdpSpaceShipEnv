import argparse
from collections import deque
import librlsimpy
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


class ActorCritic1DConv(nn.Module):
    """
implements both actor and critic in one model
"""

    def __init__(self, state_dim=521, action_dim=4):
        super(ActorCritic1DConv, self).__init__()

        self.conv = nn.Conv1d(1, 1, 30, 5)
        self.conv_c = nn.Conv1d(1, 1, 30, 5)

        self.conv2 = nn.Conv1d(1, 1, 10, 2)
        self.conv2_c = nn.Conv1d(1, 1, 10, 2)

        self.affine1_a = nn.Linear(44 + 9, 64)
        self.affine2_a = nn.Linear(64, 64)
        # self.affine3_a = nn.Linear(64, 64)
        self.action_head = nn.Linear(64, action_dim)

        self.affine1_c = nn.Linear(44 + 9, 100)
        self.affine2_c = nn.Linear(100, 50)
        self.affine3_c = nn.Linear(50, 25)
        self.value_head = nn.Linear(25, 1)
        self.log_std = nn.Parameter(torch.FloatTensor([0.5, 0.5, 0.5, 0.5]).log(), requires_grad=True)

    def forward(self, x):
        """
        forward of both actor and critic
        """
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
            
        x_a = F.tanh(self.affine1_a(x_in_a))
        x_a = F.tanh(self.affine2_a(x_a))
        # x_a = F.tanh(self.affine3_a(x_a))
        x_c = F.tanh(self.affine1_c(x_in_c))
        x_c = F.tanh(self.affine2_c(x_c))
        x_c = F.tanh(self.affine3_c(x_c))

        action_prob = F.tanh(self.action_head(x_a))
        state_values = self.value_head(x_c)
        return action_prob, state_values


class LSTMActorCritic1DConv(nn.Module):
    """
implements both actor and critic in one model
"""

    def __init__(self, state_dim=521, hs_dim=50,  action_dim=4, recurrent_layers=1):
        super(LSTMActorCritic1DConv, self).__init__()
        self.recurrent_layers = recurrent_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor_lstm = nn.LSTM(44 + 9, hs_dim, num_layers=self.recurrent_layers, batch_first=True)
        self.critic_lstm = nn.LSTM(44 + 9, hs_dim, num_layers=self.recurrent_layers, batch_first=True)
        self.hidden_cell_a = None
        self.hidden_cell_c = None
        self.hs_dim = hs_dim
        self.cs_dim = hs_dim

        self.conv = nn.Conv1d(1, 1, 30, 5)
        nn.init.orthogonal_(self.conv.weight, np.sqrt(2))
        self.conv_c = nn.Conv1d(1, 1, 30, 5)
        nn.init.orthogonal_(self.conv_c.weight, np.sqrt(2))

        self.conv2 = nn.Conv1d(1, 1, 10, 2)
        nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
        self.conv2_c = nn.Conv1d(1, 1, 10, 2)
        nn.init.orthogonal_(self.conv2_c.weight, np.sqrt(2))

        self.affine1_a = nn.Linear(hs_dim, 100)
        nn.init.orthogonal_(self.affine1_a.weight, np.sqrt(2))
        self.affine2_a = nn.Linear(100, 50)
        nn.init.orthogonal_(self.affine2_a.weight, np.sqrt(2))
        self.affine3_a = nn.Linear(50, 25)
        nn.init.orthogonal_(self.affine3_a.weight, np.sqrt(2))
        self.action_head = nn.Linear(100, action_dim)
        nn.init.orthogonal_(self.action_head.weight, np.sqrt(2))


        self.affine1_c = nn.Linear(hs_dim, 100)
        nn.init.orthogonal_(self.affine1_c.weight, np.sqrt(2))
        self.affine2_c = nn.Linear(100, 64)
        nn.init.orthogonal_(self.affine2_c.weight, np.sqrt(2))
        self.affine3_c = nn.Linear(64, 64)
        nn.init.orthogonal_(self.affine3_c.weight, np.sqrt(2))
        self.value_head = nn.Linear(100, 1)
        nn.init.orthogonal_(self.value_head.weight, np.sqrt(2))

        self.log_std = nn.Parameter(torch.FloatTensor([0.5, 0.5, 0.5, 0.5]).log(), requires_grad=True)

    def reset_hidden(self):
        self.hidden_cell_a = None
        self.hidden_cell_c = None

    def get_init_state(self, batch_size):
        self.hidden_cell_a = (torch.zeros(self.recurrent_layers, batch_size, self.hs_dim).to(self.device),
                              torch.zeros(self.recurrent_layers, batch_size, self.hs_dim).to(self.device))
        self.hidden_cell_c = (torch.zeros(self.recurrent_layers, batch_size, self.hs_dim).to(self.device),
                              torch.zeros(self.recurrent_layers, batch_size, self.hs_dim).to(self.device))



    def warm_up_hidden_state(self, x, terminal):
        with torch.no_grad():
            length_data = x.shape[1]
            batch_size = x.shape[0]
            x_state, x_conv = torch.split(x, [9, 512], dim=x.dim()-1)
            x_conv = x_conv.reshape((-1, x_conv.shape[2]))  # reshape to (-1, feature dim)
            x_conv = x_conv.unsqueeze(1)

            x_conv_a = F.leaky_relu_(self.conv2(F.leaky_relu_(self.conv(x_conv*2 - 1))))
            x_conv_c = F.leaky_relu_(self.conv2_c(F.leaky_relu_(self.conv_c(x_conv*2 - 1))))

            x_conv_a = x_conv_a.reshape((batch_size, length_data, -1))
            x_conv_c = x_conv_c.reshape((batch_size, length_data, -1))

            x_in_a = torch.concat((x_state, x_conv_a), -1)
            x_in_c = torch.concat((x_state, x_conv_c), -1)
            x_in_a = x_in_a.reshape((batch_size, length_data, 53))
            x_in_c = x_in_c.reshape((batch_size, length_data, 53))
            terminal = torch.hstack([torch.zeros((batch_size, 1)).to(self.device), terminal])
            for i in range(length_data):
                _, self.hidden_cell_a = self.actor_lstm(x_in_a[:, i:i+1], ((1.0 - terminal[:, i]).view(1, -1, 1)*self.hidden_cell_a[0],
                                      (1.0 - terminal[:, i]).view(1, -1, 1)*self.hidden_cell_a[1]))
                _, self.hidden_cell_c = self.critic_lstm(x_in_c[:, i:i+1], ((1.0 - terminal[:, i]).view(1, -1, 1)*self.hidden_cell_c[0],
                                                                    (1.0 - terminal[:, i]).view(1, -1, 1)*self.hidden_cell_c[1]))

    def forward(self, x, terminal, act_hxs=None, act_cxs=None, crit_hxs=None, crit_cxs=None, sequence_length = 1):
        """
        forward of both actor and critic
        """
        if sequence_length > 1:
            # Reshape to (batch_size, length_, feature dim)
            x = x.reshape((x.shape[0]//sequence_length, sequence_length, x.shape[1]))
        if not x.dim() == 3:
            x = x.unsqueeze(1)
        length_data = x.shape[1]
        batch_size = x.shape[0]
        x_state, x_conv = torch.split(x, [9, 512], dim=x.dim()-1)
        x_conv = x_conv.reshape((-1, x_conv.shape[2]))  # reshape to (-1, feature dim)
        x_conv = x_conv.unsqueeze(1)

        x_conv_a = F.leaky_relu_(self.conv2(F.leaky_relu_(self.conv(x_conv*2 - 1))))
        x_conv_c = F.leaky_relu_(self.conv2_c(F.leaky_relu_(self.conv_c(x_conv*2 - 1))))

        x_conv_a = x_conv_a.reshape((batch_size, length_data, -1))
        x_conv_c = x_conv_c.reshape((batch_size, length_data, -1))

        x_in_a = torch.concat((x_state, x_conv_a), -1)
        x_in_c = torch.concat((x_state, x_conv_c), -1)
        if act_hxs is not None:
            self.hidden_cell_a = (act_hxs.contiguous(), act_cxs.contiguous())
            self.hidden_cell_c = (crit_hxs.contiguous(), crit_cxs.contiguous())
        if self.hidden_cell_a is None or batch_size != self.hidden_cell_a[0].shape[1]:
            self.get_init_state(batch_size)


        x_in_a = x_in_a.reshape((batch_size, length_data, 53))
        x_in_c = x_in_c.reshape((batch_size, length_data, 53))

        if terminal is not None and terminal.dim() == 1:
            self.hidden_cell_a = ((1.0 - terminal).view(1, -1, 1)*self.hidden_cell_a[0],
                                  (1.0 - terminal).view(1, -1, 1)*self.hidden_cell_a[1])
            self.hidden_cell_c = ((1.0 - terminal).view(1, -1, 1)*self.hidden_cell_c[0],
                                  (1.0 - terminal).view(1, -1, 1)*self.hidden_cell_c[1])
        x_in_a_, self.hidden_cell_a = self.actor_lstm(x_in_a, self.hidden_cell_a)
        x_in_c_, self.hidden_cell_c = self.critic_lstm(x_in_c, self.hidden_cell_c)

        x_in_a_ = torch.flatten(x_in_a_, 0, 1)
        x_in_c_ = torch.flatten(x_in_c_, 0, 1)
        x_a = F.tanh(self.affine1_a(x_in_a_))
        # x_a = F.tanh(self.affine2_a(x_a))
        # x_a = F.tanh(self.affine3_a(x_a))
        x_c = F.tanh(self.affine1_c(x_in_c_))
        # x_c = F.tanh(self.affine2_c(x_c))
        # x_c = F.tanh(self.affine3_c(x_c))

        action_prob = F.tanh(self.action_head(x_a))
        state_values = self.value_head(x_c)
        return action_prob, state_values
