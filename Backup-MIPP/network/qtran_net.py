#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:17:04 2020

@author: yongyongwei
"""

import torch
import torch.nn as nn 

class Qtran_basenet(nn.Module):
    def __init__(self, args):
        super(Qtran_basenet, self).__init__()
        self.args = args
        
        ae_input = self.args.rnn_hidden_dim + self.args.n_actions
        self.hidden_action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                                    nn.ReLU(),
                                                    nn.Linear(ae_input,ae_input))
        
        self.q = nn.Sequential(nn.Linear(ae_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim,1))
        
    def forward(self, hidden_states, actions):
        episode_num, max_episode_len, n_agents, _ = actions.shape
        hidden_actions = torch.cat([hidden_states, actions], dim=-1)
        hidden_actions = hidden_actions.reshape(-1, self.args.rnn_hidden_dim + self.args.n_actions)
        hidden_actions_encoding = self.hidden_action_encoding(hidden_actions)
        hidden_actions_encoding = hidden_actions_encoding.reshape(episode_num * max_episode_len, n_agents, -1)
        hidden_actions_encoding = hidden_actions_encoding.sum(dim=-2)
        
        q = self.q(hidden_actions_encoding)
        
        return q
        