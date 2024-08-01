#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 10:17:37 2020

@author: yongyongwei
"""


import torch
import torch.nn as nn 

class CANNet(nn.Module):
    def __init__(self, args):
        super(CANNet, self).__init__()
        self.args = args
        
        ae_input = self.args.rnn_hidden_dim + self.args.n_actions
        self.hidden_action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                                    nn.ReLU(),
                                                    nn.Linear(ae_input,ae_input))
        
        self.pred_r = nn.Sequential(nn.Linear(ae_input, self.args.can_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.can_hidden_dim, self.args.can_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.can_hidden_dim,1))
        
    def forward(self, hidden_states, actions):
        #episode_num, max_episode_len, n_agents, _ =  hidden_evals.shape
        episode_num, max_episode_len, n_agents, _ = actions.shape
        hidden_actions = torch.cat([hidden_states, actions], dim=-1)
        hidden_actions = hidden_actions.reshape(-1, self.args.rnn_hidden_dim + self.args.n_actions)
        hidden_actions_encoding = self.hidden_action_encoding(hidden_actions)
        hidden_actions_encoding = hidden_actions_encoding.reshape(episode_num * max_episode_len * n_agents, -1)

        pred_r = self.pred_r(hidden_actions_encoding)
        
        return pred_r


"""
#imp1
import torch.nn.functional as F
class CANNet(nn.Module):
    def __init__(self, args):
        super(CANNet, self).__init__()
        self.args = args
        
        ae_input = self.args.rnn_hidden_dim + self.args.n_actions
        self.hidden_action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                                    nn.ReLU(),
                                                    nn.Linear(ae_input,ae_input))
        
        self.pred_r = nn.Sequential(nn.Linear(ae_input, self.args.can_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.can_hidden_dim, self.args.can_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.can_hidden_dim,args.n_agents))
        
    def forward(self, hidden_states, actions):
        episode_num, max_episode_len, n_agents, _ = actions.shape
        hidden_actions = torch.cat([hidden_states, actions], dim=-1)
        hidden_actions = hidden_actions.reshape(-1, self.args.rnn_hidden_dim + self.args.n_actions)
        hidden_actions_encoding = self.hidden_action_encoding(hidden_actions)
        hidden_actions_encoding = hidden_actions_encoding.reshape(episode_num * max_episode_len * n_agents, -1)
        hidden_actions_encoding = hidden_actions_encoding.reshape(episode_num * max_episode_len, n_agents, -1)
        hidden_actions_encoding = hidden_actions_encoding.reshape(episode_num * max_episode_len, n_agents, -1)
        hidden_actions_encoding = F.relu(torch.sum(hidden_actions_encoding,dim=-2))
        pred_r = self.pred_r(hidden_actions_encoding)
        pred_r = pred_r.reshape(episode_num * max_episode_len, n_agents, -1)
        return pred_r      
"""