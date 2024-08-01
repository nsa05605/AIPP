#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:00:38 2020

@author: yongyongwei
"""

import torch.nn.functional as F
import torch.nn as nn
import torch

class VDNNet(nn.Module):
    def __init__(self):
        super(VDNNet,self).__init__()
        
    def forward(self, q_values):
        return torch.sum(q_values, dim = 2, keepdim = True)
    
    
class QMixNet(nn.Module):
    def __init__(self, input_shape, args):
        super(QMixNet, self).__init__()
        self.args = args
        self.input_shape = input_shape
        
        self.hyper_encoding = nn.GRUCell(input_shape, args.hyper_hidden_dim)
        
        self.hyper_w1 = nn.Linear(args.hyper_hidden_dim, args.n_robots * args.qmix_hidden_dim)
        self.hyper_w2 = nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim * 1)
        
        self.hyper_b1 = nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(args.qmix_hidden_dim,1)
                                     )
        
    def init_hidden(self):
        return self.hyper_w1.weight.new(1, self.args.hyper_hidden_dim).zero_()
        
    def forward(self, q_value, state, hidden_state):
        """Only for one transition step:
            q_value: episode_num *  n_robots
            state: episode_num * input_shape
            hidden: episode_num * hyper_hidden_dim
        """
        h_en = self.hyper_encoding(state,hidden_state) #episode_num * hyper_hidden_dim
        
        w1 = torch.abs(self.hyper_w1(h_en)) #episode_num * (n_agents * qmix_hidden_dim)
        b1 = self.hyper_b1(h_en) #episode_num * qmix_hidden_dim
        
        w1 = w1.reshape(-1, self.args.n_robots, self.args.qmix_hidden_dim) #episode_num, n_agents, qmix_hidden
        b1 = b1.reshape(-1, 1, self.args.qmix_hidden_dim) #episode_num, 1, qmix_hidden_dim
        
        q_value = q_value.reshape(-1,1,self.args.n_robots) #episode_num, 1, n_agents
        mix_hidden = F.elu(torch.bmm(q_value, w1) + b1) #episode_num, 1, qmix_hidden_dim
        
        w2 = torch.abs(self.hyper_w2(h_en)) #episode_num, qmix_hidden_dim
        b2 = self.hyper_b2(h_en) #episode_num, 1
        
        w2 = w2.reshape(-1, self.args.qmix_hidden_dim,1) #episode_num, qmix_hidden_dim,1
        b2 = b2.reshape(-1, 1, 1) #episode_num, 1, 1
        
        q_total = torch.bmm(mix_hidden, w2) + b2 #episode_num,1,1
        q_total = q_total.squeeze(-1) #episode_num,1
        
        return q_total, h_en
        
        
        
        
        
        
        