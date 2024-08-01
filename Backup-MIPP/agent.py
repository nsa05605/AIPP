#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:36:55 2020

@author: yongyongwei
"""

import numpy as np
import torch
from torch.distributions import Categorical
from policy.reinforce import Reinforce
from policy.dqn import DQN
from policy.a2c import A2C,ABR_A2C
from policy.mix import MIX,ABR_MIX
from policy.coma import COMA
from policy.qtran_base import QtranBase
from policy.can import CAN
from policy.acca import ACCA
from policy.jal import JAL
from policy.bdq import ABR_BDQ

from policy.ligm import LIGM

import itertools

from common.utils import is_onpolicy,is_coopmarl


class Agents:
    def __init__(self,args):
        if args.alg.find('reinforce') > -1:
            self.policy = Reinforce(args)
        elif args.alg.find('dqn') > -1:
            self.policy = DQN(args)
        elif args.alg.find('a2c') > -1:
            self.policy = A2C(args)
        elif args.alg.find('mix') > -1:
            self.policy = MIX(args)
        elif args.alg.find('coma') >-1:
            self.policy = COMA(args)
        elif args.alg.find('qtran_base') > -1:
            self.policy = QtranBase(args)
        elif args.alg.find('can') >-1:
            self.policy = CAN(args)
        elif args.alg.find('acca') >-1:
            self.policy = ACCA(args)
        elif args.alg.find('ligm') >-1:
            self.policy = LIGM(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        
        print('Init Agents with policy {}'.format(args.alg))
        
    def choose_action(self, nn_input, agent_num, valid_action, epsilon, evaluate = False):
        #RNN case
        if self.args.state_type == 0:
            #note here valid action is a list of action number, not one hot
            if is_coopmarl(self.args.alg) and self.args.reuse_network == False:
                hidden_state = self.policy.eval_hidden[agent_num]
            else:
                hidden_state = self.policy.eval_hidden[:, agent_num,:]
        else:
            hidden_state = None
                
        nn_input = torch.tensor(nn_input, dtype = torch.float32).unsqueeze(0)
        
        
        if self.args.cuda: 
            nn_input = nn_input.cuda()
            if self.args.state_type ==0:
                hidden_state = hidden_state.cuda()
            
        #RNN case
        if self.args.state_type==0:
            if is_coopmarl(self.args.alg) and self.args.reuse_network == False: 
                q_value, self.policy.eval_hidden[agent_num] = self.policy.eval_net[agent_num].forward(nn_input, hidden_state)
            else:
                q_value, self.policy.eval_hidden[:, agent_num,:] = self.policy.eval_net.forward(nn_input, hidden_state)
        #MLP or CNN case
        else:
            if is_coopmarl(self.args.alg) and self.args.reuse_network == False: 
                q_value = self.policy.eval_net[agent_num].forward(nn_input)
            else:
                q_value = self.policy.eval_net.forward(nn_input)            
            
        if is_onpolicy(self.args.alg):
            action = self._choose_action_from_softmax(q_value.cpu(), valid_action,epsilon, evaluate)
        else:
            if np.random.uniform() < epsilon:
                action = np.random.choice(valid_action)
            else:
                #q_value = q_value.cpu()
                action = valid_action[torch.argmax(q_value[0,valid_action])]
                
        return action
            
    
    def _choose_action_from_softmax(self, logits, valid_action, epsilon, evaluate = False):
        m2 = torch.full(logits.shape, -float("inf"))
        m2[:, valid_action] = 0
        prob = torch.nn.functional.softmax(logits + m2, dim = -1)
        
        if self.args.softmax_noise:
            prob = (1-epsilon) * prob + torch.ones_like(prob) * epsilon / len(valid_action)
            m3 = torch.zeros_like(prob)
            m3[:, valid_action] = 1
            prob[m3 == 0] = 0.0
        
        if epsilon == 0 and evaluate:
            action = torch.argmax(prob).item()
        else:
            action = Categorical(prob).sample().item()
        
        return action
    
    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        
        episode_num = terminated.shape[0]
        batch_padded_len = terminated.shape[1]
        
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(batch_padded_len):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
                    
        return max_episode_len
    
    def train(self,batch, train_step, epsilon=None, agent_id = None):
        if self.args.state_type == 0:
            if is_onpolicy(self.args.alg) == True:
                max_episode_len = batch['terminated'].shape[1]
            else:
                max_episode_len = self._get_max_episode_len(batch)
                for key in batch.keys():
                    batch[key] = batch[key][:,:max_episode_len]
            self.policy.learn(batch, max_episode_len, train_step, epsilon, agent_id)
        else:
            self.policy.learn(batch, None, train_step, epsilon, agent_id)
        
        
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step,agent_id)
            
        
class ABR_Agents:
    def __init__(self,args):
        if args.alg.find('bdq') > -1:
            self.policy = ABR_BDQ(args)
        elif args.alg.find('a2c') >-1:
            self.policy = ABR_A2C(args)
        elif args.alg.find('mix') > -1:
            self.policy = ABR_MIX(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        
        print('Init Agents with policy {}'.format(args.alg))
        
    def choose_actions(self, nn_input, agent_num, valid_actions, epsilon, evaluate = False):
        if self.args.state_type == 0:
            #note here valid actions is a 2D list of action number, not one hot
            hidden_state = self.policy.eval_hidden[:, agent_num,:]
        else:
            hidden_state = None
            
        nn_input = torch.tensor(nn_input, dtype = torch.float32).unsqueeze(0)
        
        if self.args.cuda: 
            nn_input = nn_input.cuda()
            if self.args.state_type ==0:
                hidden_state = hidden_state.cuda()
            
        if self.args.state_type==0:
            q_values, self.policy.eval_hidden[:, agent_num,:] = self.policy.eval_net.forward(nn_input, hidden_state)
        else:
            q_values = self.policy.eval_net.forward(nn_input)

        if is_onpolicy(self.args.alg):
            actions = self._choose_actions_from_softmax(q_values, valid_actions,epsilon, evaluate)
        else:
            actions = []
            for q_value, valid_action in zip(q_values,valid_actions):
                if np.random.uniform() < epsilon:
                    action = np.random.choice(valid_action)
                else:
                    #q_value = q_value.cpu()
                    action = valid_action[torch.argmax(q_value[0,valid_action])]
                actions.append(action)
                
        return actions
            
    
    def _choose_actions_from_softmax(self, logits, valid_actions, epsilon, evaluate = False):
        actions = []
        for logit,valid_action in zip(logits, valid_actions):
            logit = logit.cpu()
            m2 = torch.full(logit.shape, -float("inf"))
            m2[:, valid_action] = 0
            prob = torch.nn.functional.softmax(logit + m2, dim = -1)
            
            if self.args.softmax_noise:
                prob = (1-epsilon) * prob + torch.ones_like(prob) * epsilon / len(valid_action)
                m3 = torch.zeros_like(prob)
                m3[:, valid_action] = 1
                prob[m3 == 0] = 0.0
            if epsilon == 0 and evaluate:
                action = torch.argmax(prob).item()
            else:
                action = Categorical(prob).sample().item()
            
            actions.append(action)
            
        return actions
    
    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        
        episode_num = terminated.shape[0]
        batch_padded_len = terminated.shape[1]
        
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(batch_padded_len):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
                    
        return max_episode_len
    
    def train(self,batch, train_step, epsilon=None, agent_id = None):
        if self.args.state_type ==0:
            if is_onpolicy(self.args.alg) == True:
                max_episode_len = batch['terminated'].shape[1]
            else:
                max_episode_len = self._get_max_episode_len(batch)
                for key in batch.keys():
                    batch[key] = batch[key][:,:max_episode_len]
            self.policy.learn(batch, max_episode_len, train_step, epsilon)
        else:
            self.policy.learn(batch, None, train_step, epsilon)
            
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
            
        
class JAL_Agents:
    def __init__(self,args):
        self.policy = JAL(args)
        self.args = args
        print('Init Agents with policy {}'.format(args.alg))
        
    def choose_actions(self, nn_input, agent_num, valid_actions, epsilon, evaluate = False):
        if self.args.state_type == 0:
            #note here valid actions is a 2D list of action number, not one hot
            hidden_state = self.policy.eval_hidden[:, agent_num,:]
        else:
            hidden_state = None
            
        nn_input = torch.tensor(nn_input, dtype = torch.float32).unsqueeze(0)
        
        if self.args.cuda: 
            nn_input = nn_input.cuda()
            if self.args.state_type ==0:
                hidden_state = hidden_state.cuda()
            
        if self.args.state_type==0:
            q_values, self.policy.eval_hidden[:, agent_num,:] = self.policy.eval_net.forward(nn_input, hidden_state)
        else:
            q_values = self.policy.eval_net.forward(nn_input)


        actions = []
        if np.random.uniform() < epsilon:
            for valid_action in valid_actions:
                actions.append(np.random.choice(valid_action))
        else:
            valid_indices =  self._get_joint_action_indices(valid_actions)
            action_index = valid_indices[torch.argmax(q_values[0,valid_indices])]
            actions = self._get_joint_action_from_index(action_index)
            #Note the order of the joint action
            for action,valid_action in zip(actions,valid_actions):
                assert action in valid_action,"the mapping is problematic!"

        return actions
            

    def _get_joint_action_indices(self,valid_actions):
        action_size = self.args.single_robot_action_size
        indices=[]
        for joint_action in itertools.product(*valid_actions):
            index=0
            for i,action_num in enumerate(joint_action):
                index += (action_size ** i) * action_num
            indices.append(index)
        return indices

    def _get_joint_action_from_index(self,action_index):
        actions=[]
        action_size = self.args.single_robot_action_size
        if action_index<action_size:
            actions.append(action_index)
        else:
            while(action_index//action_size >=1):
                actions.append(action_index % action_size)
                action_index = action_index //action_size
            if (action_index % action_size !=0):
                actions.append(action_index % action_size)
        if len(actions) < self.args.n_robots:
            actions =  actions + [0] * (self.args.n_robots - len(actions))
        return actions
            
        
    
    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        
        episode_num = terminated.shape[0]
        batch_padded_len = terminated.shape[1]
        
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(batch_padded_len):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
                    
        return max_episode_len
    
    def train(self,batch, train_step, epsilon=None, agent_id = None):
        if self.args.state_type ==0:
            if is_onpolicy(self.args.alg) == True:
                max_episode_len = batch['terminated'].shape[1]
            else:
                max_episode_len = self._get_max_episode_len(batch)
                for key in batch.keys():
                    batch[key] = batch[key][:,:max_episode_len]
            self.policy.learn(batch, max_episode_len, train_step, epsilon)
        else:
            self.policy.learn(batch, None, train_step, epsilon)
            
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)
            
            
