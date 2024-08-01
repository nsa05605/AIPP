#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:06:39 2020

@author: yongyongwei
"""

import numpy as np
import threading
from common.utils import is_coopmarl

def padding_episodes(episodes, max_episode_len = None):
    if max_episode_len == None:
        max_episode_len = np.max([e['s'].shape[0] for e in episodes])
    padded_episodes = []
    for episode in episodes:
        padded_episode={}
        e_step_len = episode['s'].shape[0]
        if e_step_len < max_episode_len:
            diff = max_episode_len - e_step_len
            for key in episode.keys():
                if key not in ['padded','terminated']:
                    padded_episode[key] = np.concatenate((episode[key],np.zeros([diff]+list(episode[key].shape[1:]))),axis=0)
                else:
                    padded_episode[key] = np.concatenate((episode[key],np.ones([diff]+list(episode[key].shape[1:]))),axis=0)
            padded_episodes.append(padded_episode)
        else:
            padded_episodes.append(episode)
            
    episode_batch = {}
    for key in padded_episodes[0].keys():
        episode_batch[key] = np.stack([episode[key] for episode in padded_episodes])
    return episode_batch
        
def merge_episodes(episodes):
    episode_batch={}
    for key in episodes[0].keys():
        episode_batch[key] = np.concatenate([episode[key] for episode in episodes])
    return episode_batch
        

class ReplayBuffer:
    def __init__(self,args, max_episode_len = 100, num_of_nodes = None ):
        self.args = args
        self.max_episode_len = max_episode_len
        self.num_of_nodes = num_of_nodes
        
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        if args.n_agents == 1:
            self.state_dim = 3 * self.args.n_robots 
        elif self.args.reuse_network == False and is_coopmarl(args.alg) == False:
            self.state_dim = 3 * self.args.n_robots 
        else:
            self.state_dim = 3 * self.args.n_robots + self.args.n_robots 
        
        self.size = self.args.buffer_size
        
        self.current_idx = 0
        self.current_size = 0

        self.reward_dim = args.n_robots

        self.lock = threading.Lock()
        
        #for the case with RNN
        if self.num_of_nodes == None:
            self.buffer = {
                's': np.empty([self.size, self.max_episode_len, self.n_agents, self.state_dim]),
                's_next': np.empty([self.size, self.max_episode_len, self.n_agents, self.state_dim]),
                'u': np.empty([self.size, self.max_episode_len, self.n_agents, 1]),
                'u_onehot': np.empty([self.size, self.max_episode_len, self.n_agents, self.n_actions]),
                'r': np.empty([self.size, self.max_episode_len, self.reward_dim]),
                'valid_u': np.empty([self.size, self.max_episode_len, self.n_agents,self.n_actions]),
                'valid_u_next': np.empty([self.size, self.max_episode_len, self.n_agents,self.n_actions]),
                'padded': np.empty([self.size, self.max_episode_len, 1]),
                'terminated': np.empty([self.size, self.max_episode_len, 1])
                }
        #for the case with MLP
        else:
            self.buffer = {
                's': np.empty([self.size, self.n_agents, self.state_dim]),
                's_next': np.empty([self.size, self.n_agents, self.state_dim]),
                'nodes_info':np.empty([self.size, self.n_agents,self.num_of_nodes]),
                'nodes_info_next': np.empty([self.size, self.n_agents,self.num_of_nodes]),
                'u': np.empty([self.size, self.n_agents, 1]),
                'u_onehot': np.empty([self.size, self.n_agents, self.n_actions]),
                'r': np.empty([self.size, self.reward_dim]),
                'valid_u': np.empty([self.size,self.n_agents,self.n_actions]),
                'valid_u_next': np.empty([self.size, self.n_agents,self.n_actions]),
                'terminated': np.empty([self.size, 1])
                }            
            
        
    def store_episodes(self,episodes):        
        if self.num_of_nodes is None:
            episode_len = np.max([e['s'].shape[0] for e in episodes])
            while episode_len > self.max_episode_len:
                self._expand_episode_len()
            batch_size = len(episodes)
        
            episode_batch = padding_episodes(episodes, max_episode_len = self.max_episode_len)
            with self.lock:
                idxs = self._get_storage_idx(inc = batch_size)
                self.buffer['s'][idxs] = episode_batch['s']
                self.buffer['s_next'][idxs] = episode_batch['s_next']
                self.buffer['u'][idxs] = episode_batch['u']
                self.buffer['u_onehot'][idxs] = episode_batch['u_onehot']
                self.buffer['r'][idxs] = episode_batch['r']
                self.buffer['valid_u'][idxs] = episode_batch['valid_u']
                self.buffer['valid_u_next'][idxs] = episode_batch['valid_u_next']
                self.buffer['padded'][idxs] = episode_batch['padded']
                self.buffer['terminated'][idxs] = episode_batch['terminated']
        else:
            with self.lock:
                for episode in episodes:
                    episode_steps = episode['s'].shape[0]
                    idxs = self._get_storage_idx(inc = episode_steps)
                    self.buffer['s'][idxs] = episode['s']
                    self.buffer['s_next'][idxs] = episode['s_next']
                    self.buffer['nodes_info'][idxs] = episode['nodes_info']
                    self.buffer['nodes_info_next'][idxs] = episode['nodes_info_next']
                    self.buffer['u'][idxs] = episode['u']
                    self.buffer['u_onehot'][idxs] = episode['u_onehot']
                    self.buffer['r'][idxs] = episode['r']
                    self.buffer['valid_u'][idxs] = episode['valid_u']
                    self.buffer['valid_u_next'][idxs] = episode['valid_u_next']
                    self.buffer['terminated'][idxs] = episode['terminated']
                
            
        
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0,self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer
    
    def _expand_episode_len(self): 
        diff = self.max_episode_len 
        self.max_episode_len = self.max_episode_len * 2
        print('expanding max episode length to {}'.format(self.max_episode_len))
        buffer_new = {
            's': np.empty([self.size, self.max_episode_len, self.n_agents, self.state_dim]),
            's_next': np.empty([self.size, self.max_episode_len, self.n_agents, self.state_dim]),
            'u': np.empty([self.size, self.max_episode_len, self.n_agents, 1]),
            'u_onehot': np.empty([self.size, self.max_episode_len, self.n_agents, self.n_actions]),
            'r': np.empty([self.size, self.max_episode_len, self.reward_dim]),
            'valid_u': np.empty([self.size, self.max_episode_len, self.n_agents,self.n_actions]),
            'valid_u_next': np.empty([self.size, self.max_episode_len, self.n_agents,self.n_actions]),
            'padded': np.empty([self.size, self.max_episode_len, 1]),
            'terminated': np.empty([self.size, self.max_episode_len, 1])
            }
        for i in range(self.current_size):
            buffer_new['s'][i] = np.concatenate((self.buffer['s'][i],np.zeros([diff]+list(self.buffer['s'][i].shape[1:]))),axis=0)
            buffer_new['s_next'][i] = np.concatenate((self.buffer['s_next'][i],np.zeros([diff]+list(self.buffer['s_next'][i].shape[1:]))),axis=0)
            buffer_new['u'][i] = np.concatenate((self.buffer['u'][i],np.zeros([diff]+list(self.buffer['u'][i].shape[1:]))),axis=0)
            buffer_new['u_onehot'][i] = np.concatenate((self.buffer['u_onehot'][i],np.zeros([diff]+list(self.buffer['u_onehot'][i].shape[1:]))),axis=0)
            buffer_new['r'][i] = np.concatenate((self.buffer['r'][i],np.zeros([diff]+list(self.buffer['r'][i].shape[1:]))),axis=0)
            buffer_new['valid_u'][i] = np.concatenate((self.buffer['valid_u'][i],np.zeros([diff]+list(self.buffer['valid_u'][i].shape[1:]))),axis=0)
            buffer_new['valid_u_next'][i] = np.concatenate((self.buffer['valid_u_next'][i],np.zeros([diff]+list(self.buffer['valid_u_next'][i].shape[1:]))),axis=0)
            buffer_new['padded'][i] = np.concatenate((self.buffer['padded'][i],np.ones([diff]+list(self.buffer['padded'][i].shape[1:]))),axis=0)
            buffer_new['terminated'][i] = np.concatenate((self.buffer['terminated'][i],np.ones([diff]+list(self.buffer['terminated'][i].shape[1:]))),axis=0)
                         
        self.buffer = buffer_new                       
        
    def _get_storage_idx(self, inc  = None):
        inc = inc or 1 
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
            
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
            
        self.current_size = min(self.size, self.current_size + inc)
        
        if inc == 1:
            idx = idx[0]
            
        return idx
            
class ABR_ReplayBuffer:
    def __init__(self,args, max_episode_len = 100, num_of_nodes = None ):
        self.args = args
        self.max_episode_len = max_episode_len
        
        self.num_of_nodes = num_of_nodes
                
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.n_robots = self.args.n_robots
        
        assert self.n_agents == 1
        self.state_dim = 3 * self.args.n_robots
        self.size = self.args.buffer_size
        
        self.current_idx = 0
        self.current_size = 0
        
        
        self.reward_dim = args.n_robots

               
        self.lock = threading.Lock()
        
        if self.num_of_nodes is None:
            self.buffer = {
                's': np.empty([self.size, self.max_episode_len, self.n_agents, self.state_dim]),
                's_next': np.empty([self.size, self.max_episode_len, self.n_agents, self.state_dim]),
                'u': np.empty([self.size, self.max_episode_len, self.n_robots, 1]),
                'u_onehot': np.empty([self.size, self.max_episode_len, self.n_robots, self.n_actions]),
                'r': np.empty([self.size, self.max_episode_len, self.reward_dim]),
                'valid_u': np.empty([self.size, self.max_episode_len, self.n_robots,self.n_actions]),
                'valid_u_next': np.empty([self.size, self.max_episode_len, self.n_robots,self.n_actions]),
                'padded': np.empty([self.size, self.max_episode_len, 1]),
                'terminated': np.empty([self.size, self.max_episode_len, 1])
                }
        #for the case with MLP
        else:
            self.buffer = {
                's': np.empty([self.size, self.n_agents, self.state_dim]),
                's_next': np.empty([self.size, self.n_agents, self.state_dim]),
                'nodes_info':np.empty([self.size, self.n_agents,self.num_of_nodes]),
                'nodes_info_next': np.empty([self.size, self.n_agents,self.num_of_nodes]),
                'u': np.empty([self.size, self.n_agents, 1]),
                'u_onehot': np.empty([self.size, self.n_agents, self.n_actions]),
                'r': np.empty([self.size, self.reward_dim]),
                'valid_u': np.empty([self.size,self.n_agents,self.n_actions]),
                'valid_u_next': np.empty([self.size, self.n_agents,self.n_actions]),
                'terminated': np.empty([self.size, 1])
                }            
            
            
    def store_episodes(self,episodes):
        if self.num_of_nodes is None:
            episode_len = np.max([e['s'].shape[0] for e in episodes])
            while episode_len > self.max_episode_len:
                self._expand_episode_len()
            batch_size = len(episodes)
            episode_batch = padding_episodes(episodes, max_episode_len = self.max_episode_len)
            with self.lock:
                idxs = self._get_storage_idx(inc = batch_size)
                self.buffer['s'][idxs] = episode_batch['s']
                self.buffer['s_next'][idxs] = episode_batch['s_next']
                self.buffer['u'][idxs] = episode_batch['u']
                self.buffer['u_onehot'][idxs] = episode_batch['u_onehot']
                self.buffer['r'][idxs] = episode_batch['r']
                self.buffer['valid_u'][idxs] = episode_batch['valid_u']
                self.buffer['valid_u_next'][idxs] = episode_batch['valid_u_next']
                self.buffer['padded'][idxs] = episode_batch['padded']
                self.buffer['terminated'][idxs] = episode_batch['terminated']
                
        else:
            with self.lock:
                for episode in episodes:
                    episode_steps = episode['s'].shape[0]
                    idxs = self._get_storage_idx(inc = episode_steps)
                    self.buffer['s'][idxs] = episode['s']
                    self.buffer['s_next'][idxs] = episode['s_next']
                    self.buffer['nodes_info'][idxs] = episode['nodes_info']
                    self.buffer['nodes_info_next'][idxs] = episode['nodes_info_next']
                    self.buffer['u'][idxs] = episode['u']
                    self.buffer['u_onehot'][idxs] = episode['u_onehot']
                    self.buffer['r'][idxs] = episode['r']
                    self.buffer['valid_u'][idxs] = episode['valid_u']
                    self.buffer['valid_u_next'][idxs] = episode['valid_u_next']
                    self.buffer['terminated'][idxs] = episode['terminated']

        
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0,self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer
    
    def _expand_episode_len(self): 
        diff = self.max_episode_len 
        self.max_episode_len = self.max_episode_len * 2
        print('expanding max episode length to {}'.format(self.max_episode_len))
        buffer_new = {
            's': np.empty([self.size, self.max_episode_len, self.n_agents, self.state_dim]),
            's_next': np.empty([self.size, self.max_episode_len, self.n_agents, self.state_dim]),
            'u': np.empty([self.size, self.max_episode_len, self.n_robots, 1]),
            'u_onehot': np.empty([self.size, self.max_episode_len, self.n_robots, self.n_actions]),
            'r': np.empty([self.size, self.max_episode_len, self.reward_dim]),
            'valid_u': np.empty([self.size, self.max_episode_len, self.n_robots,self.n_actions]),
            'valid_u_next': np.empty([self.size, self.max_episode_len, self.n_robots,self.n_actions]),
            'padded': np.empty([self.size, self.max_episode_len, 1]),
            'terminated': np.empty([self.size, self.max_episode_len, 1])
            }
        for i in range(self.current_size):
            buffer_new['s'][i] = np.concatenate((self.buffer['s'][i],np.zeros([diff]+list(self.buffer['s'][i].shape[1:]))),axis=0)
            buffer_new['s_next'][i] = np.concatenate((self.buffer['s_next'][i],np.zeros([diff]+list(self.buffer['s_next'][i].shape[1:]))),axis=0)
            buffer_new['u'][i] = np.concatenate((self.buffer['u'][i],np.zeros([diff]+list(self.buffer['u'][i].shape[1:]))),axis=0)
            buffer_new['u_onehot'][i] = np.concatenate((self.buffer['u_onehot'][i],np.zeros([diff]+list(self.buffer['u_onehot'][i].shape[1:]))),axis=0)
            buffer_new['r'][i] = np.concatenate((self.buffer['r'][i],np.zeros([diff]+list(self.buffer['r'][i].shape[1:]))),axis=0)
            buffer_new['valid_u'][i] = np.concatenate((self.buffer['valid_u'][i],np.zeros([diff]+list(self.buffer['valid_u'][i].shape[1:]))),axis=0)
            buffer_new['valid_u_next'][i] = np.concatenate((self.buffer['valid_u_next'][i],np.zeros([diff]+list(self.buffer['valid_u_next'][i].shape[1:]))),axis=0)
            buffer_new['padded'][i] = np.concatenate((self.buffer['padded'][i],np.ones([diff]+list(self.buffer['padded'][i].shape[1:]))),axis=0)
            buffer_new['terminated'][i] = np.concatenate((self.buffer['terminated'][i],np.ones([diff]+list(self.buffer['terminated'][i].shape[1:]))),axis=0)
                         
        self.buffer = buffer_new                       
        
    def _get_storage_idx(self, inc  = None):
        inc = inc or 1 
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
            
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
            
        self.current_size = min(self.size, self.current_size + inc)
        
        if inc == 1:
            idx = idx[0]
            
        return idx
    

class JAL_ReplayBuffer:
    def __init__(self,args, max_episode_len = 100, num_of_nodes = None ):
        self.args = args
        self.max_episode_len = max_episode_len
        
        self.num_of_nodes = num_of_nodes
                
        self.n_actions = self.args.n_actions
        self.single_robot_action_size = self.args.single_robot_action_size
        self.n_agents = self.args.n_agents
        self.n_robots = self.args.n_robots
        
        assert self.n_agents == 1
        self.state_dim = 3 * self.args.n_robots
        self.size = self.args.buffer_size
        
        self.current_idx = 0
        self.current_size = 0
        
        
        self.reward_dim = 1

               
        self.lock = threading.Lock()
        
        #for the case with RNN
        if self.num_of_nodes == None:
            self.buffer = {
                's': np.empty([self.size, self.max_episode_len, self.n_agents, self.state_dim]),
                's_next': np.empty([self.size, self.max_episode_len, self.n_agents, self.state_dim]),
                'u': np.empty([self.size, self.max_episode_len, self.n_robots, 1]),
                'u_index': np.empty([self.size, self.max_episode_len, self.n_agents]),
                'u_onehot': np.empty([self.size, self.max_episode_len, self.n_agents, self.n_actions]),
                'r': np.empty([self.size, self.max_episode_len, self.reward_dim]),
                'valid_u': np.empty([self.size, self.max_episode_len, 1,self.n_actions]),
                'valid_u_next': np.empty([self.size, self.max_episode_len, 1,self.n_actions]),
                'padded': np.empty([self.size, self.max_episode_len, 1]),
                'terminated': np.empty([self.size, self.max_episode_len, 1])
                }
        #for the case with MLP
        else:
            self.buffer = {
                's': np.empty([self.size, self.n_agents, self.state_dim]),
                's_next': np.empty([self.size, self.n_agents, self.state_dim]),
                'nodes_info':np.empty([self.size, self.n_agents,self.num_of_nodes]),
                'nodes_info_next': np.empty([self.size, self.n_agents,self.num_of_nodes]),
                'u': np.empty([self.size, self.n_robots, 1]),
                'u_index':np.empty([self.size, self.n_agents]),
                'u_onehot': np.empty([self.size, self.n_agents, self.n_actions]),
                'r': np.empty([self.size, self.reward_dim]),
                'valid_u': np.empty([self.size,self.n_agents,self.n_actions]),
                'valid_u_next': np.empty([self.size, self.n_agents,self.n_actions]),
                'terminated': np.empty([self.size, 1])
                }            

            
    def store_episodes(self,episodes):
        if self.num_of_nodes is None:
            episode_len = np.max([e['s'].shape[0] for e in episodes])
            while episode_len > self.max_episode_len:
                self._expand_episode_len()
            batch_size = len(episodes)
            episode_batch = padding_episodes(episodes, max_episode_len = self.max_episode_len)
            with self.lock:
                idxs = self._get_storage_idx(inc = batch_size)
                self.buffer['s'][idxs] = episode_batch['s']
                self.buffer['s_next'][idxs] = episode_batch['s_next']
                self.buffer['u'][idxs] = episode_batch['u']
                self.buffer['u_index'][idxs] = episode_batch['u_index']
                self.buffer['u_onehot'][idxs] = episode_batch['u_onehot']
                self.buffer['r'][idxs] = episode_batch['r']
                self.buffer['valid_u'][idxs] = episode_batch['valid_u']
                self.buffer['valid_u_next'][idxs] = episode_batch['valid_u_next']
                self.buffer['padded'][idxs] = episode_batch['padded']
                self.buffer['terminated'][idxs] = episode_batch['terminated']

        else:
            with self.lock:
                for episode in episodes:
                    episode_steps = episode['s'].shape[0]
                    idxs = self._get_storage_idx(inc = episode_steps)
                    self.buffer['s'][idxs] = episode['s']
                    self.buffer['s_next'][idxs] = episode['s_next']
                    self.buffer['nodes_info'][idxs] = episode['nodes_info']
                    self.buffer['nodes_info_next'][idxs] = episode['nodes_info_next']
                    self.buffer['u'][idxs] = episode['u']
                    self.buffer['u_index'][idxs] = episode['u_index']
                    self.buffer['u_onehot'][idxs] = episode['u_onehot']
                    self.buffer['r'][idxs] = episode['r']
                    self.buffer['valid_u'][idxs] = episode['valid_u']
                    self.buffer['valid_u_next'][idxs] = episode['valid_u_next']
                    self.buffer['terminated'][idxs] = episode['terminated']
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0,self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer
    
    def _expand_episode_len(self): 
        diff = self.max_episode_len 
        self.max_episode_len = self.max_episode_len * 2
        print('expanding max episode length to {}'.format(self.max_episode_len))
        buffer_new = {
            's': np.empty([self.size, self.max_episode_len, self.n_agents, self.state_dim]),
            's_next': np.empty([self.size, self.max_episode_len, self.n_agents, self.state_dim]),
            'u': np.empty([self.size, self.max_episode_len, self.n_robots, 1]),
            'u_index': np.empty([self.size, self.max_episode_len, self.n_agents]),
            'u_onehot': np.empty([self.size, self.max_episode_len, self.n_agents, self.n_actions]),
            'r': np.empty([self.size, self.max_episode_len, self.reward_dim]),
            'valid_u': np.empty([self.size, self.max_episode_len,self.n_agents,self.n_actions]),
            'valid_u_next': np.empty([self.size, self.max_episode_len, self.n_agents,self.n_actions]),
            'padded': np.empty([self.size, self.max_episode_len, 1]),
            'terminated': np.empty([self.size, self.max_episode_len, 1])
            }
        for i in range(self.current_size):
            buffer_new['s'][i] = np.concatenate((self.buffer['s'][i],np.zeros([diff]+list(self.buffer['s'][i].shape[1:]))),axis=0)
            buffer_new['s_next'][i] = np.concatenate((self.buffer['s_next'][i],np.zeros([diff]+list(self.buffer['s_next'][i].shape[1:]))),axis=0)
            buffer_new['u'][i] = np.concatenate((self.buffer['u'][i],np.zeros([diff]+list(self.buffer['u'][i].shape[1:]))),axis=0)
            buffer_new['u_index'][i] = np.concatenate((self.buffer['u_index'][i],np.zeros([diff]+list(self.buffer['u_index'][i].shape[1:]))),axis=0)
            buffer_new['u_onehot'][i] = np.concatenate((self.buffer['u_onehot'][i],np.zeros([diff]+list(self.buffer['u_onehot'][i].shape[1:]))),axis=0)
            buffer_new['r'][i] = np.concatenate((self.buffer['r'][i],np.zeros([diff]+list(self.buffer['r'][i].shape[1:]))),axis=0)
            buffer_new['valid_u'][i] = np.concatenate((self.buffer['valid_u'][i],np.zeros([diff]+list(self.buffer['valid_u'][i].shape[1:]))),axis=0)
            buffer_new['valid_u_next'][i] = np.concatenate((self.buffer['valid_u_next'][i],np.zeros([diff]+list(self.buffer['valid_u_next'][i].shape[1:]))),axis=0)
            buffer_new['padded'][i] = np.concatenate((self.buffer['padded'][i],np.ones([diff]+list(self.buffer['padded'][i].shape[1:]))),axis=0)
            buffer_new['terminated'][i] = np.concatenate((self.buffer['terminated'][i],np.ones([diff]+list(self.buffer['terminated'][i].shape[1:]))),axis=0)
                         
        self.buffer = buffer_new                       
        
    def _get_storage_idx(self, inc  = None):
        inc = inc or 1 
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
            
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
            
        self.current_size = min(self.size, self.current_size + inc)
        
        if inc == 1:
            idx = idx[0]
            
        return idx