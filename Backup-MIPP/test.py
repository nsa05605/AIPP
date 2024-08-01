#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:34:01 2020
Test code

Re-load a module/function in python > 3.4

(1) import importlib
(2) importlib.reload(module)

#*******Note the module needs to be already imported before can be reload*****
E.g. to reload the load_config function, suppose previously called:
     from common.utils import load_config 
     (1) import common.utils #load module
     (2) importlib.reload(common.utils) #reload module
     (3) from common.utils import load_config #reload function


@author: yongyongwei
"""

from common.utils import load_config,RecursiveNamespace as RN
from ipp_envs import EnvMARL,EnvSeqRL
import importlib

from agent import Agents
from rollout import RolloutWorker_SEQ, RolloutWorker_MA
from common.replay_buffer import padding_episodes, ReplayBuffer

if __name__ == '__main__':
    
    
    config = load_config(config_path='config.yaml')
    
    #**************Rollout Test For Sequential case*********************
    args = RN(**config)
    #Needs first to create env and specify the number of agents
    env_seq = EnvSeqRL('area_one',config)
    env_seq.get_dim_info(3,args)
    
    agents = Agents(args,"seq+reinforce")
    rolloutWorker = RolloutWorker_SEQ(env_seq, agents, args)
    exp_buffer = ReplayBuffer(args,100)

    paths_specific = {'vses':[0,13,26], 'Bs':[30,30,30], 'chargingnodes': [0,26,13]}
    
    episodes = []
    for i in range(8):
        episode, episode_reward, episode_paths = rolloutWorker.generate_episode(paths_specific,plot=False)
        episodes.append(episode)
        
        for k,v in episode.items():
            print(k,v.shape)
        eval_reward = env_seq.paths_evaluate(episode_paths)
        print(episode_reward - eval_reward)
        
    padded_batch = padding_episodes(episodes)
    exp_buffer.store_episodes(episodes)
    
    for k in exp_buffer.buffer.keys():
        print(k,exp_buffer.buffer[k].shape)
        
    #sanity check
    for i in range(len(episodes)):
        for k in episodes[i].keys():
            source_item = episodes[i][k]
            dest_item = exp_buffer.buffer[k][i][:len(source_item)]
            comp = source_item == dest_item
            print(comp.sum() == comp.size)
            #Check the padded section
            remaining_entries = len(exp_buffer.buffer[k][i]) - len(source_item)
            if k in ['padded','terminated']:
                print('padded check:',exp_buffer.buffer['padded'][i][len(source_item):].sum() == remaining_entries)
                print('terminated check:',exp_buffer.buffer['terminated'][i][len(source_item):].sum() == remaining_entries)
            else:
                print('other fields check:',exp_buffer.buffer[k][i][len(source_item):].sum() == 0)
    #**************Rollout Test For Cooperative case*********************
    args = RN(**config)
    env_ma = EnvMARL('area_one', config)
    env_ma.get_dim_info(3,args)
    
    agents = Agents(args, "seq+reinforce")
    rolloutWorker = RolloutWorker_MA(env_ma, agents, args)
    exp_buffer = ReplayBuffer(args, 50)
    
    #paths_specific = {'vses':[0,26], 'Bs':[30,30], 'chargingnodes': [0,26,13]}
    paths_specific = {'vses':[0,13,26], 'Bs':[30,30,30], 'chargingnodes': [0,26,13]}
    
    
    episodes = []
    for i in range(8):
        episode, episode_reward, episode_paths = rolloutWorker.generate_episode(paths_specific,plot=False)
        episodes.append(episode)
        
        for k,v in episode.items():
            print(k,v.shape)    
        eval_reward = env_ma.paths_evaluate(episode_paths)
        
        print(episode_reward - eval_reward)
    padded_batch = padding_episodes(episodes)
    exp_buffer.store_episodes(episodes)
    
    for k in exp_buffer.buffer.keys():
        print(k,exp_buffer.buffer[k].shape)
    
    #sanity check
    for i in range(len(episodes)):
        for k in episodes[i].keys():
            source_item = episodes[i][k]
            dest_item = exp_buffer.buffer[k][i][:len(source_item)]
            comp = source_item == dest_item
            print(comp.sum() == comp.size)
            #Check the padded section
            remaining_entries = len(exp_buffer.buffer[k][i]) - len(source_item)
            if k in ['padded','terminated']:
                print('padded check:',exp_buffer.buffer['padded'][i][len(source_item):].sum() == remaining_entries)
                print('terminated check:',exp_buffer.buffer['terminated'][i][len(source_item):].sum() == remaining_entries)
            else:
                print('other fields check:',exp_buffer.buffer[k][i][len(source_item):].sum() == 0)
   
    