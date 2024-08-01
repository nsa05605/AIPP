#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:39:45 2020

@author: yongyongwei
"""

from agent import Agents, ABR_Agents,JAL_Agents
from rollout import RolloutWorker_SEQ, RolloutWorker_MA,RolloutWorker_ABR,RolloutWorker_JAL
from common.replay_buffer import padding_episodes,merge_episodes, ReplayBuffer,ABR_ReplayBuffer,JAL_ReplayBuffer
from common.utils import is_onpolicy, is_coopmarl
import numpy as np
import os
import pickle
import torch
class Runner:
    def __init__(self, env,evaluate_settings, args):
        #torch.manual_seed(args.random_seed)
        #np.random.seed(args.random_seed)
        self.env = env
        #the dim info of env is already calculated and contained in args
        #create multi-agents if network is not reused
        if args.n_agents > 1 and args.reuse_network == False and is_coopmarl(args.alg) == False:
            self.agents = []
            for agent_id in range(args.n_agents):
                self.agents.append(Agents(args))
        elif args.n_agents == 1 and args.alg.startswith("abr"):
            self.agents = ABR_Agents(args)
        elif args.n_agents == 1 and args.alg.startswith("jal"):
            self.agents = JAL_Agents(args)
        else:
            self.agents = Agents(args)
            
        self.evaluate_settings = evaluate_settings
        if args.alg.startswith("seq"):
            self.rolloutWorker =  RolloutWorker_SEQ(env, self.agents, args)
        elif args.alg.startswith("ma"):
            self.rolloutWorker =  RolloutWorker_MA(env, self.agents, args)
        elif args.alg.startswith("abr"):
            self.rolloutWorker = RolloutWorker_ABR(env, self.agents, args)
        elif args.alg.startswith("jal"):
            self.rolloutWorker = RolloutWorker_JAL(env, self.agents, args)
        else:
            raise Exception("unkown envionrment type")
            
        if is_onpolicy(args.alg) == False:
            num_of_nodes = None
            if args.state_type !=0:
                num_of_nodes = self.env.num_of_nodes
            if args.alg.startswith("abr"):
                self.buffer = ABR_ReplayBuffer(args,100, num_of_nodes)
            if args.alg.startswith("jal"):
                self.buffer = JAL_ReplayBuffer(args,100, num_of_nodes)
            else:
                self.buffer = ReplayBuffer(args, 100, num_of_nodes)
            
        self.args = args
        if self.args.n_agents > 1 and self.args.reuse_network == False and is_coopmarl(self.args.alg) == False:
            self.output_dir = self.agents[0].policy.output_dir
        else:
            self.output_dir = self.agents.policy.output_dir
        
        #record the training rewards and evaluate rewards
        self.training_rewards = None
        self.evaluate_rewards = None
        self.evaluate_paths = None
        
    def run(self,plot=True):
        #record the result
        training_rewards = []
        evaluate_rewards = []
        evaluate_paths = []
        
        train_steps = 0
        for epoch in range(self.args.n_epoch):
            episodes = []
            for episode_idx in range(self.args.n_episodes):
                if self.args.train_with_evaluation_settings:
                    #Use the evaluate settings as training
                    paths_specific = np.random.choice(self.evaluate_settings)
                else:
                    #For random budget and start locations
                    paths_specific = {'vses':np.random.choice(self.args.chargingnodes,self.args.path_number).tolist(), 
                                      'Bs':np.random.choice(list(range(*self.args.B_range)),self.args.path_number).tolist(), 
                                      'chargingnodes': self.args.chargingnodes}
                #Generat an episode
                episode, episode_reward, episode_paths = self.rolloutWorker.generate_episode(paths_specific,plot=False)
                episodes.append(episode)
                training_rewards.append(episode_reward)
            
            if is_onpolicy(self.args.alg):
                
                if self.args.state_type ==0:
                    training_batch = padding_episodes(episodes)
                else:
                    training_batch = merge_episodes(episodes)
                    
                if self.args.n_agents > 1 and self.args.reuse_network == False and is_coopmarl(self.args.alg) == False:
                    for agent_id in range(self.args.n_agents):
                        agent_batch = self._get_agent_batch(training_batch,agent_id)
                        self.agents[agent_id].train(agent_batch, train_steps,self.rolloutWorker.epsilon,agent_id)
                else:
                    self.agents.train(training_batch, train_steps,self.rolloutWorker.epsilon)

                    
                train_steps += 1
            else:
                self.buffer.store_episodes(episodes)
                for train_step in range(self.args.n_train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    if self.args.n_agents > 1 and self.args.reuse_network == False and is_coopmarl(self.args.alg) == False:

                        for agent_id in range(self.args.n_agents):
                            agent_batch = self._get_agent_batch(mini_batch,agent_id)
                            self.agents[agent_id].train(agent_batch, train_steps,agent_id=agent_id)
                    else:
                        self.agents.train(mini_batch, train_steps)
                    train_steps +=1
                    
            if epoch % self.args.evaluate_cycle == 0:
                rewards,paths = self.evaluate()
                evaluate_rewards.append(rewards)
                evaluate_paths.append(paths)
                if epoch % 100 == 0:
                    print('at epoch {}, trained {} steps, evalute rewards:{}'.format(epoch, train_steps, rewards))
                
        record_name = 'records_st{}.pickle'.format(self.args.state_type)
        if self.args.alg.startswith('ma') or self.args.alg.startswith('abr'):
            record_name = 'records_rt{}_st{}.pickle'.format(self.args.reward_type, self.args.state_type)
        with open(os.path.join(self.output_dir,record_name),'wb') as fout:
            pickle.dump({'arg':self.args, 'training_rewards':training_rewards,'evaluate_rewards':evaluate_rewards,'evaluate_paths':evaluate_paths},fout)
        
        if plot==True:
            pass
        #record the result
        self.training_rewards = training_rewards
        self.evaluate_rewards = evaluate_rewards
        self.evaluate_paths =evaluate_paths
        #return training_rewards, evaluate_rewards
                
    def evaluate(self):
        rewards = []
        paths = []
        if self.args.random_evaluation == False:
            for evaluate_specific in self.evaluate_settings:
                episode, episode_reward, episode_paths = self.rolloutWorker.generate_episode(evaluate_specific,evaluate=True)
                rewards.append(episode_reward)
                paths.append(episode_paths)
        else:
            evaluate_specific = {'vses':np.random.choice(self.args.chargingnodes,self.args.path_number).tolist(), 
                                 'Bs':np.random.choice(list(range(*self.args.B_range)),self.args.path_number).tolist(), 
                                 'chargingnodes': self.args.chargingnodes}

            episode, episode_reward, episode_paths = self.rolloutWorker.generate_episode(evaluate_specific,evaluate=True)
            rewards.append(episode_reward)
            paths.append(episode_paths)
                
        return rewards, paths
    
    def _get_agent_batch(self,batch,agent_id):
        batch_agent = {}
        if self.args.state_type==0:
            for key in batch.keys():
                if key in ['s','s_next','u','u_onehot','valid_u','valid_u_next']:
                    batch_agent[key] = batch[key][:,:,[agent_id]]
                elif key=='r':
                    if batch[key].shape[-1] > 1:
                        batch_agent[key] = batch[key][:,:,[agent_id]]
                    else:
                        batch_agent[key] = batch[key]
                else:
                    #padded, and terminated
                    batch_agent[key] = batch[key]
        else:
            #import pdb;pdb.set_trace()
            for key in batch.keys():
                if key in ['s','s_next','u','u_onehot','valid_u','valid_u_next','nodes_info','nodes_info_next']:
                    batch_agent[key] = batch[key][:,[agent_id]]
                elif key=='r':
                    if batch[key].shape[-1] > 1:
                        batch_agent[key] = batch[key][:,[agent_id]]
                    else:
                        batch_agent[key] = batch[key]
                else:
                    #padded, and terminated
                    batch_agent[key] = batch[key]            
        return batch_agent
                
