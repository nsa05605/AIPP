#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:25:11 2020

@author: yongyongwei
"""

import numpy as np
import torch
from torch.nn.functional import one_hot
from copy import deepcopy
import matplotlib.pyplot as plt
from common.utils import  is_coopmarl
import itertools

class RolloutWorker_SEQ:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.args = args
        
        self.n_agents = args.n_agents
        assert self.n_agents == 1
    
        self.n_actions = args.n_actions
        self.nn_input_dim = args.nn_input_dim
        
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        
        print('Init Rollout worker as sequential execution')
        
    def generate_episode(self, paths_specific, evaluate = False,plot=False):
        s, ns, u , r, valid_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        state_nodesinfo, episode_reward = self.env.reset(paths_specific, state_type = self.args.state_type)
        state, nodesinfo = state_nodesinfo
        
        step = 0
        terminated = False
        
        if self.args.state_type ==0:
            self.agents.policy.init_hidden(1)
        
        epsilon = 0 if evaluate else self.epsilon
        epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        
        while not terminated:
            if plot==True:
                plt.close('all')
            
            actions, actions_onehot, valid_actions_onehot = [], [], []
            
            valid_action = self.env.get_valid_actions()
            if self.args.unseen_priority == True:
                visited_nodes = self.env.get_robot_visited_nodes()
                unseen_nodes = set(valid_action).difference(set(visited_nodes))
                if len(unseen_nodes) > 0:
                    valid_action = list(unseen_nodes)
            if self.args.state_type ==0:
                nn_input = state
            else:
                nn_input = np.concatenate((nodesinfo,state))
            action = self.agents.choose_action(nn_input, 0, valid_action, epsilon, evaluate)
            
            actions.append(action)
            actions_onehot.append(one_hot(torch.tensor(action), self.n_actions).numpy())
            valid_action_onehot = np.zeros(self.n_actions)
            valid_action_onehot[valid_action] = 1
            valid_actions_onehot.append(valid_action_onehot)
            
            pre_state_nodeinfo, vs, reward, post_state_nodeinfo, terminated = self.env.step(action, state_type = self.args.state_type)
            #pre_state, pre_nodeinfo = pre_state_nodeinfo
            post_state, post_nodeinfo = post_state_nodeinfo
            
            #Add one dimension for the single agent
            s.append([state])
            u.append(np.reshape(actions,[self.n_agents,1]))
            u_onehot.append(actions_onehot)
            valid_u.append(valid_actions_onehot)
            r.append([reward])
            terminate.append([terminated])
            episode_reward += reward
            
            if self.args.state_type ==0:
                padded.append([0.])
            else:
                ns.append([nodesinfo])
    
            step +=1
            state = post_state
            nodesinfo = post_nodeinfo
       
            if plot==True:
                self.env.plot_graph(charging_nodes = paths_specific['chargingnodes'],paths_nodes = self.env.partial_paths,samples_pos=self.env.sample_set_splitted,arrow=True,vis_paths=False,vis_sample_edges=False)
                plt.pause(0.3)
        
        s.append([state])
        s_next = s[1:]
        s = s[:-1]
        
        if self.args.state_type !=0:
            ns.append([nodesinfo])
            ns_next = ns[1:]
            ns= ns[:-1]
        
        #Note here add one dimension manually since only one agent
        valid_u.append([np.zeros(self.n_actions)])
        valid_u_next = valid_u[1:]
        valid_u = valid_u[:-1]

        if self.args.state_type ==0:
            episode = dict(s = deepcopy(s),
                           u = deepcopy(u),
                           r = deepcopy(r),
                           valid_u = deepcopy(valid_u),
                           s_next = deepcopy(s_next),
                           valid_u_next = deepcopy(valid_u_next),
                           u_onehot = deepcopy(u_onehot),
                           padded = deepcopy(padded),
                           terminated = deepcopy(terminate))
        else:
            episode = dict(s = deepcopy(s),
                           u = deepcopy(u),
                           r = deepcopy(r),
                           valid_u = deepcopy(valid_u),
                           s_next = deepcopy(s_next),
                           valid_u_next = deepcopy(valid_u_next),
                           u_onehot = deepcopy(u_onehot),
                           nodes_info = deepcopy(ns),
                           nodes_info_next=deepcopy(ns_next),
                           terminated = deepcopy(terminate))         
                    
        for key in episode.keys():
            episode[key] = np.array(episode[key])

            
        if not evaluate:
            self.epsilon = epsilon
            
        return episode, episode_reward, deepcopy(self.env.partial_paths)
            


class RolloutWorker_MA:
    def __init__(self, env, agents, args):
        self.env = env
        #note agents may be a list if network is not reused
        self.agents = agents
        self.args = args
        
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.nn_input_dim = args.nn_input_dim
        
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
    
        if self.args.alg.startswith("jal") == True:
            assert self.n_agents == 1
        else:
            assert self.n_agents > 1
        
        print('Init Rollout worker as multiagent execution, reward type:{}'.format(args.reward_type))
        
    def generate_episode(self, paths_specific, evaluate = False, plot= False):
        s, ns, u , r, valid_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        state_nodesinfo,  episode_reward = self.env.reset(paths_specific, reward_type = self.args.reward_type, state_type = self.args.state_type)
        state, nodesinfo = state_nodesinfo
        #Note in MARL enviornment, reward is a list, length may be 1 or n_agents
        episode_reward = np.sum(episode_reward)
        episode_reward += self.env.pilot_reward

        step = 0
        terminated = False
                
        if self.args.state_type ==0:
            if self.n_agents > 1 and self.args.reuse_network == False and is_coopmarl(self.args.alg)==False:
                assert len(self.agents) == self.n_agents
                for agent_id in range(self.n_agents):
                    self.agents[agent_id].policy.init_hidden(1)
            else:
                self.agents.policy.init_hidden(1)

            
        epsilon = 0 if evaluate else self.epsilon
        epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        
        while not terminated:
            if plot==True:
                plt.close('all')           
            
            states_nn, actions, actions_onehot, valid_actions_onehot =[], [], [], []
            
            for agent_id in range(self.n_agents):
                #******* get valid action************
                valid_action = self.env.get_valid_actions(agent_id)
                if self.args.unseen_priority == True:
                    visited_nodes = self.env.get_robot_visited_nodes(agent_id)
                    unseen_nodes = set(valid_action).difference(set(visited_nodes))
                    if len(unseen_nodes) > 0:
                        valid_action = list(unseen_nodes)
                
                #******not coop marl, separate network
                if self.n_agents > 1 and self.args.reuse_network == False and is_coopmarl(self.args.alg)==False:
                    state_full = np.concatenate(state)
                    if self.args.full_observable:
                        if self.args.state_type==0:
                            nn_input = state_full
                        else:
                            nn_input = np.concatenate((nodesinfo, state_full))
                        action = self.agents[agent_id].choose_action(nn_input, 0, valid_action,epsilon, evaluate)
                    else:
                        if self.args.state_type ==0:
                            nn_input = state[agent_id]
                        else:
                            nn_input = np.concatenate((nodesinfo, state[agent_id]))
                        action = self.agents[agent_id].choose_action(nn_input, 0, valid_action,epsilon, evaluate)
                
                #*******coop marl*********************
                elif is_coopmarl(self.args.alg)==True:
                    state_full = np.concatenate((np.concatenate(state),one_hot(torch.tensor(agent_id),self.n_agents)))
                    if self.args.full_observable:
                        if self.args.reuse_network:
                            nn_input = state_full
                            if self.args.state_type !=0:
                                nn_input = np.concatenate((nodesinfo, nn_input))
                        else:
                            nn_input = np.concatenate(state)
                            if self.args.state_type !=0:
                                nn_input = np.concatenate((nodesinfo, nn_input))
                    else:
                        if self.args.reuse_network:
                            nn_input = np.concatenate((state[agent_id],one_hot(torch.tensor(agent_id),self.n_agents)))
                            if self.args.state_type !=0:
                                nn_input = np.concatenate((nodesinfo, nn_input))
                        else:
                            nn_input = state[agent_id]
                            if self.args.state_type!=0:
                                nn_input = np.concatenate((nodesinfo, nn_input))
                    action = self.agents.choose_action(nn_input, agent_id, valid_action, epsilon, evaluate)
                
                #******not coop marl, reused network
                else:
                    state_full = np.concatenate((np.concatenate(state),one_hot(torch.tensor(agent_id),self.n_agents)))
                    if self.args.full_observable:
                        nn_input = state_full
                        if self.args.state_type !=0:
                            nn_input = np.concatenate((nodesinfo,  nn_input))
                        action = self.agents.choose_action(nn_input, agent_id, valid_action,epsilon, evaluate)
                    else:
                        local_observation = np.concatenate((state[agent_id],one_hot(torch.tensor(agent_id),self.n_agents)))
                        if self.args.state_type !=0:
                            local_observation = np.concatenate((nodesinfo, local_observation))
                        action = self.agents.choose_action(local_observation, agent_id, valid_action,epsilon, evaluate)
                
                states_nn.append(state_full)
                actions.append(action)
                actions_onehot.append(one_hot(torch.tensor(action),self.n_actions).numpy())   
                valid_action_onehot = np.zeros(self.n_actions)
                valid_action_onehot[valid_action] = 1
                valid_actions_onehot.append(valid_action_onehot)
                
            pre_state_nodesinfo, vs, reward, post_state_nodesinfo, terminated = self.env.step(actions,reward_type = self.args.reward_type, state_type = self.args.state_type)
            post_state, post_nodesinfo = post_state_nodesinfo
            
            s.append(states_nn)            
            u.append(np.reshape(actions,[self.n_agents,1]))
            u_onehot.append(actions_onehot)
            valid_u.append(valid_actions_onehot)
            
            r.append(reward)
            terminate.append([terminated])
            if self.args.state_type ==0:
                padded.append([0.])
            else:
                ns.append([nodesinfo])
                
            episode_reward += np.sum(reward)
    
            step +=1
            state = post_state
            nodesinfo = post_nodesinfo

            if plot==True:
                paths_nodes = [rob.node_tracker for rob in self.env.robots]
                samples_pos = [rob.local_samples for rob in self.env.robots]
                self.env.plot_graph(paths_specific['chargingnodes'],paths_nodes,samples_pos,arrow=True,vis_paths=False,vis_sample_edges=False)
                plt.pause(0.3)
                
        states_nn, valid_actions_onehot = [], []
        for agent_id in range(self.n_agents):
            if self.n_agents > 1 and self.args.reuse_network == False and is_coopmarl(self.args.alg)==False:
                states_nn.append(np.concatenate(state))
            else:
                states_nn.append(np.concatenate((np.concatenate(state),one_hot(torch.tensor(agent_id),self.n_agents))))
            valid_actions_onehot.append(np.zeros(self.n_actions))
            
        s.append(states_nn)
        s_next = s[1:]
        s = s[:-1]
        
        if self.args.state_type !=0:
            ns.append([nodesinfo])
            ns_next = ns[1:]
            ns = ns[:-1]
            
        
        valid_u.append(valid_actions_onehot)
        valid_u_next = valid_u[1:]
        valid_u = valid_u[:-1]
        
        if self.args.state_type ==0:
            episode = dict(s = deepcopy(s),
                           u = deepcopy(u),
                           r = deepcopy(r),
                           valid_u = deepcopy(valid_u),
                           s_next = deepcopy(s_next),
                           valid_u_next = deepcopy(valid_u_next),
                           u_onehot = deepcopy(u_onehot),
                           padded = deepcopy(padded),
                           terminated = deepcopy(terminate))
        else:
            episode = dict(s = deepcopy(s),
                           u = deepcopy(u),
                           r = deepcopy(r),
                           valid_u = deepcopy(valid_u),
                           s_next = deepcopy(s_next),
                           valid_u_next = deepcopy(valid_u_next),
                           u_onehot = deepcopy(u_onehot),
                           nodes_info = deepcopy(ns),
                           nodes_info_next = deepcopy(ns_next),
                           terminated = deepcopy(terminate))            
       
        for key in episode.keys():
            episode[key] = np.array(episode[key])
            
        if not evaluate:
            self.epsilon = epsilon
            
        return episode, episode_reward, [rob.node_tracker for rob in self.env.robots]
    
    
            

class RolloutWorker_ABR:
    def __init__(self, env, agents, args):
        self.env = env
        #note agents may be a list if network is not reused
        self.agents = agents
        self.args = args
        
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.nn_input_dim = args.nn_input_dim
        
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        
        
        #in this case self.n_agents == 1 is always True
        assert self.n_agents == 1
        
        print('Init Rollout worker as multiagent execution, reward type:{}'.format(args.reward_type))
        
    def generate_episode(self, paths_specific, evaluate = False, plot= False):
        s, ns, u , r, valid_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        state_nodesinfo,  episode_reward = self.env.reset(paths_specific, reward_type = self.args.reward_type, state_type = self.args.state_type)
        state, nodesinfo = state_nodesinfo
        #Note in MARL enviornment, reward is a list, length may be 1 or n_agents
        episode_reward = np.sum(episode_reward)
        episode_reward += self.env.pilot_reward

        step = 0
        terminated = False
                
        if self.args.state_type ==0:
            self.agents.policy.init_hidden(1)
            
        
        epsilon = 0 if evaluate else self.epsilon
        epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        
        while not terminated:
            if plot==True:
                plt.close('all')           
            
            actions, actions_onehot, valid_actions_onehot =[], [], []
            
            valid_actions=[]
            for robot_id in range(self.args.n_robots):
                #******* get valid actions for each branch************
                valid_action = self.env.get_valid_actions(robot_id)
                if self.args.unseen_priority == True:
                    visited_nodes = self.env.get_robot_visited_nodes(robot_id)
                    unseen_nodes = set(valid_action).difference(set(visited_nodes))
                    if len(unseen_nodes) > 0:
                        valid_action = list(unseen_nodes)
                valid_actions.append(valid_action)
                valid_action_onehot = np.zeros(self.n_actions)
                valid_action_onehot[valid_action] = 1
                valid_actions_onehot.append(valid_action_onehot)
            
            state_full = np.concatenate(state)
            nn_input = state_full
            if self.args.state_type !=0:
                nn_input = np.concatenate((nodesinfo, nn_input))
            actions = self.agents.choose_actions(nn_input, 0, valid_actions, epsilon, evaluate)
            for action in actions:
                actions_onehot.append(one_hot(torch.tensor(action),self.n_actions).numpy())

            pre_state_nodesinfo, vs, reward, post_state_nodesinfo, terminated = self.env.step(actions,reward_type = self.args.reward_type)
            post_state, post_nodesinfo = post_state_nodesinfo
            
            s.append([state_full])
            u.append(np.reshape(actions,[self.args.n_robots,1]))
            u_onehot.append(actions_onehot)
            valid_u.append(valid_actions_onehot)
            
            r.append(reward)
            terminate.append([terminated])
            
            if self.args.state_type ==0:
                padded.append([0.])
            else:
                ns.append([nodesinfo])
                
            episode_reward += np.sum(reward)
    
            step +=1
            state = post_state
            nodesinfo = post_nodesinfo
            
            if plot==True:
                paths_nodes = [rob.node_tracker for rob in self.env.robots]
                samples_pos = [rob.local_samples for rob in self.env.robots]
                self.env.plot_graph(paths_specific['chargingnodes'],paths_nodes,samples_pos,arrow=True,vis_paths=False,vis_sample_edges=False)
                plt.pause(0.3)
                
        #deal with the last state and action
        s.append([np.concatenate(state)])
        valid_actions_onehot = []
        for robot_id in range(self.args.n_robots):
             valid_actions_onehot.append(np.zeros(self.n_actions))
        valid_u.append(valid_actions_onehot)
            
        s_next = s[1:]
        s = s[:-1]
        
        if self.args.state_type !=0:
            ns.append([nodesinfo])
            ns_next = ns[1:]
            ns = ns[:-1]
        
        valid_u_next = valid_u[1:]
        valid_u = valid_u[:-1]
        
        if self.args.state_type ==0:
            episode = dict(s = deepcopy(s),
                           u = deepcopy(u),
                           r = deepcopy(r),
                           valid_u = deepcopy(valid_u),
                           s_next = deepcopy(s_next),
                           valid_u_next = deepcopy(valid_u_next),
                           u_onehot = deepcopy(u_onehot),
                           padded = deepcopy(padded),
                           terminated = deepcopy(terminate))
           
        else:
            episode = dict(s = deepcopy(s),
                           u = deepcopy(u),
                           r = deepcopy(r),
                           valid_u = deepcopy(valid_u),
                           s_next = deepcopy(s_next),
                           valid_u_next = deepcopy(valid_u_next),
                           u_onehot = deepcopy(u_onehot),
                           nodes_info = deepcopy(ns),
                           nodes_info_next = deepcopy(ns_next),
                           terminated = deepcopy(terminate))            
            
        for key in episode.keys():
            episode[key] = np.array(episode[key])
            
        if not evaluate:
            self.epsilon = epsilon
            
        return episode, episode_reward, [rob.node_tracker for rob in self.env.robots]
    
    




class RolloutWorker_JAL:
    def __init__(self, env, agents, args):
        self.env = env
        #note agents may be a list if network is not reused
        self.agents = agents
        self.args = args
        
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.nn_input_dim = args.nn_input_dim
        
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        
        
        #in this case self.n_agents == 1 is always True
        assert self.n_agents == 1
        
        print('Init Rollout worker as JAL, reward type:{}'.format(args.reward_type))

    def _get_joint_action_indices(self,valid_actions):
        action_size = self.args.single_robot_action_size
        indices=[]
        for joint_action in itertools.product(*valid_actions):
            index=0
            for i,action_num in enumerate(joint_action):
                index += (action_size ** i) * action_num
            indices.append(index)
        return indices
    

    def _get_joint_action_index(self,actions):
        action_size = self.args.single_robot_action_size
        index=0
        for i,action_num in enumerate(actions):
            index += (action_size ** i) * action_num
        return index
        
    def generate_episode(self, paths_specific, evaluate = False, plot= False):
        s, ns, u , r, valid_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        u_index=[]
        state_nodesinfo,  episode_reward = self.env.reset(paths_specific, reward_type = self.args.reward_type, state_type = self.args.state_type)
        state, nodesinfo = state_nodesinfo
        #Note in MARL enviornment, reward is a list, length may be 1 or n_agents
        episode_reward = np.sum(episode_reward)
        episode_reward += self.env.pilot_reward

        step = 0
        terminated = False
                
        if self.args.state_type ==0:
            self.agents.policy.init_hidden(1)
            
        
        epsilon = 0 if evaluate else self.epsilon
        epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        
        while not terminated:
            if plot==True:
                plt.close('all')           
            
            actions, valid_actions_onehot =[], []
            
            valid_actions=[]
            for robot_id in range(self.args.n_robots):
                #******* get valid actions for each branch************
                valid_action = self.env.get_valid_actions(robot_id)
                if self.args.unseen_priority == True:
                    visited_nodes = self.env.get_robot_visited_nodes(robot_id)
                    unseen_nodes = set(valid_action).difference(set(visited_nodes))
                    if len(unseen_nodes) > 0:
                        valid_action = list(unseen_nodes)
                valid_actions.append(valid_action)
                
            
            valid_action_joint_index = self._get_joint_action_indices(valid_actions)
            valid_action_joint_onehot = np.zeros(self.n_actions)
            valid_action_joint_onehot[valid_action_joint_index] = 1

            state_full = np.concatenate(state)
            nn_input = state_full
            if self.args.state_type !=0:
                nn_input = np.concatenate((nodesinfo, nn_input))
            actions = self.agents.choose_actions(nn_input, 0, valid_actions, epsilon, evaluate)
            
            actions_index = self._get_joint_action_index(actions)
            u_index.append([actions_index])
            actions_onehot=one_hot(torch.tensor(actions_index),self.n_actions).numpy()

            pre_state_nodesinfo, vs, reward, post_state_nodesinfo, terminated = self.env.step(actions,reward_type = self.args.reward_type, state_type = self.args.state_type)
            post_state, post_nodesinfo = post_state_nodesinfo
            
            s.append([state_full])
            u.append(np.reshape(actions,[self.args.n_robots,1]))
            u_onehot.append([actions_onehot])
            valid_u.append([valid_action_joint_onehot])
            
            r.append(reward)
            terminate.append([terminated])
            
            if self.args.state_type ==0:
                padded.append([0.])
            else:
                ns.append([nodesinfo])
                
            episode_reward += np.sum(reward)
    
            step +=1
            state = post_state
            nodesinfo = post_nodesinfo
            
            if plot==True:
                paths_nodes = [rob.node_tracker for rob in self.env.robots]
                samples_pos = [rob.local_samples for rob in self.env.robots]
                self.env.plot_graph(paths_specific['chargingnodes'],paths_nodes,samples_pos,arrow=True,vis_paths=False,vis_sample_edges=False)
                plt.pause(0.3)
                
        #deal with the last state and action
        s.append([np.concatenate(state)])
        valid_u.append([np.zeros(self.n_actions)])
            
        s_next = s[1:]
        s = s[:-1]
        
        if self.args.state_type !=0:
            ns.append([nodesinfo])
            ns_next = ns[1:]
            ns = ns[:-1]
        
        valid_u_next = valid_u[1:]
        valid_u = valid_u[:-1]
        
        if self.args.state_type ==0:
            episode = dict(s = deepcopy(s),
                           u = deepcopy(u),
                           u_index = deepcopy(u_index),
                           r = deepcopy(r),
                           valid_u = deepcopy(valid_u),
                           s_next = deepcopy(s_next),
                           valid_u_next = deepcopy(valid_u_next),
                           u_onehot = deepcopy(u_onehot),
                           padded = deepcopy(padded),
                           terminated = deepcopy(terminate))
           
        else:
            episode = dict(s = deepcopy(s),
                           u = deepcopy(u),
                           u_index = deepcopy(u_index),
                           r = deepcopy(r),
                           valid_u = deepcopy(valid_u),
                           s_next = deepcopy(s_next),
                           valid_u_next = deepcopy(valid_u_next),
                           u_onehot = deepcopy(u_onehot),
                           nodes_info = deepcopy(ns),
                           nodes_info_next = deepcopy(ns_next),
                           terminated = deepcopy(terminate))            
            
        for key in episode.keys():
            episode[key] = np.array(episode[key])
            
            
        #Merge the reward to team reward
        episode['r'] = np.sum(episode['r'],axis=-1).reshape(-1,1)
        
        if not evaluate:
            self.epsilon = epsilon

        return episode, episode_reward, [rob.node_tracker for rob in self.env.robots]
    











