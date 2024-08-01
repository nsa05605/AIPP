#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 23:30:10 2020
Note: For the cooperative case, the recorded reward has minor 
differences compared with evaluated reward given the whole path,
because condition_entropy function has numerical difference when 
the order of locations (matrix row) changes.
@author: yongyongwei
"""


import sys
sys.dont_write_bytecode = True
import numpy as np
from scipy.spatial.distance import euclidean
import copy
from matplotlib import pyplot as plt
import itertools
from copy import deepcopy
from common.utils import load_config, GraphTools

class Robot():
    def __init__(self,ID,areaname,config,vs,B,chargingnodes,g):
        self.ID = ID
        self.vs = vs
        self.B = B
        self.chargingnodes = chargingnodes
        self.sample_interval = config[areaname]['sample_interval']
        self.g = g
        self.reset()
        
    def reset(self):
        """Reset the status of the robot"""
        self.current_node = self.vs
        self.action_tracker = [self.vs] 
        self.node_tracker = [self.vs]
        self.trajectory = [self.g.nodes[self.vs]['pos']+[self.B]]
        self.obs = self.g.nodes[self.vs]['pos']+[self.B]
        self.local_samples = [self.g.nodes[self.vs]['pos']]
        self.increased_samples = [self.g.nodes[self.vs]['pos']]
        self.rb = self.B
        self.is_active = True

    def execute(self,a):
        """Execute an action for the robot"""
        self.increased_samples = []
        #noop action  index is defined as the number of nodes in the graph
        self.action_tracker.append(a)
        if a == len(self.g.nodes()):
            assert self.is_active == False
            self.trajectory.append(self.g.nodes[self.current_node]['pos']+[self.rb])
            self.obs = self.g.nodes[self.current_node]['pos']+[self.rb]
            return
        else:
            assert self.is_active == True,"robot %d already stopped!" % self.ID
            self.node_tracker.append(a)
            a_cost = euclidean(self.g.nodes[a]['pos'],self.g.nodes[self.current_node]['pos'])
            self.rb -= a_cost
            self.trajectory.append(self.g.nodes[a]['pos']+[self.rb])
            self.obs = self.g.nodes[a]['pos']+[self.rb]

            nx_pos =  self.g.nodes[a]['pos']
            cur_pos = self.g.nodes[self.current_node]['pos']
            new_sample_num = int(np.floor(a_cost*1.0/self.sample_interval))

            for j in range(1,new_sample_num + 1):
                tmp_x = cur_pos[0] + (nx_pos[0] - cur_pos[0])*1.0*j/new_sample_num
                tmp_y = cur_pos[1] + (nx_pos[1] - cur_pos[1])*1.0*j/new_sample_num
                self.local_samples.append([tmp_x,tmp_y])
                self.increased_samples.append([tmp_x,tmp_y])

            self.current_node = a
            if a in self.chargingnodes:
                self.is_active = False

            return
        

class EnvMARL(GraphTools):
    def __init__(self,areaname,config):
        print('Init envionrment for multi-agent coorperative RL')
        GraphTools.__init__(self,areaname,config)
        self.config = deepcopy(config)
        
    def reset(self,paths_specific, reward_type = 0,state_type = 0):
        """Reset  environment for RL"""  
        assert len(paths_specific['vses']) == len(paths_specific['Bs'])
        self.robot_num = len(paths_specific['vses'])
        self.robots = []
        for i in range(self.robot_num):
            self.robots.append(Robot(i, self.areaname, self.config, paths_specific['vses'][i], paths_specific['Bs'][i],paths_specific['chargingnodes'],self.g))

        self.paths_specific = deepcopy(paths_specific)
        #print('reset with {} robots, start loations:{}, budgets:{}'.format(self.robot_num,str(paths_specific['vses']),str(paths_specific['Bs'])))

    
        #track the visit count of each nodes
        self.node_visit_counts = np.zeros(self.num_of_nodes)   
        self.node_visit_counts[paths_specific['vses']]=1
        
        #sample set, initially equals to pilot samples
        self.sample_set = copy.copy(self.pilot_pos)
        
        #track the variance of each nodes
        self.current_conden, self.current_node_vars = self.condition_entropy(self.sample_set,self.node_locs,ret_Bvar=True)
        #In this case init reward contains pilot reward, calculated separatly
        self.pilot_reward = self.node_en - self.current_conden 
        #Calculate team reward stored in self.rt
        self.reward_team()
        self.init_reward = self.rt 
        if reward_type == 0:
            return self.state_encoding(state_type),[self.init_reward]
        else:
            return self.state_encoding(state_type), np.repeat(self.init_reward*1.0/self.robot_num,self.robot_num).tolist()
    
    def reward_team(self):
        """calculate team reward as a whole"""
        newsamples_all = [rob.increased_samples for rob in self.robots]
        newsamples_all = np.array(list(itertools.chain(*newsamples_all)))
        if len(newsamples_all) == 0:
            print('Alert: No increased samples from any robots!')
            rt = 0
        else:
            self.sample_set = np.row_stack((self.sample_set,newsamples_all))
            conden, self.current_node_vars = self.condition_entropy(self.sample_set,self.node_locs, ret_Bvar=True)
            rt = self.current_conden - conden
            self.current_conden = conden
        
        self.rt = rt
        
        return [self.rt]
    
        
    def reward_member1(self):
        """Credit assignment for multiple robot reward, based on sequential act"""
        r_list = []
        last_conden = self.current_conden
        for rob in self.robots:
            if len(rob.increased_samples) == 0:
                r_list.append(0)
            else:
                self.sample_set = np.row_stack((self.sample_set ,rob.increased_samples))
                conden, self.current_node_vars = self.condition_entropy(self.sample_set,self.node_locs, ret_Bvar=True)
                r_list.append(self.current_conden - conden)
                self.current_conden = conden
        rt = last_conden - self.current_conden    
        self.rt = rt
        
        return r_list
    
    
    
    def reward_member2(self,regularize=True):
        """Credit assignment for multiple robot reward, based on difference reward
        Args:
            regularize, whether regularize the reward such that sum equals to team
            returns: list of reward for each agent
        """
        r_list = []
        #merge the samples from agents
        newsamples_all = [rob.increased_samples for rob in self.robots]
        newsamples_all = np.array(list(itertools.chain(*newsamples_all)))  

        #get team reward
        all_samples = np.row_stack((self.sample_set ,newsamples_all))
        conden,self.current_node_vars = self.condition_entropy(all_samples, self.node_locs,ret_Bvar=True)
        rt = self.current_conden - conden
        
        for i,rob in zip(range(len(self.robots)),self.robots):
            if len(rob.increased_samples) == 0:
                r_list.append(0)
            else:
                absent_samples = [self.robots[j].increased_samples for j in range(len(self.robots)) if j!=i]
                absent_samples = np.array(list(itertools.chain(*absent_samples))) 
                if len(absent_samples)==0:
                    absent_samples = self.sample_set
                else:
                    absent_samples = np.row_stack((self.sample_set ,absent_samples))
                absent_conden = self.condition_entropy(absent_samples,self.node_locs)
                r_list.append(absent_conden - conden)      
        
        #pamanently set the related attributes
        self.sample_set = np.row_stack((self.sample_set ,newsamples_all))
        self.current_conden = conden
        if regularize == True:
            r_list_sum = np.sum(r_list)
            ret = [rt * r / r_list_sum for r in r_list]
        else:
            ret = r_list
            
        self.rt = rt
        
        return ret

    def step(self,acts,reward_type = 0, state_type=0):
        """Take actions  for the team
        Args:
            acts: the actions for all robot
            reward_type: type of reward signal
                0: team reward split among active-robot(must be active, not dead)
                1: each agent a reward based on seqential allocation
                2: each agent a reward based on difference reward
        """ 
        assert len(acts) == len(self.robots)
        #must at least one robot is active
        assert np.sum([rob.is_active for rob in self.robots]) != 0
        robots_status = [rob.is_active for rob in self.robots]
        #record the current state
        pre_state = self.state_encoding(state_type)
        for i,rob in enumerate(self.robots):
            rob.execute(acts[i])
        
        #track the number of visit count (exclude dummy node)
        self.node_visit_counts[[v for v in acts if v != self.num_of_nodes]]+=1
        
        is_done = True if np.sum([rob.is_active for rob in self.robots]) == 0 else False
        
        if reward_type == 0:
            reward = self.reward_team()
            mean_reward = np.sum(reward)/np.sum(robots_status)
            reward = [mean_reward if rob_status == True else 0 for rob_status in robots_status]
        elif reward_type == 1:
            reward = self.reward_member1()
        elif reward_type == 2:
            reward = self.reward_member2()
        else:
            raise Exception('Invalid reward type')  
        post_state = self.state_encoding(state_type)
        return pre_state, acts, reward, post_state, is_done


    def state_encoding(self, state_type=0):
        """Capture the state of the MARL system
           Note: the shape would be  N * 3, does include the agent ID
           
           state_type:
               0: robot pos, remaining budgets
               1: robot pos, remaining budgets + nodes visit count
               2: robot pos, remaining budgets + nodes variance
        """
        encodings = [np.array(rob.obs) for rob in self.robots]
        if state_type == 0:
            return (deepcopy(encodings), None)
        elif state_type == 1:
            return (deepcopy(encodings), deepcopy(self.node_visit_counts))
        elif state_type == 2:
            return (deepcopy(encodings), deepcopy(self.current_node_vars))
        else:
            raise Exception('Unkown state type')
    
    def get_robot_visited_nodes(self, i):
        return self.robots[i].node_tracker[:]
    
    def get_valid_actions(self, i):
        assert  self.robots[i].rb >= 0
        
        if self.robots[i].is_active == False:
            return [self.num_of_nodes]
        else:
            current_node = self.robots[i].current_node
            rb = self.robots[i].rb
            valid_acts=[]
            nbs = self.g.nodes[current_node]['adj']
            for vt in self.paths_specific['chargingnodes']:
                min_cost = [self.node_distance(current_node, nb) + self.g_cost_dict[(nb, vt)] for nb in nbs ]
                valid_acts.extend([nbs[k] for k in range(len(nbs)) if min_cost[k] <= rb ])
            
            assert len(valid_acts) > 0
            
            return list(set(valid_acts))
        
    def get_dim_info(self, robot_num, reuse_network= True, full_observable=True, state_type=0, args = None):
        assert args is not None
        args.n_actions = self.num_of_nodes + 1
        args.n_agents = robot_num
        args.n_robots = robot_num
        if reuse_network:
            if full_observable:
                if state_type==0:
                    args.nn_input_dim = 3 * robot_num + robot_num  
                else:
                    args.nn_input_dim = 3 * robot_num + self.num_of_nodes + robot_num  
            else:
                if state_type==0:
                    args.nn_input_dim = 3 + robot_num  
                else:
                    args.nn_input_dim = 3 + self.num_of_nodes + robot_num 
            #critic always full observable
            if state_type==0:
                args.critic_dim = 3 * robot_num + robot_num
            else:
                args.critic_dim = 3 * robot_num + self.num_of_nodes + robot_num
        else:
            if full_observable:
                if state_type==0:
                    args.nn_input_dim = 3 * robot_num 
                else:
                    args.nn_input_dim = 3 * robot_num +  self.num_of_nodes
            else:
                if state_type==0:
                    args.nn_input_dim = 3
                else:
                    args.nn_input_dim = 3 + self.num_of_nodes
            
            if state_type==0:
                args.critic_dim = 3 * robot_num
            else:
                args.critic_dim = 3 * robot_num + self.num_of_nodes

    
class EnvSeqRL(GraphTools):
    """Envionrment for sequential RL"""
    def __init__(self,areaname,config):
        GraphTools.__init__(self,areaname,config)
        self.sample_interval = config[areaname]['sample_interval']
        self.config = deepcopy(config)
        print('Init environment for sequential RL')

    def reset(self, paths_specific,state_type = 0):
        """Reset the environment"""  
        assert len(paths_specific['vses']) == len(paths_specific['Bs'])
        self.robot_num = len(paths_specific['vses'])
        self.vses = paths_specific['vses']
        self.Bs = paths_specific['Bs']
        self.chargingnodes = paths_specific['chargingnodes']
        self.paths_specific = deepcopy(paths_specific)
        #print('reset with {} robots, start loations:{}, budgets:{}'.format(self.robot_num,str(self.vses),str(self.Bs)))

        self.active_ID = 0
        self.cnode = self.vses[0]
        self.rb = self.Bs[0]
        
        self.node_tracker = [self.vses[0]]
        
        self.total_plen = 0
        self.reward_list = []
        
        self.sample_set = [self.g.nodes[self.cnode]['pos']]
        
        
        #track the visit count of each nodes
        self.node_visit_counts = np.zeros(self.num_of_nodes)   
        self.node_visit_counts[paths_specific['vses'][self.active_ID]]=1
        
        
        
        #init the obervation of each robot
        self.obs = []
        for robot_idx in range(len(self.vses)):
            self.obs.append(self.g.nodes[self.vses[robot_idx]]['pos']+[self.Bs[robot_idx]])
    
        self.partial_paths = [[] for i in range(self.robot_num)]
        self.partial_paths[self.active_ID].append(self.cnode)
        
        self.sample_set_splitted = [[] for i in range(self.robot_num)]
        self.sample_set_splitted[self.active_ID].append(self.g.nodes[self.cnode]['pos'])

        #the corrent conditional entropy of the node locations
        if self.pilot_pos is not None:
            self.current_conden, self.current_node_vars = self.condition_entropy(np.row_stack((self.sample_set ,self.pilot_pos)),self.node_locs,ret_Bvar=True)
            
        else:
            self.current_conden, self.current_node_vars = self.condition_entropy(np.array(self.sample_set),self.node_locs,ret_Bvar=True)

        #Initial reward using the pilot position and initial position
        #In this case init reward contains pilot reward
        self.init_reward = self.node_en - self.current_conden 
        self.reward_list.append(self.init_reward)

        return self.state_encoding(state_type), self.init_reward
    
    
    
    def step(self,act,state_type=0):
        assert act in self.g.nodes[self.cnode]['adj']
        cost = self.node_distance(self.cnode,act)
        assert self.rb >= cost
        is_done = None
        #record previous state
        pre_state = self.state_encoding(state_type)
        
        #sample and compute reward
        new_samples = []
        nx_pos =  self.g.nodes[act]['pos']
        cur_pos = self.g.nodes[self.cnode]['pos']
        new_sample_num = int(np.floor(cost*1.0/self.sample_interval))
        #Note +1
        for j in range(1,new_sample_num + 1):
            tmp_x = cur_pos[0] + (nx_pos[0] - cur_pos[0])*1.0*j/new_sample_num
            tmp_y = cur_pos[1] + (nx_pos[1] - cur_pos[1])*1.0*j/new_sample_num
            new_samples.append([tmp_x,tmp_y])
        
        self.sample_set_splitted[self.active_ID].extend(new_samples)

        self.cnode = act
        self.node_tracker.append(act)
        self.rb -= cost
        self.total_plen += cost
        
        self.node_visit_counts[act] +=1
        
        #update the observation of the active robot
        self.obs[self.active_ID] = self.g.nodes[act]['pos']+[self.rb]
        self.partial_paths[self.active_ID].append(act)
        
        
        #If one agent terminates
        if act in self.chargingnodes:
            if self.active_ID == self.robot_num -1:
                #all robot has finished
                is_done = True
            else:
                #Swith to a new robot
                is_done = False
                self.active_ID +=1
                self.cnode = self.vses[self.active_ID]
                self.rb  = self.Bs[self.active_ID]

                self.node_tracker.append(self.cnode)
                
                self.node_visit_counts[self.cnode] += 1
                
                new_samples.append(self.g.nodes[self.cnode]['pos'])
                self.partial_paths[self.active_ID].append(self.cnode)
                self.sample_set_splitted[self.active_ID].append(self.g.nodes[self.cnode]['pos'])
        else:
            is_done = False
            
        self.sample_set.extend(new_samples)
        
        new_samples = np.array(new_samples)
        
        if self.pilot_pos is not None:
            new_conden, self.current_node_vars = self.condition_entropy(np.row_stack((np.array(self.sample_set),self.pilot_pos)),self.node_locs, ret_Bvar=True)
        else:
            new_conden, self.current_node_vars = self.condition_entropy(np.array(self.sample_set),self.node_locs, ret_Bvar=True)
            
        reward = self.current_conden - new_conden 
        self.reward_list.append(reward)
        self.current_conden  = new_conden

        post_state = self.state_encoding(state_type)
        
        assert is_done is not None
        
        return  pre_state, act, reward, post_state, is_done
        

    def state_encoding(self, state_type):
        """Capture the state of the MARL system
        """
        encodings = np.concatenate(self.obs,axis=-1)
        #add active robot ID
        #active_ID_onehot= np.zeros(len(self.vses))
        #active_ID_onehot[self.active_ID] = 1
        #encodings = np.hstack((encodings,active_ID_onehot))
        
        if state_type == 0:
            return (deepcopy(encodings), None)
        elif state_type == 1:
            return (deepcopy(encodings), deepcopy(self.node_visit_counts))
        elif state_type == 2:
            return (deepcopy(encodings), deepcopy(self.current_node_vars))
        else:
            raise Exception('Unkown state type')

    def get_robot_visited_nodes(self):
        return self.node_tracker[:]
    
    def get_valid_actions(self):
        assert self.rb >= 0 
        nbs = self.g.nodes[self.cnode]['adj']
        valid_acts = []
        for vt in self.chargingnodes:
            min_cost = [self.node_distance(self.cnode, nb) + self.g_cost_dict[(nb, vt)] for nb in nbs]
            valid_acts.extend([nbs[k] for k in range(len(nbs)) if min_cost[k] <= self.rb])
        assert len(valid_acts) >0
        return list(set(valid_acts))
    
    def get_dim_info(self, robot_num, state_type, args = None):
        assert args is not None
        if state_type==0:
            critic_dim = 3 * robot_num
            nn_input_dim = 3 * robot_num 
        else:
            critic_dim = 3 * robot_num + self.num_of_nodes
            nn_input_dim =  3 * robot_num + self.num_of_nodes
            
        args.n_actions =  self.num_of_nodes
        args.n_agents = 1
        args.n_robots = robot_num
        args.nn_input_dim = nn_input_dim
        args.critic_dim = critic_dim
        

class EnvABR(GraphTools):
    """Almost the same with EnvMARL, execept a little bit in reward_team
        and get_dim_info
    """
    def __init__(self,areaname,config):
        print('Init envionrment for action branching RL')
        GraphTools.__init__(self,areaname,config)
        self.config = deepcopy(config)
        
    def reset(self,paths_specific, reward_type = 0, state_type=0):
        """Reset  environment for RL"""  
        assert len(paths_specific['vses']) == len(paths_specific['Bs'])
        self.robot_num = len(paths_specific['vses'])
        self.robots = []
        for i in range(self.robot_num):
            self.robots.append(Robot(i, self.areaname, self.config, paths_specific['vses'][i], paths_specific['Bs'][i],paths_specific['chargingnodes'],self.g))
        self.paths_specific = deepcopy(paths_specific)


        self.node_visit_counts = np.zeros(self.num_of_nodes)
        self.node_visit_counts [paths_specific['vses']]=1
        

        #sample set, initially equals to pilot samples
        self.sample_set = copy.copy(self.pilot_pos)
        self.current_conden, self.current_node_vars = self.condition_entropy(self.sample_set,self.node_locs, ret_Bvar=True)
        #In this case init reward contains pilot reward, calculated separatly
        self.pilot_reward = self.node_en - self.current_conden 
        #Calculate team reward stored in self.rt
        self.reward_team()
        self.init_reward = self.rt 
        if reward_type == 0:
            return self.state_encoding(state_type),[self.init_reward]
        else:
            return self.state_encoding(state_type), np.repeat(self.init_reward*1.0/self.robot_num,self.robot_num).tolist()

    def reward_team(self):
        """equally split the team reward"""
        newsamples_all = [rob.increased_samples for rob in self.robots]
        newsamples_all = np.array(list(itertools.chain(*newsamples_all)))
        if len(newsamples_all) == 0:
            print('Alert: No increased samples from any robots!')
            rt = 0
        else:
            self.sample_set = np.row_stack((self.sample_set,newsamples_all))
            conden, self.current_node_vars  = self.condition_entropy(self.sample_set,self.node_locs, ret_Bar=True)
            rt = self.current_conden - conden
            self.current_conden = conden
        
        self.rt = rt
        
        return [self.rt]
    
        
    def reward_member1(self):
        """Credit assignment for multiple robot reward, based on sequential act"""
        r_list = []
        last_conden = self.current_conden
        for rob in self.robots:
            if len(rob.increased_samples) == 0:
                r_list.append(0)
            else:
                self.sample_set = np.row_stack((self.sample_set ,rob.increased_samples))
                conden, self.current_node_vars  = self.condition_entropy(self.sample_set,self.node_locs, ret_Bvar=True)
                r_list.append(self.current_conden - conden)
                self.current_conden = conden
        rt = last_conden - self.current_conden    
        self.rt = rt
        
        return r_list
    
    
    
    def reward_member2(self,regularize=True):
        """Credit assignment for multiple robot reward, based on difference reward
        Args:
            regularize, whether regularize the reward such that sum equals to team
            returns: list of reward for each agent
        """
        r_list = []
        #merge the samples from agents
        newsamples_all = [rob.increased_samples for rob in self.robots]
        newsamples_all = np.array(list(itertools.chain(*newsamples_all)))  

        #get team reward
        all_samples = np.row_stack((self.sample_set ,newsamples_all))
        conden, self.current_node_vars  = self.condition_entropy(all_samples, self.node_locs, ret_Bvar=True)
        rt = self.current_conden - conden
        
        for i,rob in zip(range(len(self.robots)),self.robots):
            if len(rob.increased_samples) == 0:
                r_list.append(0)
            else:
                absent_samples = [self.robots[j].increased_samples for j in range(len(self.robots)) if j!=i]
                absent_samples = np.array(list(itertools.chain(*absent_samples))) 
                if len(absent_samples)==0:
                    absent_samples = self.sample_set
                else:
                    absent_samples = np.row_stack((self.sample_set ,absent_samples))
                absent_conden = self.condition_entropy(absent_samples,self.node_locs)
                r_list.append(absent_conden - conden)      
        
        #pamanently set the related attributes
        self.sample_set = np.row_stack((self.sample_set ,newsamples_all))
        self.current_conden = conden
        if regularize == True:
            r_list_sum = np.sum(r_list)
            ret = [rt * r / r_list_sum for r in r_list]
        else:
            ret = r_list
            
        self.rt = rt
        
        return ret

    def step(self,acts,reward_type = 0, state_type=0):
        """Take actions  for the team
        Args:
            acts: the actions for all robot
            reward_type: type of reward signal
                0: team reward
                1: each agent a reward based on seqential allocation
                2: each agent a reward based on difference reward
        """ 
        assert len(acts) == len(self.robots)
        #must at least one robot is active
        assert np.sum([rob.is_active for rob in self.robots]) != 0
        #record the current state
        pre_state = self.state_encoding(state_type)
        for i,rob in enumerate(self.robots):
            rob.execute(acts[i])
        
        is_done = True if np.sum([rob.is_active for rob in self.robots]) == 0 else False
        
        if reward_type == 0:
            reward = self.reward_team()
        elif reward_type == 1:
            reward = self.reward_member1()
        elif reward_type == 2:
            reward = self.reward_member2()
        else:
            raise Exception('Invalid reward type')  
        post_state = self.state_encoding(state_type)
        return pre_state, acts, reward, post_state, is_done


    def state_encoding(self, state_type):
        """Capture the state of the MARL system
           Note: the shape would be  N * 3
        """
        encodings = [np.array(rob.obs) for rob in self.robots]
        if state_type == 0:
            return (deepcopy(encodings), None)
        elif state_type == 1:
            return (deepcopy(encodings), deepcopy(self.node_visit_counts))
        elif state_type == 2:
            return (deepcopy(encodings), deepcopy(self.current_node_vars))
        else:
            raise Exception('Unkown state type')
            
    def get_robot_visited_nodes(self, i):
        return self.robots[i].node_tracker[:]
    
    def get_valid_actions(self, i):
        assert  self.robots[i].rb >= 0
        
        if self.robots[i].is_active == False:
            return [self.num_of_nodes]
        else:
            current_node = self.robots[i].current_node
            rb = self.robots[i].rb
            valid_acts=[]
            nbs = self.g.nodes[current_node]['adj']
            for vt in self.paths_specific['chargingnodes']:
                min_cost = [self.node_distance(current_node, nb) + self.g_cost_dict[(nb, vt)] for nb in nbs ]
                valid_acts.extend([nbs[k] for k in range(len(nbs)) if min_cost[k] <= rb ])
            
            assert len(valid_acts) > 0
            
            return list(set(valid_acts))
        
    def get_dim_info(self, robot_num, state_type=0, args = None):
        assert args is not None
        
        if state_type==0:
            nn_input_dim = 3 * robot_num
            critic_dim = 3 * robot_num
        else:
            nn_input_dim = 3 * robot_num + self.num_of_nodes
            critic_dim = 3 * robot_num + self.num_of_nodes
            
        args.n_actions = self.num_of_nodes + 1
        args.n_agents = 1
        args.n_robots = robot_num 
        args.nn_input_dim = nn_input_dim
        args.critic_dim = critic_dim
        
                
        
def random_valid_policy_marl(env, paths_specific,unseen_priority = False):
    """Generate random valid actions for the robots
    Args: 
        env: the enviornment
        paths_specific: the paths specification
        unseen_priority: give a priority to select unseen nodes.
    """
    actions = []
    for i, rob in enumerate(env.robots):
        assert rob.rb >= 0
        valid_actions = env.get_valid_actions(i)
        if unseen_priority == False:
            actions.append(np.random.choice(valid_actions))
        else:
            visited_nodes = env.get_robot_visited_nodes(i)
            new_actions = set(valid_actions).difference(set(visited_nodes))
            if len(new_actions)>0:
                actions.append(np.random.choice(list(new_actions)))
            else:
                actions.append(np.random.choice(valid_actions))
    return actions


        
def random_valid_policy_seqrl(env, paths_specific,unseen_priority = False):
    """Generate random valid actions for the robots
    Args: 
        env: the enviornment
        paths_specific: the paths specification
        unseen_priority: give a priority to select unseen nodes.
    """
    assert env.rb >=0
    valid_actions = env.get_valid_actions()
    if unseen_priority == False:
        return np.random.choice(valid_actions)
    else:
        visited_nodes = env.get_robot_visited_nodes()
        new_actions = set(valid_actions).difference(set(visited_nodes))
        if len(new_actions) > 0:
            return np.random.choice(list(new_actions))
        else:
            return np.random.choice(valid_actions)





if __name__=='__main__':
    
    #Load public settings
    config = load_config(config_path='config.yaml')
    #robot =  Robot(1,'area_one',config,0,30,[0,26])
    #paths specifics
    #paths_specific = {'vses':[0,26,0,26,0], 'Bs':[30,30,30,30,30], 'chargingnodes': [0,26]}
    #paths_specific = {'vses':[0,26], 'Bs':[10,30], 'chargingnodes': [0,26]}
    #envmarl = EnvMARL('area_one',config)
    
    #for another area
    paths_specific = {'vses':[18,45], 'Bs':[30,30], 'chargingnodes': [18,45]}
    envmarl = EnvMARL('area_three',config)
    reward_type = 0
    state_type=2

    #Test with random policy
    rewards_recorded=[]
    rewards_evaluated=[]
    
    for i in range(1):
        #Create Env
        init_state,init_reward = envmarl.reset(paths_specific,reward_type = reward_type,state_type=state_type)
        reward_list = [envmarl.pilot_reward,envmarl.init_reward]
        done = False
        while done !=True:
            plt.close('all')
            actions = random_valid_policy_marl(envmarl,paths_specific,unseen_priority = True)
            pre_state, vs, reward, post_state, done = envmarl.step(actions,reward_type = reward_type, state_type=state_type)
            print(pre_state)
            reward_list.append(np.sum(reward))
            #reward_list.append(reward_obj[1])
            paths_nodes = [rob.node_tracker for rob in envmarl.robots]
            samples_pos = [rob.local_samples for rob in envmarl.robots]
            envmarl.plot_graph(paths_specific['chargingnodes'],paths_nodes,samples_pos,arrow=True,vis_paths=False,vis_sample_edges=False,figsize=(9,9))
            plt.pause(0.3)
            if done == True:
                rewards_recorded.append(np.sum([np.sum(x) for x in reward_list]))
                rewards_evaluated.append(envmarl.paths_evaluate([rob.node_tracker for rob in envmarl.robots]))
                break

    """
    fig, ax1 = plt.subplots(1, 1, figsize=(12,6))
    plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=3.5)
    ax1.grid()
    ax1.plot(range(len(rewards_recorded)),rewards_recorded,label='Recorded')
    ax1.plot(range(len(rewards_evaluated)),rewards_evaluated,label='Evaluated')
    ax1.set_xlabel('Number Episode',fontsize=13,fontweight='bold')
    ax1.set_ylabel('Reward',fontsize=15)
    plt.legend()
    plt.show() 
    """
    
    #Load public settings
    config = load_config(config_path='config.yaml')
    #robot =  Robot(1,'area_one',config,0,30,[0,26])
    #paths specifics
    #paths_specific = {'vses':[0,26,0,26,0], 'Bs':[30,30,30,30,30], 'chargingnodes': [0,26]}
    
    #paths_specific = {'vses':[0,26], 'Bs':[30,30], 'chargingnodes': [0,26,13]}
    #envseq = EnvSeqRL('area_one',config)
    
    
    paths_specific = {'vses':[18,45], 'Bs':[30,30], 'chargingnodes': [18,45]}
    envseq = EnvSeqRL('area_three',config)    
    
    
    state_type = 2
    
    #Test with random policy
    rewards_recorded=[]
    rewards_evaluated=[]
    
    for i in range(1):
        #Create Env
        init_state,reward = envseq.reset(paths_specific, state_type = state_type)
        done = False
        reward_list=[reward]
        while done !=True:
            plt.close('all')
            action = random_valid_policy_seqrl(envseq,paths_specific,unseen_priority = True)
            pre_state, vs, reward, post_state, done = envseq.step(action, state_type = state_type)
            #print(pre_state)
            reward_list.append(reward)
            envseq.plot_graph(charging_nodes = paths_specific['chargingnodes'],paths_nodes = envseq.partial_paths,samples_pos=envseq.sample_set_splitted,arrow=True,vis_paths=False,vis_sample_edges=False)
            plt.pause(0.3)
            if done == True:
                rewards_recorded.append(np.sum(reward_list))
                rewards_evaluated.append(envseq.paths_evaluate(envseq.partial_paths))
                break
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(12,6))
    plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=3.5)
    ax1.grid()
    ax1.plot(range(len(rewards_recorded)),rewards_recorded,label='Recorded')
    ax1.plot(range(len(rewards_evaluated)),rewards_evaluated,label='Evaluated')
    ax1.set_xlabel('Number Episode',fontsize=13,fontweight='bold')
    ax1.set_ylabel('Reward',fontsize=15)
    plt.legend()
    plt.show()    
    """
    
    
