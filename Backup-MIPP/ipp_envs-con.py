#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:01:15 2020
Environment for continous action space IPP
action is clipped in the enviornment
see an open discussion
https://github.com/openai/baselines/issues/121
need to assume robot is a point

The action clipping is based on visibility graph
@author: yongyongwei
"""
import sys
sys.dont_write_bytecode = True
import numpy as np
from scipy.spatial.distance import euclidean
import copy
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
import itertools
from copy import deepcopy
from common.utils import load_config, GraphTools
from  shapely.geometry import Point, Polygon,LineString
import networkx as nx
#import gc

def get_shapegraph(g, shapenodes):
    """get the polygon graph which set the boundary
       args:
           g: the base graph 
           shapenodes: nodes of the shape polygon
       returns:
           shape graph
    """
    spn = copy.deepcopy(shapenodes)
    assert spn[0] == spn[-1]
    sg = nx.Graph()
    sg.add_nodes_from(spn[:-1])
    coords = {v:g.nodes[v]['pos'] for v in spn}
    nx.set_node_attributes(sg, coords,'pos')
    for v1, v2 in zip(spn[:-1],spn[1:]):
        sg.add_edge(v1,v2,length=euclidean(sg.nodes[v1]['pos'],sg.nodes[v2]['pos']))
    return sg

def plot_graph(g,samples_pos = None,figsize=(5,4)):
    "plot a given graph"
    tg = g
    nodes = [v for v in tg.nodes()]
    nodeposes = {v: tg.nodes[v]['pos'] for v in tg.nodes()}
    nodelabels = {v: v for v in nodes}
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    nx.draw_networkx_nodes(tg, nodeposes,nodes, node_color = 'g', node_size=150,ax = ax)
    nx.draw_networkx_labels(tg, nodeposes, labels=nodelabels, font_size=8, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=ax)
    nx.draw_networkx_edges(tg,nodeposes,edge_color='gold',style='-.')    

    if samples_pos!=None:
        color=cm.rainbow(np.linspace(0,1,len(samples_pos)))
        for ci,sample_pos in enumerate(samples_pos):
                            
            lw=np.linspace(2,8,len(samples_pos))
            ls=['-','--','-.',':'][ci%4]
            #node_sizes = np.linspace(50,70,len(samples_pos))
            
            node_shape =  list('o^>v<sdph8')[ci%10]
            obnodes = range(len(sample_pos))
            ob_pos = {k:sample_pos[k] for k in obnodes}
            path_edges = list(zip(obnodes, obnodes[1:]))
            nx.draw_networkx_nodes(tg,ob_pos,nodelist=obnodes, \
                    node_color=color[ci].reshape(1,-1),node_shape=node_shape,node_size=50,alpha=0.5,ax=ax)
            nx.draw_networkx_edges(tg,ob_pos,edgelist=path_edges,\
                edge_color=matplotlib.colors.rgb2hex(color[ci]),width=lw[ci],style=ls,alpha=0.5,ax=ax)   
        #set tight layout        
        plt.tight_layout()
        plt.show()   
        
def rototate_angle(pos, angle, steplen, po, unit = np.pi/10):
    """ rotate the angle such that one step follow the angle falls within the polygon
        return directly if the angle is valid
    args:
        pos -- current position
        angle -- moving direction, range from (0,2pi)
        steplen -- step length from pos towards angle
        pos -- boundary polygon
    returns:
        a valid angle
    """
    next_x = pos[0] + steplen * np.cos(angle)
    next_y = pos[1] + steplen * np.sin(angle)
    if po.contains (Point(next_x,next_y)):
        return np.mod(angle, np.pi * 2) 
    else:
        ret = None
        total_trial_num = int(np.ceil(np.pi * 2 / unit))
        for trial_idx in range(1,total_trial_num):
            next_x = pos[0] + steplen * np.cos(angle + trial_idx * unit)
            next_y = pos[1] + steplen * np.sin(angle + trial_idx * unit)  
            if po.contains (Point(next_x,next_y)) == True:
                ret = np.mod(angle + trial_idx * unit, np.pi * 2)
                break
        assert ret != None, "failed to search the correction direction, change unit and retry!"
        return ret
    
def prune_angle(shapegraph,pos, angle, steplen,targetpos, po, rb, unit = np.pi/10):
    """prune the angle towards the target position
       ensure the the agent will (1) not hit the wall and 
       (2) will have enough budget towards target position
    Args:
        shapegraph - the shape graph
        pos - current position
        angle - original moving diretion
        steplen - step length
        targetpos - the target position
        po - shape polygon
        rb - remaining budget
    returns: 
        angle, shortest path cost (if clipped, and without rotation)
    """
    angle = np.clip(angle, 0, 2 * np.pi)
    x = pos[0] + steplen * np.cos(angle)
    y = pos[1] + steplen * np.sin(angle)
    valid = po.contains(LineString([(pos[0],pos[1]),(x,y)]))
    reachable = False
    if valid == True:
        ng = copy.deepcopy(shapegraph)
        max_node_ID = np.max(ng.nodes())
        basenodes = ng.nodes()
        #here use next posotion
        ng.add_node(max_node_ID + 1, pos = [x,y])
        ng.add_node(max_node_ID + 2, pos = [targetpos[0],targetpos[1]])
        #create visibility eges
        for bn in basenodes:
            if po.contains(LineString([(ng.nodes[bn]['pos'][0],ng.nodes[bn]['pos'][1]),(x,y)])):
                ng.add_edge(bn,max_node_ID + 1,length = euclidean(ng.nodes[bn]['pos'],[x,y]))
            if po.contains(LineString([(ng.nodes[bn]['pos'][0],ng.nodes[bn]['pos'][1]),(targetpos[0],targetpos[1])])):
                ng.add_edge(bn,max_node_ID + 2,length = euclidean(ng.nodes[bn]['pos'],targetpos))
        if po.contains(LineString([(x,y),(targetpos[0],targetpos[1])])):
            ng.add_edge(max_node_ID + 1,max_node_ID + 2,length = euclidean([x,y],targetpos))
        spnodes = nx.shortest_path(ng,max_node_ID+1,max_node_ID+2,weight='length')
        sp_cost = np.sum([ng.edges[v1,v2]['length'] for v1, v2 in zip(spnodes[:-1],spnodes[1:])])
        if steplen + sp_cost <= rb:
            reachable = True
        del ng
    if valid == True and reachable == True:
        return np.mod(angle, np.pi * 2), None
    else:
        #prune action, either because hit the wall or budget not enough
        ng = copy.deepcopy(shapegraph)
        max_node_ID = np.max(ng.nodes())
        basenodes = ng.nodes()
        #note here use current pos
        ng.add_node(max_node_ID + 1, pos = pos)
        ng.add_node(max_node_ID + 2, pos = [targetpos[0],targetpos[1]])
        #create visibility eges
        for bn in basenodes:
            if po.contains(LineString([(ng.nodes[bn]['pos'][0],ng.nodes[bn]['pos'][1]),(pos[0],pos[1])])):
                ng.add_edge(bn,max_node_ID + 1,length = euclidean(ng.nodes[bn]['pos'],(pos[0],pos[1])))
            if po.contains(LineString([(ng.nodes[bn]['pos'][0],ng.nodes[bn]['pos'][1]),(targetpos[0],targetpos[1])])):
                ng.add_edge(bn,max_node_ID + 2,length = euclidean(ng.nodes[bn]['pos'],targetpos))
        if po.contains(LineString([(pos[0],pos[1]),(targetpos[0],targetpos[1])])):
            ng.add_edge(max_node_ID + 1,max_node_ID + 2,length = euclidean(pos,targetpos))
        spnodes = nx.shortest_path(ng,max_node_ID+1,max_node_ID+2,weight='length')
        sp_cost = np.sum([ng.edges[v1,v2]['length'] for v1, v2 in zip(spnodes[:-1],spnodes[1:])])
        dx = ng.nodes[spnodes[1]]['pos'][0] - ng.nodes[spnodes[0]]['pos'][0]
        dy = ng.nodes[spnodes[1]]['pos'][1] - ng.nodes[spnodes[0]]['pos'][1]
        #shortest path heading
        heading = np.mod(np.arctan2(dy,dx),2*np.pi)
        #this function will rotate if nessesary, otherwise return
        heading = rototate_angle(pos, heading, steplen, po, unit)            
        del ng
        return  heading, sp_cost

def sample_cont_paths(basegraph, paths, sample_interval, steplen, plot = False,figsize=(5,4)):
    """sample from continous paths
        Args:
            basegraph - the base graph, mainly for plot purpose
            paths: paths in the form of coordinates, not graph nodes
            sample_interval: interval
            steplen: length of each step
        return:
            N * 2 array of locations
    """
    samples_pos = []
    for path in paths:
        sample_pos = []
        assert len(path) >=1
        cur_pos = path[0]
        if len(path) == 1:
            sample_pos.append(cur_pos)
            samples_pos.append(sample_pos)
            continue
        for i in range(1, len(path)):
            sample_pos.append(cur_pos)
            nx_pos = path[i]
            edge_sample_num = int(steplen/sample_interval)
            for j in range(1, edge_sample_num):
                tmp_x = cur_pos[0] + (nx_pos[0] - cur_pos[0])*1.0*j/edge_sample_num
                tmp_y = cur_pos[1] + (nx_pos[1] - cur_pos[1])*1.0*j/edge_sample_num
                sample_pos.append([tmp_x,tmp_y])   
            if i==len(path) - 1: #last node
                sample_pos.append(path[i])
                break
            else:
                cur_pos = nx_pos
        samples_pos.append(sample_pos)
    if plot == True:
        plot_graph(basegraph,samples_pos=samples_pos,figsize = figsize)
    return np.array(samples_pos)
                    

class Robot():
    def __init__(self,ID,areaname,config,init_pos,B,terminal_poses,g):
        """the terminal poeses is a list of coords, could be multiple"""
        self.ID = ID
        self.init_pos = init_pos
        self.B = B
        self.terminal_poses = terminal_poses
        self.steplen = config[areaname]['steplen']
        self.sample_interval = config[areaname]['sample_interval-cont']
        self.g = g
        boundry_coords = np.array([g.nodes[v]['pos'] for v in config[areaname]['shape_nodes']])
        self.shape_polygon = Polygon(boundry_coords)
        self.reset()
        #max number of steps
        self.max_step_num = int(self.B/self.steplen) 
        
    def reset(self):
        """Reset the status of the robot
            note: trajectory include remaining budget, also  repeative entry if stopped
            while path does not include repeatitive positions
        """
        self.current_pos = copy.copy(self.init_pos)
        self.trajectory = [self.current_pos+[self.B]]   
        self.path = [self.current_pos] 
        self.local_samples = [self.current_pos]
        self.increased_samples = [self.current_pos]
        self.rb = self.B
        self.is_active = True

    def execute(self,a):
        """Execute an action for the robot
        the action is already clipped by the envionment
        a: the direction (angle) of moving, [0,2pi], if a ==-1: robot stopped, NOOP
        """
        self.increased_samples = []
        if a == -1:
            assert self.is_active == False
            self.trajectory.append(self.current_pos+[self.rb])
            return
        else:
            assert self.is_active == True,"robot %d already stopped!" % self.ID
            self.rb -= self.steplen
            nx_x = self.current_pos[0] + self.steplen * np.cos(a)
            nx_y = self.current_pos[1] + self.steplen * np.sin(a)
            nx_pos =  [nx_x,nx_y]
            cur_pos = self.current_pos[:]
            new_sample_num = int(np.floor(self.steplen*1.0/self.sample_interval))
            assert  self.shape_polygon.contains(Point(nx_x,nx_y)), "Invalid next position"

            for j in range(1,new_sample_num + 1):
                tmp_x = cur_pos[0] + (nx_pos[0] - cur_pos[0])*1.0*j/new_sample_num
                tmp_y = cur_pos[1] + (nx_pos[1] - cur_pos[1])*1.0*j/new_sample_num
                self.local_samples.append([tmp_x,tmp_y])
                self.increased_samples.append([tmp_x,tmp_y])
            #update tracking
            self.current_pos = [nx_x, nx_y]
            self.trajectory.append(self.current_pos + [self.rb])
            self.path.append(self.current_pos)
            #check whether terminate the robot (note prevent early top require steps num)
            for terminal_pos in self.terminal_poses:
                if euclidean(self.current_pos, terminal_pos) <= self.steplen and len(self.trajectory)>=self.max_step_num-3:
                    self.is_active = False
                    break                
            if self.is_active == True:
                assert  self.rb >0,  "Invalid state, error!, please check!"
            return
        
    
class EnvMARL(GraphTools):
    def __init__(self,areaname,config):
        print('Init envionrment for multi-agent coorperative RL')
        GraphTools.__init__(self,areaname,config)
        self.config = deepcopy(config)
        self.shapegraph = get_shapegraph(self.g, config[areaname]['shape_nodes'])
        boundry_coords = np.array([self.g.nodes[v]['pos'] for v in config[areaname]['shape_nodes']])
        self.shape_polygon = Polygon(boundry_coords)
        self.steplen = config[areaname]['steplen']
        self.sample_interval = config[areaname]['sample_interval-cont']
        
    def reset(self,paths_specific, reward_type = 0):
        """Reset  environment for RL"""  
        assert len(paths_specific['init_poses']) == len(paths_specific['Bs'])
        self.robot_num = len(paths_specific['Bs'])
        self.terminal_poses = paths_specific['terminal_poses']
        self.robots = []
        for i in range(self.robot_num):
            self.robots.append(Robot(i, self.areaname, self.config, paths_specific['init_poses'][i], paths_specific['Bs'][i],self.terminal_poses,self.g))
        self.paths_specific = deepcopy(paths_specific)
        print('reset with {} robots, start loations:{}, budgets:{}'.format(self.robot_num,str(paths_specific['init_poses']),str(paths_specific['Bs'])))
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
            return self.get_state(),[self.init_reward]
        else:
            return self.get_state(), np.repeat(self.init_reward*1.0/self.robot_num,self.robot_num).tolist()
    
    def prune_action(self, angles):
        """prune actions for each robot
            will call prune_angle(shapegraph,pos, angle, steplen,targetpos, po, rb, unit = np.pi/10)
        """
        pruned_angles = []
        #for each robot
        for i,angle in enumerate(angles):
            if len(self.terminal_poses) == 0:
                pruned_angle, sp_dist = prune_angle(self.shapegraph,self.robots[i].current_pos, angle, self.steplen, self.terminal_poses[0], self.shape_polygon, self.robots[i].rb)
                pruned_angles.append(pruned_angle)
            else:
                #for the case of multiple candidate terminal positions
                #need to make a choice
                base_dist = 100000
                selection = None
                for terminal_pos in self.terminal_poses:
                    pruned_angle, sp_dist = prune_angle(self.shapegraph,self.robots[i].current_pos, angle, self.steplen, terminal_pos, self.shape_polygon, self.robots[i].rb)
                    #prioritized to choose original angle (without  shortest path, sp_dist is None)
                    if sp_dist is None:
                        selection = pruned_angle
                        break
                    #if no such option, then select the shortest (among all terminal) direction
                    else:
                        if sp_dist < base_dist:
                            base_dist = sp_dist
                            selection = pruned_angle
                assert selection is not None, "Invalid selection!"
                pruned_angles.append(selection)
        return pruned_angles
                    
                        
            
    def step(self,acts,reward_type = 0):
        """Take actions  for the team
        Args:
            acts: the actions for all robot
            reward_type: type of reward signal
                0: team reward split among active-robot(must be active, not dead)
                1: each agent a reward based on seqential allocation
                2: each agent a reward based on difference reward
        """ 
        assert len(acts) == len(self.robots)
        assert np.sum([rob.is_active for rob in self.robots]) != 0
        pruned_actions = self.prune_action(acts)
        #print('input actions:',acts,' pruned actions:',pruned_actions)
        robots_status = [rob.is_active for rob in self.robots]
        #clip for dead robots
        for i,status in enumerate(robots_status):
            if status == False:
                pruned_actions[i] = -1

        pre_state = self.get_state()
            
        for i,rob in enumerate(self.robots):
            rob.execute(pruned_actions[i])
        
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
        post_state = self.get_state()
        return pre_state, acts, reward, post_state, is_done
        
    
    def plot_trajectories(self,figsize=(5,4)):
        samples_pos = [rob.local_samples for rob in self.robots]
        plot_graph(self.g,samples_pos = samples_pos,figsize=figsize)
        
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

    def contpaths_evaluate(self, paths, plot = False,figsize=(5,4)):
        sample_pos=sample_cont_paths(self.g, paths, self.sample_interval, self.steplen, plot = plot,figsize=figsize)
        #change shape, merge samples from multiple paths
        sample_pos = list(itertools.chain(*sample_pos))
        #Note also consider the pilot effect
        if self.pilot_pos is not None:
            sample_pos = np.row_stack((np.array(sample_pos),self.pilot_pos))
        reward = self.samples_evaluate(sample_pos)
        return reward

    def get_state(self):
        """Capture the state of the MARL system
        dict:{'current_poses'':[],'rbs':[],'nodevars':}
        """
        positions = np.array([rob.current_pos for rob in self.robots])
        rbs = np.array([rob.rb for rob in self.robots])
        nodevars = self.current_node_vars
        return {'poses':positions,'rbs':rbs,'nodevars':nodevars}
    


class EnvSeqRL(GraphTools):
    """Envionrment for sequential RL
        note this is environment the only difference of trajectory and path is 
        that trajectory will track the budget information, unlike MARL env, since 
        thre is no dummy action.
    """
    def __init__(self,areaname,config):
        GraphTools.__init__(self,areaname,config)
        self.config = deepcopy(config)
        self.shapegraph = get_shapegraph(self.g, config[areaname]['shape_nodes'])
        boundry_coords = np.array([self.g.nodes[v]['pos'] for v in config[areaname]['shape_nodes']])
        self.shape_polygon = Polygon(boundry_coords)
        self.sample_interval = config[areaname]['sample_interval-cont']
        self.steplen = config[areaname]['steplen']
        print('Init environment for sequential RL')

    def reset(self, paths_specific):
        """Reset the environment"""  
        assert len(paths_specific['init_poses']) == len(paths_specific['Bs'])
        self.robot_num = len(paths_specific['Bs'])
        self.init_poses = paths_specific['init_poses']
        self.Bs = paths_specific['Bs']
        self.terminal_poses = paths_specific['terminal_poses']
        self.paths_specific = deepcopy(paths_specific)
        #print('reset with {} robots, start loations:{}, budgets:{}'.format(self.robot_num,str(self.vses),str(self.Bs)))

        self.max_step_nums = []
        for rob_ID in range(self.robot_num):
            self.max_step_nums.append(int(self.Bs[rob_ID]/self.steplen))
        #current_pos: the positio of active robot, positions: all positions of all robots
        self.active_ID = 0
        self.current_pos = copy.copy(self.init_poses[0])
        self.rb = self.Bs[0]
                
        #track the status of each robot
        self.rbs = copy.copy(self.Bs)
        self.positions = copy.deepcopy(self.init_poses)

        #the following variables are updated follow active robots
        self.paths = [[] for i in range(self.robot_num)]
        self.paths[self.active_ID].append(self.current_pos)
        
        self.sample_set = [[] for i in range(self.robot_num)]
        self.sample_set[self.active_ID].append(self.current_pos)
    
        self.trajectories = [[] for i in range(self.robot_num)]
        self.trajectories[self.active_ID].append(self.init_poses[self.active_ID]+[self.Bs[self.active_ID]])
    
        #the corrent conditional entropy of the node locations
        if self.pilot_pos is not None:
            all_samples = np.row_stack((np.array(list(itertools.chain(*self.sample_set))),self.pilot_pos)) 
            self.current_conden, self.current_node_vars = self.condition_entropy(all_samples,self.node_locs,ret_Bvar=True)
        else:
            all_samples = np.array(list(itertools.chain(*self.sample_set)))
            self.current_conden, self.current_node_vars = self.condition_entropy(all_samples,self.node_locs,ret_Bvar=True)

        #Initial reward using the pilot position and initial position
        #In this case init reward contains pilot reward
        self.init_reward = self.node_en - self.current_conden 
        #self.reward_list.append(self.init_reward)
        return self.get_state(), self.init_reward


    def prune_action(self, angle):
        """prune the action for the active robot
            will call prune_angle(shapegraph,pos, angle, steplen,targetpos, po, rb, unit = np.pi/10)
        """
        if len(self.terminal_poses) == 0:
            pruned_angle, sp_dist = prune_angle(self.shapegraph,self.current_pos, angle, self.steplen, self.terminal_poses[0], self.shape_polygon, self.rb)
        else:
            base_dist = 100000
            selection = None
            for terminal_pos in self.terminal_poses:
                pruned_angle, sp_dist = prune_angle(self.shapegraph,self.current_pos, angle, self.steplen, terminal_pos, self.shape_polygon, self.rb)
                #prioritized to choose original angle (without  shortest path, sp_dist is None)
                if sp_dist is None:
                    selection = pruned_angle
                    break
                #if no such option, then select the shortest (among all terminal) direction
                else:
                    if sp_dist < base_dist:
                        base_dist = sp_dist
                        selection = pruned_angle
            assert selection is not None, "Invalid selection!"
            pruned_angle = selection
        return pruned_angle
                    
    
    def step(self,act):
        act = self.prune_action(act)
        is_done = None
        #record previous state
        pre_state = self.get_state()
        #sample and compute reward
        new_samples = []
        nx_x = self.current_pos[0] + self.steplen * np.cos(act)
        nx_y = self.current_pos[1] + self.steplen * np.sin(act)
        nx_pos =  [nx_x, nx_y]
        cur_pos = self.current_pos
        new_sample_num = int(np.floor(self.steplen*1.0/self.sample_interval))
        #Note +1
        for j in range(1,new_sample_num + 1):
            tmp_x = cur_pos[0] + (nx_pos[0] - cur_pos[0])*1.0*j/new_sample_num
            tmp_y = cur_pos[1] + (nx_pos[1] - cur_pos[1])*1.0*j/new_sample_num
            new_samples.append([tmp_x,tmp_y])
            
        self.sample_set[self.active_ID].extend(new_samples)

        self.current_pos = copy.copy(nx_pos)
        self.positions[self.active_ID] = copy.copy(nx_pos)
        
        self.rb -= self.steplen
        self.rbs[self.active_ID] -= self.steplen

        #update the observation of the active robot
        self.trajectories[self.active_ID].append(self.current_pos+[self.rb])
        self.paths[self.active_ID].append(self.current_pos)
        
        
        #check if is done
        rob_stopped = False
        for terminal_pos in self.terminal_poses:
            if euclidean(self.current_pos, terminal_pos) <= self.steplen \
                and len(self.trajectories[self.active_ID])>=self.max_step_nums[self.active_ID] - 3:
                    rob_stopped = True
                    break
        if rob_stopped == True:
            if self.active_ID == self.robot_num -1:
                is_done = True
            else:
                is_done = False
                self.active_ID +=1
                self.rb = self.Bs[self.active_ID]
                self.current_pos = self.init_poses[self.active_ID]
                
                new_samples.append(self.current_pos)
                self.paths[self.active_ID].append(self.current_pos)
                self.trajectories[self.active_ID].append(self.init_poses[self.active_ID]+[self.Bs[self.active_ID]])
                self.sample_set[self.active_ID].append(self.current_pos)
        else:
            is_done = False
                
        
        if self.pilot_pos is not None:
            all_samples = np.row_stack((np.array(list(itertools.chain(*self.sample_set))),self.pilot_pos)) 
            new_conden, self.current_node_vars = self.condition_entropy(all_samples,self.node_locs, ret_Bvar=True)
        else:
            all_samples = np.array(list(itertools.chain(*self.sample_set)))
            new_conden, self.current_node_vars = self.condition_entropy(all_samples,self.node_locs, ret_Bvar=True)
        reward = self.current_conden - new_conden 
    
        self.current_conden  = new_conden

        post_state = self.get_state()
                
        return  pre_state, act, reward, post_state, is_done
        

    def get_state(self):
        """Capture the state of the MARL system
        """
        return {'poses':self.positions,'rbs':self.rbs,'nodevars':self.current_node_vars}
    
    def contpaths_evaluate(self, paths, plot = False,figsize=(5,4)):
        sample_pos=sample_cont_paths(self.g, paths, self.sample_interval, self.steplen, plot = plot,figsize=figsize)
        #change shape, merge samples from multiple paths
        sample_pos = list(itertools.chain(*sample_pos))
        #Note also consider the pilot effect
        if self.pilot_pos is not None:
            sample_pos = np.row_stack((np.array(sample_pos),self.pilot_pos))
        reward = self.samples_evaluate(sample_pos)
        return reward

    def plot_trajectories(self,figsize=(5,4)):
        plot_graph(self.g,samples_pos = self.sample_set,figsize=figsize)

if __name__=='__main__':
    
    #Load public settings
    config = load_config(config_path='config-con.yaml')
    config['area_one']['sample_interval-cont'] = 0.5
    envmarl = EnvMARL('area_one',config)
    paths_specific = {'init_poses':[[10.,8.],[10.,8.]], 'Bs':[40,40], 'terminal_poses':[envmarl.g.nodes[21]['pos']]}
    figsize=(5,4)
    
    config = load_config(config_path='config-con.yaml')
    config['area_two']['sample_interval-cont'] = 1
    envmarl = EnvMARL('area_two',config)
    paths_specific = {'init_poses':[[19,4],[19,4]], 'Bs':[140,140], 'terminal_poses':[envmarl.g.nodes[44]['pos'],envmarl.g.nodes[52]['pos']]}
    figsize=(10,4)
    
    
    reward_type = 0
    #Test with random policy
    rewards_recorded=[]
    rewards_evaluated=[]

    for i in range(40000):
        #Create Env
        init_state, init_reward = envmarl.reset(paths_specific,reward_type = reward_type)
        reward_list = [envmarl.pilot_reward,envmarl.init_reward]
        done = False
        while done !=True:
            #plt.close('all')
            actions = (np.pi * 2 * np.random.random(len(paths_specific['Bs']))).tolist()
            pre_state, acts,reward,post_state,done = envmarl.step(actions,reward_type = reward_type)
            reward_list.append(np.sum(reward))
            #envmarl.plot_trajectories()
            #plt.pause(0.6)
            if done == True:
                #plt.close('all')
                #envmarl.plot_trajectories(figsize=figsize)
                #plt.pause(0.3)
                print('remaining budge:',[rob.rb for rob in envmarl.robots])
                
                rewards_recorded.append(np.sum([np.sum(x) for x in reward_list]))
                paths = [rob.path for rob in envmarl.robots]
                eval_reward = envmarl.contpaths_evaluate(paths,plot=False,figsize=figsize)
                rewards_evaluated.append(eval_reward)
            
                #print(np.sum([np.sum(x) for x in reward_list]), eval_reward)
                assert abs((np.sum([np.sum(x) for x in reward_list])- eval_reward)) <0.000001
                break
    
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(5,4))
    plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=3.5)
    ax1.grid()
    ax1.plot(range(len(rewards_recorded)),rewards_recorded,label='Recorded')
    ax1.plot(range(len(rewards_evaluated)),rewards_evaluated,label='Evaluated')
    ax1.set_xlabel('Number Episode',fontsize=13,fontweight='bold')
    ax1.set_ylabel('Reward',fontsize=15)
    plt.legend()
    plt.show() 
    """
    
    
    #========================For sequential rollout envrionment================
    #Load public settings
    config = load_config(config_path='config-con.yaml')
    config['area_one']['sample_interval-cont'] = 0.5
    envseq = EnvSeqRL('area_one',config)
    paths_specific = {'init_poses':[[10.,8.],[10.,8.]], 'Bs':[40,40], 'terminal_poses':[envseq.g.nodes[21]['pos']]}
    figsize=(5,4)
    
    config = load_config(config_path='config-con.yaml')
    config['area_two']['sample_interval-cont'] = 1
    envseq = EnvSeqRL('area_two',config)
    paths_specific = {'init_poses':[[19,4],[19,4]], 'Bs':[140,140], 'terminal_poses':[envseq.g.nodes[44]['pos'],envseq.g.nodes[52]['pos']]}
    figsize=(10,4)
    
    #Test with random policy
    rewards_recorded=[]
    rewards_evaluated=[]

    for i in range(40000):
        #Create Env
        init_state, reward = envseq.reset(paths_specific)
        reward_list = [reward]
        done = False
        while done !=True:
            #plt.close('all')
            action = np.pi * 2 * np.random.random()
            pre_state, act,reward,post_state,done = envseq.step(action)
            reward_list.append(reward)
            #envseq.plot_trajectories()
            #plt.pause(0.6)
            if done == True:
                #plt.close('all')
                #envseq.plot_trajectories(figsize=figsize)
                #plt.pause(0.3)
                print('remaining budge:',[envseq.trajectories[i][-1][-1] for i in range(len(envseq.trajectories))])
                
                rewards_recorded.append(np.sum(reward_list))
        
                eval_reward = envseq.contpaths_evaluate(envseq.paths,plot=False,figsize=figsize)
                rewards_evaluated.append(eval_reward)
            
                #print(np.sum(reward_list), eval_reward)
                assert abs((np.sum(reward_list)-eval_reward)) <0.000001
                
                break
    
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(5,4))
    plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=3.5)
    ax1.grid()
    ax1.plot(range(len(rewards_recorded)),rewards_recorded,label='Recorded')
    ax1.plot(range(len(rewards_evaluated)),rewards_evaluated,label='Evaluated')
    ax1.set_xlabel('Number Episode',fontsize=13,fontweight='bold')
    ax1.set_ylabel('Reward',fontsize=15)
    plt.legend()
    plt.show() 
    """