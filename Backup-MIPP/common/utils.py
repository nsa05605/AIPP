#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:09:51 2020
Encasulate the graph related operations!
@author: yongyongwei
"""

import sys
sys.dont_write_bytecode = True


import pickle
import os
import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean
from GPy.kern import Exponential
from numpy.linalg import inv
from matplotlib import pyplot as plt
import matplotlib
import yaml
import itertools
from matplotlib.pyplot import cm
import torch

#Note the tsp solver part only works for python2
#sys.path.append('/Users/yongyongwei/Seafile/Research/Spyder/MIPP/Lib')
#from pycon.tsp import TSPSolver

#Also note the difference of networkx versions
#https://networkx.github.io/documentation/stable/release/migration_guide_from_1.x_to_2.0.html
#g.node - > g.nodes
#set_node_attributes the order of args has been changed



def load_config(config_path='config.yaml'):
    with open(config_path) as f:
        config = yaml.full_load(f)
    return config
    
def is_onpolicy(alg):
    if alg.find('reinforce') > -1 or alg.find('coma') > -1 or alg.find('acca') > -1 \
        or alg.find('central_v') > -1 or alg.find('a2c') > -1:
        return True
    else:
        return False
    
def is_coopmarl(alg):
    if alg.find('coma') > -1 or alg.find('mix') > -1 or alg.find('qtran') >-1 or alg.find('acca') >-1 \
        or alg.find('maven') > -1 or alg.find('liir') > -1 or alg.find('can') >-1 or alg.find('ligm')>-1:
            return True
    else:
        return False
        
class RecursiveNamespace:
    
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)
        elif isinstance(entry,list):
            return list(map(RecursiveNamespace.map_entry,entry))
        else:
            return entry
        
    def __init__(self,**kwargs):
        for key,val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(RecursiveNamespace.map_entry,val)))
            else:
                setattr(self, key, val)
                

def recursivenamespace_2dict(rnobj):
    ret={}
    for k,v in rnobj.__dict__.items():
        if isinstance(v,RecursiveNamespace):
            ret[k] = recursivenamespace_2dict(v)
        else:
            ret[k] = v
    return ret

class GraphTools:
    def __init__(self, areaname, config):
        self.areaname = areaname
        self.sample_interval = config[areaname]['sample_interval']
        self.gp_hypers = config[areaname]['gp_hypers']
        #Read the graph information from the file
        with open(config[areaname]['prob_file'], 'r') as f_ipp:
            coors = {}
            adj_list = {}
            in_sec = False
            adj_sec = False
            n_nodes = -1
            for l in f_ipp:
                if 'DIMENSION' in l:
                    n_nodes = int(l.strip().split(' ')[-1].strip())
                if in_sec:
                    idx, x, y = [w.strip() for w in l.strip().split(' ') if len(w.strip())]
                    idx = int(idx)
                    coors[idx] = [float(x), float(y)]
                    assert len(coors) == idx + 1
                    if len(coors) == n_nodes:
                        in_sec = False
                if adj_sec:
                    idx, vlist = [w.strip() for w in l.strip().split(':')]
                    idx = int(idx)
                    vlist = [int(v) for v in vlist.split() ]
                    adj_list[idx] = vlist
                    if len(adj_list) == n_nodes:
                        adj_sec = False
                elif 'NODE_COORD_SECTION' in l:
                    in_sec = True
                elif 'ADJACENT_NODES_SECTION' in l:
                    adj_sec = True
        assert len(coors) == n_nodes
        assert len(adj_list) == n_nodes
        #Create the graph
        g = nx.Graph()
        g.add_nodes_from(range(n_nodes))
        
        #nx.set_node_attributes(g, 'pos', coors)
        #nx.set_node_attributes(g,'adj',adj_list)
        #for networkx >2. 0 need to exchange arg order
        nx.set_node_attributes(g, coors,'pos')
        nx.set_node_attributes(g, adj_list,'adj')
        for v in g.nodes():
            for nb in g.nodes[v]['adj']:
                g.add_edge(v,nb,length=euclidean(g.nodes[v]['pos'],g.nodes[nb]['pos']))
        
        self.g = g
        self.node_locs = np.array([self.g.nodes[v]['pos'] for v in self.g.nodes()])
        self.node_en = self.entropy(self.node_locs)
        self.num_of_nodes = len(self.g.nodes())
        
        #set the shortest path dict for the true graph, for IppEnv1, this is useless 
        self.g_traj_dict={}
        self.g_cost_dict={}
        for vi in self.g.nodes():
            for vj in self.g.nodes():
                if (vj,vi) in self.g_traj_dict:
                    assert( (vj,vi) in self.g_cost_dict )
                    self.g_cost_dict[(vi,vj)] = self.g_cost_dict[(vj,vi)]
                    self.g_traj_dict[(vi,vj)] = self.g_traj_dict[(vj,vi)][::-1]
                else:    
                    self.g_cost_dict[(vi,vj)] = nx.shortest_path_length(self.g, vi, vj,weight='length')
                    self.g_traj_dict[(vi,vj)] = nx.shortest_path(self.g,vi, vj,weight='length')
                    
        
        #read the pilot locations
        with open(config[areaname]['pilot_file'],'rb') as fin:
            self.pilot_pos, _ = pickle.load(fin,encoding='iso-8859-1')
            
    def plot_graph(self,charging_nodes=None,paths_nodes=None,samples_pos=None,arrow=False,vis_paths = True,vis_sample_edges=True,figsize=(9,11)):
        """Plot the graph with different conditions(based on  graph)
        
        Args:
            - path_nodes: the node sequence which forms a path
            - sample_pos: A list of locations where there are samples(observations) such as [(x0,y0),...]
        Returns:
            - None
        """
        
        #1 plot the basic treasure graph
        tg = self.g
        nodes = [v for v in tg.nodes()]
        nodeposes = {v: tg.nodes[v]['pos'] for v in tg.nodes()}
        nodelabels = {v: v for v in nodes}
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        nx.draw_networkx_nodes(tg, nodeposes,nodes, node_color = 'g', node_size=150,ax = ax)
        nx.draw_networkx_labels(tg, nodeposes, labels=nodelabels, font_size=8, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=ax)
        nx.draw_networkx_edges(tg,nodeposes,edge_color='gold',style='-.')
        
        if self.pilot_pos is not None:
            ax.scatter(np.array(self.pilot_pos)[:,0],np.array(self.pilot_pos)[:,1],marker='+')
            
        if charging_nodes !=None:
            charging_pos = np.array([self.g.nodes[charging_node]['pos'] for charging_node in charging_nodes])
            ax.scatter(charging_pos[:,0],charging_pos[:,1],marker='P', s = 400)
        
                
        #2 show path(or tour) if needed
        if paths_nodes !=None:
            color=cm.rainbow(np.linspace(0,1,len(paths_nodes)))    
            for ci,path_nodes in enumerate(paths_nodes):
                
                lw=np.linspace(2,8,len(paths_nodes))
                ls=['-','--','-.',':'][ci%4]

                    
                tmp_path_nodes = path_nodes[:]
                    
                if vis_paths == True:
                    path_poses = {n:tg.nodes[n]['pos'] for n in tmp_path_nodes}
                    path_edges = list(zip(tmp_path_nodes, tmp_path_nodes[1:]))
                    nx.draw_networkx_nodes(tg,path_poses,nodelist=tmp_path_nodes, \
                            node_color=color[ci].reshape(1,-1),node_shape='D',node_size=50,ax=ax)
                        
                    nx.draw_networkx_edges(tg,path_poses,edgelist=path_edges,\
                            edge_color=matplotlib.colors.rgb2hex(color[ci]),width=lw[ci],style=ls,alpha=0.5,ax=ax)
                    
                
                if arrow == True:       
                    for v1,v2 in zip(tmp_path_nodes[:-1],tmp_path_nodes[1:]):
                        p1 = tg.nodes[v1]['pos'];p2=tg.nodes[v2]['pos']
                        offset = 0.2
                        p1 = [p1[0]-offset,p1[1]-offset]
                        p2=[p2[0]-offset,p2[1]-offset]
                        ax.arrow(p1[0], p1[1],p2[0]-p1[0], p2[1] - p1[1],head_width=0.2, head_length=0.3, fc='grey', ec='grey',ls='-')
                  
                    
        #3 add sampled positions if needed
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

                if vis_sample_edges==True:
                    nx.draw_networkx_edges(tg,ob_pos,edgelist=path_edges,\
                        edge_color=matplotlib.colors.rgb2hex(color[ci]),width=lw[ci],style=ls,alpha=0.5,ax=ax)   

        
        #set tight layout        
        plt.tight_layout()
        if 'agg' in matplotlib.get_backend():
            plotfile='%s_graph.png' % os.path.basename(self.problem_file).split('.')[0]
            fig.savefig(plotfile)
        else:
            plt.show()          

    def node_distance(self,v1, v2):
        """Get the euclidean distance between two nodes
        
        Args:
            -- v1 : The vertex from
            -- v2 : The vertex to
        Returns:
            The distance between v1 to v2
        """
        dist=euclidean(self.g.nodes[v1]['pos'],self.g.nodes[v2]['pos'])
        return dist
        
    def path_length(self,path):
        """Compute the given path length from the graph
        
        Args:
            path: The node sequence forming a path
        Returns: 
            The length of the path
        """
        length = 0
        for i in range(len(path)-1):
            length += self.node_distance(path[i], path[i+1])
        return length        

    def sample_paths(self, paths, plotpaths=False):
        """Take sample along the paths (multi-paths)
        Note this sample function does not check whther the path is legal (reachable)
        Samples(Observations) are taken along the graph
        Args:
            --paths: a list, the paths to follow, consider a tour (go back to start)
            
        Returns:
            List of coordinates sampled
        """
        samples_pos = []
        for path in paths:
            sample_pos = []
            assert len(path) >=1
            cur_vertex = path[0]
            cur_sample = self.g.nodes[cur_vertex]['pos']
            if len(path) == 1:
                sample_pos.append(cur_sample)
                samples_pos.append(sample_pos)
                continue
            for i in range(1,len(path)):
                next_vertex = path[i]
                sample_pos.append(cur_sample)
                cur_pos = self.g.nodes[cur_vertex]['pos']
                nx_pos = self.g.nodes[next_vertex]['pos']
                #note here just use euclidean dist, the input path need to garantee that the path is legal
                edge_len = euclidean(cur_pos,nx_pos)
                edge_sample_num = int(np.floor(edge_len*1.0/self.sample_interval))
                for j in range(1,edge_sample_num):
                    tmp_x = cur_pos[0] + (nx_pos[0] - cur_pos[0])*1.0*j/edge_sample_num
                    tmp_y = cur_pos[1] + (nx_pos[1] - cur_pos[1])*1.0*j/edge_sample_num
                    sample_pos.append([tmp_x,tmp_y])
                if i == len(path) - 1: #last vertex position
                    sample_pos.append(self.g.nodes[next_vertex]['pos'])
                    break
                else:
                    cur_vertex = next_vertex
                    cur_sample = self.g.nodes[cur_vertex]['pos']
            samples_pos.append(sample_pos)
    
        if plotpaths != False:      
            self.plot_graph(samples_pos=samples_pos)
            
        return samples_pos

    def condition_entropy(self,A, B,ret_Bvar=False):
        """Compute the conditional Entropy of given A of B
        
        Args:
            A: observed location
            B: unobserved locaton
            ret_Bvar: whether return the variances at B as well
        Returns:
            Conditional Entropy
        """
        kernel = Exponential(2, variance = self.gp_hypers[0], lengthscale = self.gp_hypers[1])
        sigmaBB = kernel.K(B,B)
        sigmaBA = kernel.K(B,A)
        sigmaAA_noise_inverse = inv((kernel.K(A,A) + self.gp_hypers[2] * np.identity(len(A))))
        cov_matrix = sigmaBB + self.gp_hypers[2] * np.identity(len(B))\
                     - np.dot(sigmaBA,np.dot(sigmaAA_noise_inverse,sigmaBA.T))
        #entropy=0.5 * np.log(((2*np.pi*np.e)**len(B)) * np.linalg.det(cov_matrix))
        sign, logdet = np.linalg.slogdet(cov_matrix)
        assert(sign == 1)
        en = 0.5 * (len(B) + len(B) * np.log(2*np.pi)) + 0.5 * sign *logdet
        if ret_Bvar == True:
            return en, np.diagonal(cov_matrix)
        else:
            return en

    def entropy(self,B,ret_Bvar=False):
        """Calculate the entropy from a series of locations
        Args:
            B - the locations, N*2 matrix
            ret_Bvar: whether return the variances at B
        Returns: 
            The entropy
        """
        kernel = Exponential(2, variance = self.gp_hypers[0], lengthscale = self.gp_hypers[1])
        sigmaBB = kernel.K(B,B) +  self.gp_hypers[2]  * np.identity(len(B))
        sign, logdet = np.linalg.slogdet(sigmaBB)
        assert(sign == 1)
        en = 0.5 * (len(B) + len(B) * np.log(2*np.pi)) + 0.5 * sign *logdet
        if ret_Bvar == True:
            return en, np.diagonal(sigmaBB)
        else:
            return en

    def paths_evaluate(self,paths,plotpath=False):
        """Evaluate the reward of multiple paths
        
        Args:
            path: The path to evaluate
            plotpath: Whether visualize the path
        Returns:
            The Mutual Information as the reward
        """
        sample_pos = self.sample_paths(paths,plotpath)
        #change shape, merge samples from multiple paths
        sample_pos = list(itertools.chain(*sample_pos))
        #Note also consider the pilot effect
        if self.pilot_pos is not None:
            sample_pos = np.row_stack((np.array(sample_pos),self.pilot_pos))
            
        reward = self.samples_evaluate(sample_pos)
        return reward
        
    def samples_evaluate(self,sample_pos):
        """Evaluate the samples already taken
        Note: sample_pos should contain the pilot position.
        Note: this is a little bit different compared with previous version.
              pervious version use conden0 with init_pos
        Args:
            sample_pos: List of sample locations
            plotsamples: Wheter or not plot the samples
        Returns:
            the reward of the samples
        """
        sample_pos = np.array(sample_pos)
        conden = self.condition_entropy(sample_pos, self.node_locs)
        return self.node_en  - conden
    
    def stainer_tsp(self,vs,vt,_req_nodes,magnify=10,plot=False):
        """Strainer TSP solver
        (At least 5 nodes are needed for it to work correctly)
        Args:
            vs: start node
            vt: terminal node
            _req_nodes: the required nodes that must be visited
            
        Returns:
            a path from vs to vt and pass req_nodes
        """
        raise Exception("Stainer TSP not suppurted in this version!")
        """
        #Set the list of nodes to visit
        req_nodes = _req_nodes[:]
        if vs not in req_nodes:
            req_nodes.insert(0,vs)
        if vt not in req_nodes:
            req_nodes.append(vt)
        #If for non-tour case, add a dummy node between vs and vt with small len
        if vs != vt:
            dummy_node_ID = self.num_of_nodes
            req_nodes.append(dummy_node_ID)
        #Construct the distance matrix
        dim = len(req_nodes)
        fullmat = []
        
        if vs != vt:
            for i in range(dim):
                for j in range(dim):
                    #dummy node case
                    if i == dim -1 or j == dim -1:
                        if j == dim -1:
                            if i == dim -2 or i == 0:
                                cost = 0.1
                            elif i==j:
                                cost = 0
                            else:
                                cost = 1000000
                        else:#i==dim -1
                            if j == 0 or j == dim -2:
                                cost = 0.1
                            else:
                                cost = 1000000
                    #general i==j case
                    elif i==j:
                        cost = 0
                    #other i!=j case
                    else:
                        cost = self.g_cost_dict[(req_nodes[i],req_nodes[j])]
                    #append the cost
                    fullmat.append(cost)
        else:
            for i in range(dim):
                for j in range(dim):
                    if i==j:
                        cost = 0
                    else:
                        cost = self.g_cost_dict[(req_nodes[i],req_nodes[j])]
                    fullmat.append(cost)
            
        fullmat = [int(e*magnify) for e in fullmat]
        #construct the tsp solver
        solver=TSPSolver.from_fullmat(dim,fullmat[:],"EXPLICIT")
        #print(fullmat)
        tour_data = solver.solve()
                    
        if tour_data.success == False or tour_data.found_tour == False:
            print('failed to find the path, reqnodes:%s' % str(req_nodes))
            return None        
    
        retpath = [req_nodes[s] for s in tour_data.tour]
        
        #remove the dummy node if possible
        if vs!=vt:
            retpath.remove(dummy_node_ID)
            
        #reorder from start
        s_index = retpath.index(vs)
        retpath = retpath[s_index:]+retpath[:s_index]
        
        
        #Adjust again since it might be [0, 26, 20, 14, 8, 7, 4, 3, 6, 5, 2, 1]
        if vs != vt and retpath[1] == vt:
            retpath = [retpath[0]]+retpath[1:][::-1]
        #For tour, add the start at the end
        if vs == vt:
            retpath.append(retpath[0])
            
        #Now recover the true path
        real_path = []
        for v1,v2 in zip(retpath,retpath[1:]):
            real_path.extend(self.g_traj_dict[(v1,v2)])
        #remove duplicate adjacent elements
        real_path = [k for k, g in itertools.groupby(real_path)]
        
        if vs!=vt:
            assert real_path[-1] == vt
            
        if vs==vt:
            assert real_path[-1] == vs
        
        if plot==True:
            self.sample_paths([real_path],True)

        return real_path
    """
    
def td_lambda_target(batch, max_episode_len, q_targets, args):    
    """calculate td-lambda return, q_targets: (episode_num, max_episode_len,n_agents)
    Ref: https://towardsdatascience.com/reinforcement-learning-td-%CE%BB-introduction-686a5e4f4e60
    """
    episode_num = batch['s'].shape[0]
    mask = (1-batch['padded'].float()).repeat(1,1, args.n_agents)
    terminated = (1-batch['terminated'].float()).repeat(1,1,args.n_agents)
    r = batch['r'].repeat(1,1,args.n_agents)
    
    n_step_return = torch.zeros(episode_num, max_episode_len, args.n_agents, max_episode_len)
    for transition_idx in range(max_episode_len - 1, -1, -1):
        n_step_return[:,transition_idx,:,0] = (r[:,transition_idx] + args.gamma * q_targets[:,transition_idx] * terminated[:,transition_idx]) * mask[:,transition_idx]
        for n in range(1, max_episode_len - transition_idx):
            n_step_return[:,transition_idx,:,n] = (r[:,transition_idx] + args.gamma * n_step_return[:,transition_idx+1,:,n-1]) * mask[:,transition_idx]
        
    lambda_return = torch.zeros(episode_num, max_episode_len, args.n_agents)
    for transition_idx in range(max_episode_len):
        returns = torch.zeros(episode_num, args.n_agents)
        for n in range(1, max_episode_len - transition_idx):
            returns += pow(args.td_lambda, n-1) * n_step_return[:,transition_idx,:,n-1]
            lambda_return[:,transition_idx] = (1-args.td_lambda) * returns + \
                pow(args.td_lambda, max_episode_len-transition_idx-1) * n_step_return[:,transition_idx,:,max_episode_len - transition_idx - 1]
    
    return lambda_return

if __name__=="__main__":
    #Basic Test
    config = load_config(config_path='config.yaml')
    gt = GraphTools('area_one',config)
    gt.plot_graph(paths_nodes=[[0,2,5],[26,25]])
    #gt.paths_evaluate([[0,2,5,6],[12,18,19],[14,15,16]],True)
    #gt.sample_paths([[0,2,5,6],[12,18,19],[14,15,16]],True)
    #gt.plot_graph([4,5,6])
    
    
    #gt.stainer_tsp(0,0,[13,6,7,8,4],plot=True)
    #gt.stainer_tsp(0,26,[13,6,7,8,4],plot=True)
    
  