#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 08:38:29 2019
Script for synthesize problem instance for IPP
i.e, define the problem graph and the pilot locations
@author: yongyongwei
"""
import sys,os
sys.dont_write_bytecode = True

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle


#Set the X,Y coordinate for the vertices
grid_step = 2
mesh = np.meshgrid(range(1,19,2),range(1,19,2))
grid_coords = 1.0*np.column_stack((np.ravel(mesh[0]),np.ravel(mesh[1])))
pilot_num = 25

#add vertices
g = nx.Graph()
g.add_nodes_from(range(len(grid_coords)))
vertice_posdict = {i:grid_coords[i] for i in range(len(grid_coords))}
nx.set_node_attributes(g, vertice_posdict,'pos')

#add edges between neighbor nodes only
  
edge_dict={}
#Init the edge dict
nodes = g.nodes()
for i in nodes:
    for j in nodes:
        if i!=j:
            edge_dict[(i,j)]=False
#add edge
for i in nodes:
    for j in nodes:
        if i!=j and edge_dict[(i,j)] == False and edge_dict[(j,i)]==False:
            nodei_pos = g.nodes[i]['pos']
            nodej_pos = g.nodes[j]['pos']
            if nodei_pos[0] == nodej_pos[0] and np.abs(nodei_pos[1]-nodej_pos[1])== grid_step:
                 edge_dict[(i,j)]=grid_step
                 edge_dict[(j,i)]=grid_step
                 g.add_edge(i,j)                         
            elif nodei_pos[1] == nodej_pos[1] and np.abs(nodei_pos[0]-nodej_pos[0])==grid_step:
                 edge_dict[(i,j)]=grid_step
                 edge_dict[(j,i)]=grid_step
                 g.add_edge(i,j)
            else:
                pass
    


#Get the adjacent vertice list for each vertex
adj_dict={v:g.neighbors(v) for v in g.nodes()}
nx.set_node_attributes(g, adj_dict,'adj')

#Get random pilot locations
pilot_pos=np.column_stack((np.random.uniform(1,17,pilot_num),np.random.uniform(1,17,pilot_num)))


tg = g
nodes = [v for v in tg.nodes()]
nodeposes = {v: tg.nodes[v]['pos'] for v in tg.nodes()}
nodelabels = {v: v for v in nodes}
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

nx.draw_networkx_nodes(tg, nodeposes,nodes, node_color = 'g', node_size=150,ax = ax)
nx.draw_networkx_labels(tg, nodeposes, labels=nodelabels, font_size=8, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=ax)
nx.draw_networkx_edges(tg,nodeposes,edge_color='gold',style='-.')
ax.scatter(pilot_pos[:,0],pilot_pos[:,1],marker='+')
plt.show()

#Output the result instance
inst_name = "inst5"
out=open(os.path.join('prob',inst_name+".ipp"),'w')
out.write('NAME: %s\n' % inst_name)
out.write('TYPE: %s\n' % 'IPP')
out.write('DIMENSION: %d\n' % len(grid_coords))
out.write('EDGE_WEIGHT_TYPE: %s\n' % 'EUC_2D')
out.write('NODE_COORD_SECTION\n')
for v in g.nodes():
    out.write('%d %f %f\n'% (v, g.nodes[v]['pos'][0],g.nodes[v]['pos'][1]))
out.write('ADJACENT_NODES_SECTION\n')
for v in g.nodes():
    out.write('%d:' % v)
    for n in g.nodes[v]['adj']:
        out.write(' %d' % n)
    out.write('\n')
out.write('EOF')
out.close()

#Ouput put the pilot data
with open(os.path.join('data',inst_name+"_pilot.pickle"),'wb') as fout:
    pickle.dump([pilot_pos,{}],fout)
    
