#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:09:42 2020

@author: yongyongwei
"""

from common.utils import load_config,RecursiveNamespace as RN
from common.utils import recursivenamespace_2dict as RN2dict
from ipp_envs import EnvMARL,EnvSeqRL,EnvABR
from runner import Runner
import pprint
import pdb

if __name__ == '__main__':
    
    config = load_config(config_path="config.yaml")
    
    #**************other parameter setting******************
    args = RN(**config)
    args.path_number = 2
    args.area_name = "area_one"
    args.chargingnodes = [0,26]
    args.B_range = [30,51]
    """Instructions for Algorithms: general rule: env+alg
    (1) sequential enviornment(no reward type required): 
            seq+[reinforce,dqn,a2c] 
    (2) multi-agent enviornment(work with reward type):
            ma + [mix, coma, reinforce, dqn, a2c]
            mix, coma (other MARL algs): 
                reward type = 0
            reinforce, dqn, a2c: 
                reward type = 0 (equally split)
                reward type = 1 (seqential assign)
                reward type = 2 (assign based marginal utility)
    (3) single agent with action branchcing (work with reward type):
            abr+[bdq]
            MARL based algs: reward type=0
            independent based algs: reward type=0, 1, 2
            
            For this case, like the sequential case, there is no difference of 
            reuse_network or full_observable. since implicityly already 
            reuse_network and full_observable
    """
    
    
    #Added
    args.nstep_return = 1
    #******************Choose the algorithm***********************************    

    args.reuse_network = False
    args.full_observable = True
    
    args.alg = 'ma+ligm'
    
    #args.alg = "jal+jal"
    #args.alg = "seq+dqn"
    args.reward_type=0
    args.state_type=0
    #********************
    
    #********************
    """
    args.alg = 'abr+abr_bdq'
    args.alg = 'abr+abr_a2c'
    args.alg = 'abr+abr_mix'
    args.mixer='vdn'
    args.reward_type=0
    """
    #********************
    
    
    
    #**************create the environment*************************************
    if args.alg.startswith("seq"):
        env = EnvSeqRL(args.area_name, config)
        env.get_dim_info(args.path_number, args.state_type, args)
    elif args.alg.startswith("ma"):
        env = EnvMARL(args.area_name, config)
        env.get_dim_info(args.path_number,args.reuse_network, args.full_observable,args.state_type,args)
    elif args.alg.startswith('abr'):
        env = EnvABR(args.area_name, config)
        env.get_dim_info(args.path_number, args.state_type, args)
    elif args.alg.startswith('jal'):
        env = EnvMARL(args.area_name, config)
        env.get_dim_info(args.path_number,args.reuse_network, args.full_observable,args.state_type,args)
        #Reset the number of actions and agents
        args.n_agents = 1
        args.n_robots = args.path_number
        args.n_actions = (env.num_of_nodes+1) ** args.path_number
        args.single_robot_action_size = (env.num_of_nodes+1) 

    else:
        raise Exception("Unkown env type")
    print('nn input shape:',args.nn_input_dim)
    #*************print the parameters*****************************************
    print(pprint.pformat(RN2dict(args),indent=4,width=1))
    
    #*************Running Entry************************************************
    evaluate_settings =  [{'vses':[0,26], 'Bs':[20,20], 'chargingnodes': args.chargingnodes}]
    runner = Runner(env, evaluate_settings, args)
    runner.run()
   
    #*************Plot the result**********************************************
   
    import pickle
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    if type(runner.agents) == list:
        res_folder = runner.agents[0].policy.output_dir
    else:
        res_folder = runner.agents.policy.output_dir
    record_name = 'records.pickle'
    if args.alg.startswith('ma') or args.alg.startswith('abr'):
        record_name = 'records_rt{}_st{}.pickle'.format(args.reward_type,args.state_type)
    elif args.alg.startswith('jal'):
        record_name = 'records_st{}.pickle'.format(args.state_type)
    with open(os.path.join(res_folder,record_name),'rb') as fin:
        dat = pickle.load(fin)
        training_rewards = dat['training_rewards']
        evaluate_rewards = dat['evaluate_rewards']
    print('max reward recorded',np.max(training_rewards))
  
    episode, episode_reward, episode_paths = runner.rolloutWorker.generate_episode(runner.evaluate_settings[0],evaluate=True)
    print(episode_reward, episode_paths)
    print(env.paths_evaluate(episode_paths))
    
    r_group = np.array(training_rewards).reshape(-1,50)
    r_group_mean = np.mean(r_group,axis=1)
    plt.plot(range(len(r_group_mean)),r_group_mean)
    plt.show()

    plt.plot(range(len(evaluate_rewards)),np.array(evaluate_rewards).squeeze(1))
    plt.show()

    #plot with conf region
    averaging_window = 50
    r_mean = np.array([np.mean(training_rewards[r_ind:r_ind+averaging_window]) for r_ind in range(0,len(training_rewards),averaging_window)])
    r_std = np.array([np.std(training_rewards[r_ind:r_ind+averaging_window]) for r_ind in range(0,len(training_rewards),averaging_window)])

    plt.plot(range(len(r_mean)),r_mean,color='r',linewidth=1.0)
    plt.fill_between(range(len(r_mean)),r_mean - r_std, r_mean+r_std,facecolor='lightgreen', alpha=0.2)
    plt.show()
