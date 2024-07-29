import copy
import os

import imageio
import numpy as np
import torch
from env import Env
from attention_net import AttentionNet
from parameters import *
import scipy.signal as signal

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class Worker:
    def __init__(self, metaAgentID, localNetwork, global_step, budget_range, sample_size=SAMPLE_SIZE, sample_length=None, device='cuda', greedy=False, save_image=False):

        self.device = device
        self.greedy = greedy
        self.metaAgentID = metaAgentID
        self.global_step = global_step
        self.save_image = save_image
        self.sample_length = sample_length
        self.sample_size = sample_size

        self.env = Env(sample_size=self.sample_size, k_size=K_SIZE, budget_range=budget_range, save_image=self.save_image)
        # self.local_net = AttentionNet(2, 128, device=self.device)
        # self.local_net.to(device)
        self.local_net = localNetwork
        self.experience = None

    # 에피소드 실행
    def run_episode(self, currEpisode):
        # 에이전트 경험을 저장할 episode_buffer와 성능 메트릭을 저장한 딕셔너리 perf_metrics
        episode_buffer = []
        perf_metrics = dict()
        for i in range(13):
            episode_buffer.append([])

        # 환경 초기화
        done = False
        node_coords, graph, node_info, node_std, budget = self.env.reset()
        
        # 각 노드의 정보를 준비하고, 텐서 형태로 변환하여 모델에 입력
        n_nodes = node_coords.shape[0]
        node_info_inputs = node_info.reshape((n_nodes, 1))
        node_std_inputs = node_std.reshape((n_nodes,1))
        budget_inputs = self.calc_estimate_budget(budget, current_idx=1)
        node_inputs = np.concatenate((node_coords, node_info_inputs, node_std_inputs), axis=1)
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device) # (1, sample_size+2, 4)
        budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device) # (1, sample_size+2, 1)
        
        # 그래프의 연결 정보를 준비하고, 위치 인코딩을 계산하여 모델에 입력
        graph = list(graph.values())
        edge_inputs = []
        for node in graph:
            node_edges = list(map(int, node))
            edge_inputs.append(node_edges)

        pos_encoding = self.calculate_position_embedding(edge_inputs)
        pos_encoding = torch.from_numpy(pos_encoding).float().unsqueeze(0).to(self.device) # (1, sample_size+2, 32)

        edge_inputs = torch.tensor(edge_inputs).unsqueeze(0).to(self.device) # (1, sample_size+2, k_size)

        # 현재 노드 인덱스를 텐서로 변환하고, 초기 경로를 설정
        current_index = torch.tensor([self.env.current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device) # (1,1,1)
        route = [current_index.item()]

        # LSTM의 초기 hidden_state와 cell_state를 초기화
        LSTM_h = torch.zeros((1,1,EMBEDDING_DIM)).to(self.device)
        LSTM_c = torch.zeros((1,1,EMBEDDING_DIM)).to(self.device)

        mask = torch.zeros((1, self.sample_size+2, K_SIZE), dtype=torch.int64).to(self.device)

        # 에피소드 시작 루프
        for i in range(256):
            # episode_buffer에 기록
            episode_buffer[9] += LSTM_h
            episode_buffer[10] += LSTM_c
            episode_buffer[11] += mask
            episode_buffer[12] += pos_encoding

            # self.local_net은 AttentionNet 모델을 의미
            # logp_list는 각 행동에 대한 로그 확률 리스트
            # value는 현재 상태의 가치
            with torch.no_grad():
                logp_list, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
            # greedy가 True인 경우에는 logp_list에서 가장 높은 확률을 가진 행동을 선택하고,
            # greedy가 False인 경우에는 logp_list를 확률 분포로 변환한 후, 확률에 따라 행동을 무작위로 선택함(multinomial(A, 1) 이면 A 분포에서 1개 샘플링한다는 의미)
            # 일단 기본 설정은 greedy=False
            if self.greedy:
                action_index = torch.argmax(logp_list, dim=1).long()
            else:
                action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

            episode_buffer[0] += node_inputs
            episode_buffer[1] += edge_inputs
            episode_buffer[2] += current_index
            episode_buffer[3] += action_index.unsqueeze(0).unsqueeze(0)
            episode_buffer[4] += value
            episode_buffer[8] += budget_inputs 

            # 다음으로 방문할 노드 인덱스 계산 및 경로(route)에 기록
            next_node_index = edge_inputs[:, current_index.item(), action_index.item()]
            route.append(next_node_index.item())
            # 이후 한 단계 실행해서 각 정보 계산
            reward, done, node_info, node_std, remain_budget = self.env.step(next_node_index.item(), self.sample_length)
            #if (not done and i==127):
                #reward += -np.linalg.norm(self.env.node_coords[self.env.current_node_index,:]-self.env.node_coords[0,:])

            episode_buffer[5] += torch.FloatTensor([[[reward]]]).to(self.device)

            # 현재 노드 인덱스 업데이트하고, 나머지 정보들도 업데이트
            current_index = next_node_index.unsqueeze(0).unsqueeze(0)
            node_info_inputs = node_info.reshape(n_nodes, 1)
            node_std_inputs = node_std.reshape(n_nodes, 1)
            budget_inputs = self.calc_estimate_budget(remain_budget, current_idx=current_index.item())
            node_inputs = np.concatenate((node_coords, node_info_inputs, node_std_inputs), axis=1)
            node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)
            budget_inputs = torch.FloatTensor(budget_inputs).unsqueeze(0).to(self.device)
            #print(node_inputs)
            
            # mask last five node
            mask = torch.zeros((1, self.sample_size+2, K_SIZE), dtype=torch.int64).to(self.device)
            #connected_nodes = edge_inputs[0, current_index.item()]
            #current_edge = torch.gather(edge_inputs, 1, current_index.repeat(1, 1, K_SIZE))
            #current_edge = current_edge.permute(0, 2, 1)
            #connected_nodes_budget = torch.gather(budget_inputs, 1, current_edge) # (1, k_size, 1)
            #n_available_node = sum(int(x>0) for x in connected_nodes_budget.squeeze(0))
            #if n_available_node > 5:
            #    for j, node in enumerate(connected_nodes.squeeze(0)):
            #        if node.item() in route[-2:]:
            #            mask[0, route[-1], j] = 1

            # 이미지 저장
            # save a frame
            if self.save_image:
                if not os.path.exists(gifs_path):
                    os.makedirs(gifs_path)
                self.env.plot(route, self.global_step, i, gifs_path)

            # 에피소드가 끝났으면, perf_metrics도 업데이트
            if done:
                episode_buffer[6] = episode_buffer[4][1:]
                episode_buffer[6].append(torch.FloatTensor([[0]]).to(self.device))
                if self.env.current_node_index == 0:
                    perf_metrics['remain_budget'] = remain_budget / budget
                    #perf_metrics['collect_info'] = 1 - remain_info.sum()
                    perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
                    perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(self.env.ground_truth)
                    perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - self.env.cov_trace
                    perf_metrics['MI'] = self.env.gp_ipp.evaluate_mutual_info(self.env.high_info_area)
                    perf_metrics['cov_trace'] = self.env.cov_trace
                    perf_metrics['success_rate'] = True
                    print('{} Goodbye world! We did it!'.format(i))
                else:
                    perf_metrics['remain_budget'] = np.nan
                    perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
                    perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(self.env.ground_truth)
                    perf_metrics['delta_cov_trace'] = self.env.cov_trace0 - self.env.cov_trace
                    perf_metrics['MI'] = self.env.gp_ipp.evaluate_MI(self.env.high_info_area)
                    perf_metrics['cov_trace'] = self.env.cov_trace
                    perf_metrics['success_rate'] = False
                    print('{} Overbudget!'.format(i))
                break
        # 아직 에피소드 끝나기 전이면, 현재 상태에서의 가치를 계산하고, perf_matrics 업데이트해서 기록함
        if not done:
            episode_buffer[6] = episode_buffer[4][1:]
            with torch.no_grad():
                 _, value, LSTM_h, LSTM_c = self.local_net(node_inputs, edge_inputs, budget_inputs, current_index, LSTM_h, LSTM_c, pos_encoding, mask)
            episode_buffer[6].append(value.squeeze(0))
            perf_metrics['remain_budget'] = remain_budget / budget
            perf_metrics['RMSE'] = self.env.gp_ipp.evaluate_RMSE(self.env.ground_truth)
            perf_metrics['F1Score'] = self.env.gp_ipp.evaluate_F1score(self.env.ground_truth)
            perf_metrics['delta_cov_trace'] =  self.env.cov_trace0 - self.env.cov_trace
            perf_metrics['MI'] = self.env.gp_ipp.evaluate_mutual_info(self.env.high_info_area)
            perf_metrics['cov_trace'] = self.env.cov_trace
            perf_metrics['success_rate'] = False

        print('route is ', route)
        reward = copy.deepcopy(episode_buffer[5])
        reward.append(episode_buffer[6][-1])
        for i in range(len(reward)):
            reward[i] = reward[i].cpu().numpy()
        reward_plus = np.array(reward,dtype=object).reshape(-1)
        discounted_rewards = discount(reward_plus, GAMMA)[:-1]
        discounted_rewards = discounted_rewards.tolist()
        target_v = torch.FloatTensor(discounted_rewards).unsqueeze(1).unsqueeze(1).to(self.device)

        for i in range(target_v.size()[0]):
            episode_buffer[7].append(target_v[i,:,:])

        # save gif
        if self.save_image:
            if self.greedy:
                from test_driver import result_path as path
            else:
                path = gifs_path
            self.make_gif(path, currEpisode)

        self.experience = episode_buffer
        return perf_metrics


    # runner.py 파일에서 에피소드를 실행하던 함수로, run_episode() 함수를 실행
    def work(self, currEpisode):
        '''
        Interacts with the environment. The agent gets either gradients or experience buffer
        '''
        self.currEpisode = currEpisode
        self.perf_metrics = self.run_episode(currEpisode)


    # 남은 예산을 계산하는 친구
    # 현재 노드에서 각 노드까지의 거리와 최종 목표 지점까지의 거리를 고려하여 예산을 추정하는 함수
    # 각 노드까지 갔다가 최종 노드까지 가는 예산을 모두 저장해두어 이를 고려하여 경로 계획을 하도록 함
    def calc_estimate_budget(self, budget, current_idx):
        all_budget = []
        current_coord = self.env.node_coords[current_idx]
        end_coord = self.env.node_coords[0]
        for i, point_coord in enumerate(self.env.node_coords):
            dist_current2point = self.env.prm.calcDistance(current_coord, point_coord)
            dist_point2end = self.env.prm.calcDistance(point_coord, end_coord)
            estimate_budget = (budget - dist_current2point - dist_point2end) / 10
            # estimate_budget = (budget - dist_current2point - dist_point2end) / budget
            all_budget.append(estimate_budget)
        return np.asarray(all_budget).reshape(i+1, 1)

    # 그래프의 위치 인코딩을 계산하는 함수
    # GPT 설명으로는, 그래프의 노드 간의 연결 관계를 나타내는 인접 행렬을 사용하여
    # 그래프 라플라시안 행렬을 만들고, 그 행렬의 고유 벡터를 위치 인코딩으로 사용한다고 함
    def calculate_position_embedding(self, edge_inputs):
        # A와 D는 각각 인접행렬과 대각행렬
        A_matrix = np.zeros((self.sample_size+2, self.sample_size+2))
        D_matrix = np.zeros((self.sample_size+2, self.sample_size+2))
        # 인접행렬 A 초기화
        # 노드 i, j가 연결되어 있고, 둘이 다르면 A[i][j]를 1로 설정
        for i in range(self.sample_size+2):
            for j in range(self.sample_size+2):
                if j in edge_inputs[i] and i != j:
                    A_matrix[i][j] = 1.0
        # 대각행렬 D 초기화
        # D[i][i]를 노드 i에 연결된 노드 수의 제곱근의 역수로 설정
        for i in range(self.sample_size+2):
            D_matrix[i][i] = 1/np.sqrt(len(edge_inputs[i])-1)
        
        # 그래프 라플라시안 행렬 L 계산
        # L은 항등 행렬에서 D와 A의 곱을 뺀 값으로 정의
        L = np.eye(self.sample_size+2) - np.matmul(D_matrix, A_matrix, D_matrix)

        # 고유 값 및 고유 벡터 계산
        eigen_values, eigen_vector = np.linalg.eig(L)
        idx = eigen_values.argsort()
        eigen_values, eigen_vector = eigen_values[idx], np.real(eigen_vector[:, idx])
        eigen_vector = eigen_vector[:,1:32+1]
        return eigen_vector
    
    # 이미지 만드는 친구
    def make_gif(self, path, n):
        with imageio.get_writer('{}/{}_cov_trace_{:.4g}.gif'.format(path, n, self.env.cov_trace), mode='I', duration=0.5) as writer:
            for frame in self.env.frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.frame_files[:-1]:
            os.remove(filename)




if __name__=='__main__':
    device = torch.device('cuda')
    localNetwork = AttentionNet(INPUT_DIM, EMBEDDING_DIM).cuda()
    worker = Worker(1, localNetwork, 0, budget_range=(4, 6), save_image=False, sample_length=0.05)
    worker.run_episode(0)
