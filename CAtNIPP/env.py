import os
import copy
import numpy as np
from itertools import product
from classes import PRMController, Obstacle, Utils
from classes.Gaussian2D import Gaussian2D
from matplotlib import pyplot as plt
from gp_ipp import GaussianProcessForIPP
from parameters import ADAPTIVE_AREA



class Env():
    def __init__(self, sample_size=500, k_size=10, start=None, destination=None, obstacle=[], budget_range=None, save_image=False, seed=None):
        self.sample_size = sample_size
        self.k_size = k_size
        self.budget_range = budget_range
        self.budget = np.random.uniform(*self.budget_range)
        if start is None:
            self.start = np.random.rand(1, 2)
        else:
            self.start = np.array([start])
        # 목적지가 없으면, 그냥 랜덤으로 설정하는 듯
        if destination is None:
            self.destination = np.random.rand(1, 2)
        else:
            self.destination = np.array([destination])
        self.obstacle = obstacle
        self.seed = seed
        
        # generate PRM
        # PRM(Probabilistic RoadMap) 컨트롤러는 고차원 공간에서 경로 계획을 해결하는 데 사용되는 알고리즘
        # runPRM() 함수를 통해 node_coords라는 노드 집합을 초기화하는데, 이는 collision이 없는 노드를 의미함
        # runner.py의 singleThreadedJob() 함수가 실행될 때마다 Worker 클래스가 초기화되고, 그와 동시에 Env 클래스가 초기화되니까, 
        self.prm = PRMController(self.sample_size, self.obstacle, self.start, self.destination, self.budget_range,
                                 self.k_size)
        self.budget = np.random.uniform(*self.budget_range)
        self.node_coords, self.graph = self.prm.runPRM(saveImage=False, seed=seed)
        
        # underlying distribution
        self.underlying_distribution = None
        self.ground_truth = None
        self.high_info_area = None

        # GP
        self.gp_ipp = None
        self.node_info, self.node_std = None, None
        self.node_info0, self.node_std0, self.budget0 = copy.deepcopy((self.node_info, self.node_std,self.budget))
        self.RMSE = None
        self.F1score = None
        self.cov_trace = None
        self.MI = None
        self.MI0 = None
        
        # start point
        self.current_node_index = 1
        self.sample = self.start
        self.dist_residual = 0
        self.route = []

        self.save_image = save_image
        self.frame_files = []

    # worker에서 에피소드 실행할 때 한 번 사용함
    # 환경의 각 변수들 초기화하는 역할
    def reset(self, seed=None):
        # generate PRM
        # self.start = np.random.rand(1, 2)
        # self.destination = np.random.rand(1, 2)
        # self.prm = PRMController(self.sample_size, self.obstacle, self.start, self.destination, self.budget_range, self.k_size)
        # self.budget = np.random.uniform(*self.budget_range)
        # self.node_coords, self.graph = self.prm.runPRM(saveImage=False)
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)

        # underlying distribution
        # 확률 분포를 2차원 가우시안 분포로 설정하고,
        # GT를 가져온다는데, 그냥 정의한 함수로 분포를 만드는 것 같은데 잘은 모르겠음
        self.underlying_distribution = Gaussian2D()
        self.ground_truth = self.get_ground_truth()

        # initialize gp
        self.gp_ipp = GaussianProcessForIPP(self.node_coords)
        self.high_info_area = self.gp_ipp.get_high_info_area() if ADAPTIVE_AREA else None
        self.node_info, self.node_std = self.gp_ipp.update_node()
        
        # initialize evaluations
        #self.F1score = self.gp_ipp.evaluate_F1score(self.ground_truth)
        self.RMSE = self.gp_ipp.evaluate_RMSE(self.ground_truth)
        self.cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)
        self.MI = self.gp_ipp.evaluate_mutual_info(self.high_info_area)
        self.cov_trace0 = self.cov_trace

        # save initial state
        self.node_info0, self.node_std0, self.budget = copy.deepcopy((self.node_info, self.node_std,self.budget0))

        # start point
        self.current_node_index = 1
        self.sample = self.start
        self.dist_residual = 0
        self.route = []
        np.random.seed(None)

        return self.node_coords, self.graph, self.node_info, self.node_std, self.budget

    # worker.py에서 for문 내부에서 실행되는 함수로, 에피소드 내에서 스텝 진행하는 함수
    def step(self, next_node_index, sample_length, measurement=True):
        # 현재 노드와 다음 노드 사이의 거리 dist 계산
        # remain_length는 이동해야할 남은 거리를 의미
        dist = np.linalg.norm(self.node_coords[self.current_node_index] - self.node_coords[next_node_index])
        remain_length = dist

        # sample_length는 parameters.py 파일에서 정의하는데, 기존에 0.2로 설정됨
        # dist_residual
        next_length = sample_length - self.dist_residual
        reward = 0
        # 다음 노드가 0번이면==마지막 노드이면, done=True
        done = True if next_node_index == 0 else False
        no_sample = True
        # 이동해야할 거리가 다음 샘플 길이보다 큰 동안 반복
        # measurement=True이면, 샘플 위치에서 관측 값을 계산하고 gp_ipp 모델에 이를 전달
        while remain_length > next_length:
            if no_sample:
                self.sample = (self.node_coords[next_node_index] - self.node_coords[
                    self.current_node_index]) * next_length / dist + self.node_coords[self.current_node_index]
            else:
                self.sample = (self.node_coords[next_node_index] - self.node_coords[
                    self.current_node_index]) * next_length / dist + self.sample
            if measurement:
                observed_value = self.underlying_distribution.distribution_function(
                    self.sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)
            else:
                observed_value = np.array([0])
            self.gp_ipp.add_observed_point(self.sample, observed_value)

            remain_length -= next_length
            next_length = sample_length
            no_sample = False

        # GP 모델을 업데이트하고, 노드의 정보와 표준편차도 업데이트
        self.gp_ipp.update_gp()
        self.node_info, self.node_std = self.gp_ipp.update_node()

        # high_info_area를 정의하고, RMSE 계산 및 공분산 행렬의 trace 계산
        if measurement:
            self.high_info_area = self.gp_ipp.get_high_info_area() if ADAPTIVE_AREA else None
        #F1score = self.gp_ipp.evaluate_F1score(self.ground_truth)
            RMSE = self.gp_ipp.evaluate_RMSE(self.ground_truth)
            self.RMSE = RMSE
        cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)
        #self.F1score = F1score

        # 최근 두 경로가 중복되면 패널티(-0.1) 부과
        if next_node_index in self.route[-2:]:
            reward += -0.1
        # 공분산의 trace가 감소하면 보상(self.cov_trace)는 이전의 공분산 trace 의미
        elif self.cov_trace > cov_trace:
            reward += (self.cov_trace - cov_trace) / self.cov_trace
        self.cov_trace = cov_trace
        # 에피소드가 종료되면 추가 패널티 부과
        if done:
            reward -= cov_trace/900

        # 남은 거리와 예산을 업데이트하고, 노드 인덱스 업데이트
        self.dist_residual = self.dist_residual + remain_length if no_sample else remain_length
        self.budget -= dist
        self.current_node_index = next_node_index
        self.route.append(next_node_index)
        assert self.budget >= 0  # Dijsktra filter
         
        return reward, done, self.node_info, self.node_std, self.budget

    # 사용되는 곳 없음
    def route_step(self, route, sample_length, measurement=True):
        current_node = route[0]
        for next_node in route[1:]:
            dist = np.linalg.norm(current_node - next_node)
            remain_length = dist
            next_length = sample_length - self.dist_residual
            no_sample = True
            while remain_length > next_length:
                if no_sample:
                    self.sample = (next_node - current_node) * next_length / dist + current_node
                else:
                    self.sample = (next_node - current_node) * next_length / dist + self.sample
                observed_value = self.underlying_distribution.distribution_function(self.sample.reshape(-1, 2)) + np.random.normal(0, 1e-10)
                self.gp_ipp.add_observed_point(self.sample, observed_value)
                remain_length -= next_length
                next_length = sample_length
                no_sample = False

            self.dist_residual = self.dist_residual + remain_length if no_sample else remain_length
            self.dist_residual_tmp = self.dist_residual
            if measurement:
                self.budget -= dist
            current_node = next_node

        self.gp_ipp.update_gp()

        if measurement:
            self.high_info_area = self.gp_ipp.get_high_info_area() if ADAPTIVE_AREA else None
            cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)
            self.cov_trace = cov_trace
        else:
            cov_trace = self.gp_ipp.evaluate_cov_trace(self.high_info_area)

        return cov_trace

    def get_ground_truth(self):
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)
        x1x2 = np.array(list(product(x1, x2)))
        ground_truth = self.underlying_distribution.distribution_function(x1x2)
        return ground_truth

    def plot(self, route, n, step, path, testID=0, CMAES_route=False, sampling_path=False):
        # Plotting shorest path
        plt.switch_backend('agg')
        self.gp_ipp.plot(self.ground_truth)
        # plt.subplot(1,3,1)
        # plt.scatter(self.start[:,0], self.start[:,1], c='r', s=15)
        # plt.scatter(self.destination[:,0], self.destination[:,1], c='r', s=15)
        if CMAES_route:
            pointsToDisplay = route
        else:
            pointsToDisplay = [(self.prm.findPointsFromNode(path)) for path in route]
        x = [item[0] for item in pointsToDisplay]
        y = [item[1] for item in pointsToDisplay]
        for i in range(len(x)-1):
            plt.plot(x[i:i+2], y[i:i+2], c='black', linewidth=4, zorder=5, alpha=0.25+0.6*i/len(x))
        if sampling_path:
            pointsToDisplay2 = [(self.prm.findPointsFromNode(path)) for path in sampling_path]
            x0 = [item[0] for item in pointsToDisplay2]
            y0 = [item[1] for item in pointsToDisplay2]
            x1 = [item[0] for item in pointsToDisplay2[:3]]
            y1 = [item[1] for item in pointsToDisplay2[:3]]
            for i in range(len(x0) - 1):
                plt.plot(x0[i:i + 2], y0[i:i + 2], c='white', linewidth=4, zorder=5, alpha=1- 0.2 * i / len(x0))
            for i in range(len(x1) - 1):
                plt.plot(x1[i:i + 2], y1[i:i + 2], c='red', linewidth=4, zorder=6)

        plt.subplot(2,2,4)
        plt.title('High interest area')
        xh = self.high_info_area[:,0]
        yh = self.high_info_area[:,1]
        plt.hist2d(xh, yh, bins=30, range=[[0,1], [0,1]], vmin=0, vmax=1, rasterized=True)
        # plt.scatter(self.start[:,0], self.start[:,1], c='r', s=15)
        # plt.scatter(self.destination[:,0], self.destination[:,1], c='r', s=15)

        # x = [item[0] for item in pointsToDisplay]
        # y = [item[1] for item in pointsToDisplay]

        for i in range(len(x)-1):
            plt.plot(x[i:i+2], y[i:i+2], c='black', linewidth=4, zorder=5, alpha=0.25+0.6*i/len(x))
        plt.suptitle('Budget: {:.4g}/{:.4g},  Cov trace: {:.4g}'.format(
            self.budget, self.budget0, self.cov_trace))
        plt.tight_layout()
        plt.savefig('{}/{}_{}_{}_samples.png'.format(path, n, testID, step, self.sample_size), dpi=150)
        # plt.show()
        frame = '{}/{}_{}_{}_samples.png'.format(path, n, testID, step, self.sample_size)
        self.frame_files.append(frame)

if __name__=='__main__':
    env = Env(sample_size=200, budget_range=(7.999,8), save_image=True)
    nodes, graph, info, std, budget = env.reset()
    print(nodes)
    print(graph)

