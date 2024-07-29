import warnings
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from classes.Gaussian2D import Gaussian2D
from parameters import *


class GaussianProcessForIPP():
    # 모델 초기화
    def __init__(self, node_coords):
        # 커널 함수 정의하고,
        # GPR 모델은 Scikit-learn 사용해서 초기화
        # 이후 env.py에서 진행하면서 관측값을 넣어줄 리스트 2개 정의하고,
        # 입력받은 node_coords로 초기화. 이때 node_coords는 env.py에서 PRMController.py를 통해 정의한 것으로 collisionFree 노드들을 말하는 것 같음
        # self.kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))
        # self.kernel = RBF(0.2)
        self.kernel = Matern(length_scale=0.45)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, optimizer=None, n_restarts_optimizer=0)
        self.observed_points = []
        self.observed_value = []
        self.node_coords = node_coords

    # env.py의 step() 함수에서 사용되는 것으로, 관측값 넣어주는 부분
    def add_observed_point(self, point_pos, value):
        self.observed_points.append(point_pos)
        self.observed_value.append(value)

    # GP 업데이트는 관측값이 들어오면, gp.fit() 함수로 GP 모델 최적화
    def update_gp(self):
        if self.observed_points:
            X = np.array(self.observed_points).reshape(-1,2)
            y = np.array(self.observed_value).reshape(-1,1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(X, y)

    # 예측하려는 좌표(node_coords)를 넣어서 해당 위치의 강도(y_pred)와 표준편차(std)를 계산
    def update_node(self):
        y_pred, std = self.gp.predict(self.node_coords, return_std=True)
        return y_pred, std

    # GT가 있으면 GT랑 y_pred 비교
    def evaluate_RMSE(self, y_true):
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)
        x1x2 = np.array(list(product(x1, x2)))
        y_pred, std = self.gp.predict(x1x2, return_std=True)
        RMSE = np.sqrt(mean_squared_error(y_pred, y_true))
        return RMSE

    # GT 사용해서 F1score도 계산
    def evaluate_F1score(self, y_true):
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)
        x1x2 = np.array(list(product(x1, x2)))
        score = self.gp.score(x1x2,y_true)
        return score

    # 공분산 계산해서 불확실성 계산하는 것에 쓰는 것 같은데, 확실하지 않음
    def evaluate_cov_trace(self, X=None):
        if X is None:
            x1 = np.linspace(0, 1, 30)
            x2 = np.linspace(0, 1, 30)
            X = np.array(list(product(x1, x2)))
        _, std = self.gp.predict(X, return_std=True)
        trace = np.sum(std*std)
        return trace

    def evaluate_mutual_info(self, X=None):
        if X is None:
            x1 = np.linspace(0, 1, 30)
            x2 = np.linspace(0, 1, 30)
            X = np.array(list(product(x1, x2)))
        n_sample = X.shape[0]
        _, cov = self.gp.predict(X, return_cov=True)
        
        mi = (1 / 2) * np.log(np.linalg.det(0.01*cov.reshape(n_sample, n_sample) + np.identity(n_sample)))
        return mi

    # high_info_area(=interest area)를 정의?하는 함수
    # GP로 예측해서 값이 threshold보다 크면 그 노드를 관심 지역으로 추가함
    def get_high_info_area(self, t=ADAPTIVE_TH, beta=1):
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)
        x1x2 = np.array(list(product(x1, x2)))
        y_pred, std = self.gp.predict(x1x2, return_std=True)
        
        high_measurement_area = []
        for i in range(900):
            if y_pred[i] + beta * std[i] >= t:
                high_measurement_area.append(x1x2[i])
        high_measurement_area = np.array(high_measurement_area)
        return high_measurement_area

    def plot(self, y_true):
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)

        x1x2 = np.array(list(product(x1, x2)))
        y_pred, std = self.gp.predict(x1x2, return_std=True)

        X0p, X1p = x1x2[:, 0].reshape(30, 30), x1x2[:, 1].reshape(30, 30)
        y_pred = np.reshape(y_pred, (30, 30))
        std = std.reshape(30,30)

        X = np.array(self.observed_points)

        fig = plt.figure(figsize=(6,6))
        #if self.observed_points:
        #    plt.scatter(X[:, 0].reshape(1, -1), X[:, 1].reshape(1, -1), s=10, c='r')
        plt.subplot(2, 2, 2) # ground truth
        plt.title('Ground truth')
        plt.pcolormesh(X0p, X1p, y_true.reshape(30, 30), shading='auto', vmin=0, vmax=1)
        plt.subplot(2, 2, 3) # stddev
        plt.title('Predict std')
        plt.pcolormesh(X0p, X1p, std, shading='auto', vmin=0, vmax=1)
        plt.subplot(2, 2, 1) # mean
        plt.title('Predict mean')
        plt.pcolormesh(X0p, X1p, y_pred, shading='auto', vmin=0, vmax=1)
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        # if self.observed_points:
        #     plt.scatter(X[:, 0].reshape(1, -1), X[:, 1].reshape(1, -1), s=2, c='r')
        # plt.show()
        return y_pred

if __name__ == '__main__':
    example = Gaussian2D()
    x1 = np.linspace(0, 1)
    x2 = np.linspace(0, 1)
    x1x2 = np.array(list(product(x1, x2)))
    y_true = example.distribution_function(X=x1x2)
    # print(y_true.shape)
    node_coords = np.random.uniform(0,1,(100,2))
    gp_ipp = GaussianProcessForIPP(node_coords)
    gp_ipp.plot(y_true.reshape(50,50))
    for i in range(node_coords.shape[0]):
        y_observe = example.distribution_function(node_coords[i].reshape(-1,2))
        # print(node_coords[i], y_observe)
        gp_ipp.add_observed_point(node_coords[i], y_observe)
        gp_ipp.update_gp()
        y_pre, std = gp_ipp.update_node()
        print(gp_ipp.evaluate_cov_trace())
    gp_ipp.plot(y_true)
    print(gp_ipp.evaluate_F1score(y_true))
    print(gp_ipp.gp.kernel_)







