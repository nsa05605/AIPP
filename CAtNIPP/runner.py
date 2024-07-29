import torch
import numpy as np
import ray
import os
from attention_net import AttentionNet
from worker import Worker
from parameters import *


class Runner(object):
    """Actor object to start running simulation on workers.
    Gradient computation is also executed on this object."""

    # 초기화.
    # 디바이스 설정하고, 네트워크 설정하고
    def __init__(self, metaAgentID):
        self.metaAgentID = metaAgentID
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.localNetwork = AttentionNet(INPUT_DIM, EMBEDDING_DIM)
        self.localNetwork.to(self.device)

    # 신경망 모델의 현재 가중치를 반환
    def get_weights(self):
        return self.localNetwork.state_dict()

    # 외부에서 가져온 가중치로 현재 신경망 가중치 설정(아래 job 함수와 eval/test_driver에서만 사용됨)
    def set_weights(self, weights):
        self.localNetwork.load_state_dict(weights)

    # 각 변수에 대해 먼저 다루면,
    # episodeNumber : 현재 에피소드 번호
    # budget_range : 예산 범위
    # sample_size, sample_length : 샘플링 크기와 길이
    def singleThreadedJob(self, episodeNumber, budget_range, sample_size, sample_length):
        # 결과 이미지 저장 여부
        save_img = True if (SAVE_IMG_GAP != 0 and episodeNumber % SAVE_IMG_GAP == 0) else False

        # Worker 객체 생성 및 에피소드 실행(work() 함수)
        # 이후 Worker 내부에서 Env 불러오고, Env 내부에 GP 환경 불러옴
        worker = Worker(self.metaAgentID, self.localNetwork, episodeNumber, budget_range, sample_size, sample_length, self.device, save_image=save_img, greedy=False)
        worker.work(episodeNumber)

        # 에피소드 실행 후 수집된 경험 데이터와 성능 메트릭 저장
        jobResults = worker.experience
        perf_metrics = worker.perf_metrics
        return jobResults, perf_metrics

    # driver.py에서 meta_agent.job.remote() 형태로 실행됨
    # 이 과정에서 singleThreadedJob() 메서드를 통해 에피소드 실행
    def job(self, global_weights, episodeNumber, budget_range, sample_size=SAMPLE_SIZE, sample_length=None):
        print("starting episode {} on metaAgent {}".format(episodeNumber, self.metaAgentID))
        # set the local weights to the global weight values from the master network
        self.set_weights(global_weights)

        jobResults, metrics = self.singleThreadedJob(episodeNumber, budget_range, sample_size, sample_length)

        info = {
            "id": self.metaAgentID,
            "episode_number": episodeNumber,
        }

        return jobResults, metrics, info

  
@ray.remote(num_cpus=1, num_gpus=len(CUDA_DEVICE)/NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, metaAgentID):        
        super().__init__(metaAgentID)


if __name__=='__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.singleThreadedJob.remote(1)
    out = ray.get(job_id)
    print(out[1])
