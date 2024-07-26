import copy

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

from attention_net import AttentionNet
from runner import RLRunner
from parameters import *


# TensorBoard에 로깅하기 위한 SummaryWriter를 생성 및 모델과 gif를 저장할 디렉토리 생성
ray.init()
print("Welcome to PRM-AN!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

global_step = None


# Tensorboard에 로그를 작성하는 함수
def writeToTensorBoard(writer, tensorboardData, curr_episode, plotMeans=True):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    if plotMeans == True:
        tensorboardData = np.array(tensorboardData)
        tensorboardData = list(np.nanmean(tensorboardData, axis=0))
        metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'MI', 'F1Score', 'cov_trace']
        reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr = tensorboardData
    else:
        reward, value, policyLoss, valueLoss, entropy, gradNorm, returns, remain_budget, success_rate, RMSE, dct, MI, F1, cov_tr = tensorboardData

    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policyLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Value Loss', scalar_value=valueLoss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Grad Norm', scalar_value=gradNorm, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Returns', scalar_value=returns, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Remain Budget', scalar_value=remain_budget, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/RMSE', scalar_value=RMSE, global_step=curr_episode)
    writer.add_scalar(tag='Perf/F1 Score', scalar_value=F1, global_step=curr_episode)
    writer.add_scalar(tag='GP/MI', scalar_value=MI, global_step=curr_episode)
    writer.add_scalar(tag='GP/Delta Cov Trace', scalar_value=dct, global_step=curr_episode)
    writer.add_scalar(tag='GP/Cov Trace', scalar_value=cov_tr, global_step=curr_episode)


def main():
    # GPU 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_DEVICE)[1:-1]
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    
    # 사용하는 모델 초기화
    global_network = AttentionNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    # global_network.share_memory()
    global_optimizer = optim.Adam(global_network.parameters(), lr=LR)
    lr_decay = optim.lr_scheduler.StepLR(global_optimizer, step_size=DECAY_STEP, gamma=0.96)
    # Automatically logs gradients of pytorch model
    #wandb.watch(global_network, log_freq = SUMMARY_WINDOW)

    # 모델 불러올거면 불러오기
    best_perf = 900
    curr_episode = 0
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth')
        global_network.load_state_dict(checkpoint['model'])
        global_optimizer.load_state_dict(checkpoint['optimizer'])
        lr_decay.load_state_dict(checkpoint['lr_decay'])
        curr_episode = checkpoint['episode']
        print("curr_episode set to ", curr_episode)

        best_model_checkpoint = torch.load(model_path + '/best_model_checkpoint.pth')
        best_perf = best_model_checkpoint['best_perf']
        print('best performance so far:', best_perf)
        print(global_optimizer.state_dict()['param_groups'][0]['lr'])

    # meta agents가 뭔지는 모르겠음. agent니까 로봇을 의미하는 것 같기도 하고, 병렬로 몇 개를 실행할지 정하는 것 같기도 함.
    # NUM_META_AGENT 같은 변수는 parameters.py 파일에서 설정
    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get initial weigths
    if device != local_device:
        weights = global_network.to(local_device).state_dict()
        global_network.to(device)
    else:
        weights = global_network.state_dict()

    # launch the first job on each runner
    dp_model = nn.DataParallel(global_network)

    # jobList, metric_name 초기화.
    # jobList는 가중치, 현재 에피소드, 예산 범위, 샘플 크기, 샘플 길이를 저장.
    # sample_size는 한 에피소드에서 샘플링할 노드(포인트)의 개수를 의미함.
    # SMAPLE_LENGTH는 샘플 포인트 간의 이동 거리를 의미. 여기서는 0.2로 0.2m를 의미하는 것으로 보임
    # metric_name은 남은 예산, 경로 계획의 성공률, 에러(예측 정확도), 공분산 변화량, 상호 정보량, F1 스코어, 공분산을 저장함.
    jobList = []
    sample_size = np.random.randint(200,400)
    for i, meta_agent in enumerate(meta_agents):
        jobList.append(meta_agent.job.remote(weights, curr_episode, BUDGET_RANGE, sample_size, SAMPLE_LENGTH))
        curr_episode += 1
    metric_name = ['remain_budget', 'success_rate', 'RMSE', 'delta_cov_trace', 'MI', 'F1Score', 'cov_trace']
    tensorboardData = []
    trainingData = []
    experience_buffer = []
    for i in range(13):
        experience_buffer.append([])

    try:
        while True:
            # wait for any job to be completed
            # 로봇이 완료한 작업(=jobList에서 완료된 작업)을 기다림.
            done_id, jobList = ray.wait(jobList, num_returns=NUM_META_AGENT)
            # get the results
            # 완료된 작업의 결과 가져오고, shuffle
            done_jobs = ray.get(done_id)
            random.shuffle(done_jobs)
            #done_jobs = list(reversed(done_jobs))

            # 각 metric에 대한 결과와 작업의 결과를 저장
            # 이후 buffer에 jobResults를 저장
            perf_metrics = {}
            for n in metric_name:
                perf_metrics[n] = []
            for job in done_jobs:
                jobResults, metrics, info = job
                for i in range(13):
                    experience_buffer[i] += jobResults[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])
            
            # 모델의 성능을 확인하고, 필요하면 최적의 모델을 저장함
            # 성능 평가는 공분산 행렬의 추적값(perf_metrics['cov_trace'])을 사용하여 수행
            if np.mean(perf_metrics['cov_trace']) < best_perf and curr_episode % 32 == 0:
                best_perf = np.mean(perf_metrics['cov_trace'])
                print('Saving best model', end='\n')
                checkpoint = {"model": global_network.state_dict(),
                              "optimizer": global_optimizer.state_dict(),
                              "episode": curr_episode,
                              "lr_decay": lr_decay.state_dict(),
                              "best_perf": best_perf}
                path_checkpoint = "./" + model_path + "/best_model_checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')

            # experience_buffer가 충분히 쌓이면 모델 업데이트
            update_done = False
            while len(experience_buffer[0]) >= BATCH_SIZE:
                rollouts = copy.deepcopy(experience_buffer)
                for i in range(len(rollouts)):
                    rollouts[i] = rollouts[i][:BATCH_SIZE]
                for i in range(len(experience_buffer)):
                    experience_buffer[i] = experience_buffer[i][BATCH_SIZE:]
                if len(experience_buffer[0]) < BATCH_SIZE:
                    update_done = True
                if update_done:
                    experience_buffer = []
                    for i in range(13):
                        experience_buffer.append([])
                    sample_size = np.random.randint(200,400)

                # 데이터들을 배치 형태로 만들고, 디바이스에 전달
                # node_inputs : 에이전트가 현재 상태에서 관측한 노드 정보(좌표, 정보값, 표준편차 등)
                # edge_inputs : 노드들 간의 연결 관계(그래프 정보)
                # current_inputs : 현재 노드의 인덱스 정보
                # action : agent가 취한 액션 정보(다음 노드의 인덱스)
                # value : agent의 현재 상태에 대한 가치(value) 예측(현재 상태에서 예상하는 총 보상)
                # reward : agent가 각 단계에서 받은 보상(각 액션 후 받은 보상의 누적)
                # value_prime : 다음 상태에 대한 가치 예측
                # target_v : target_value로, agent의 학습 목표
                # budget_inputs : 예산
                # LSTM_h : LSTM의 hidden state
                # LSTM_c : LSTM의 cell state
                # mask : 그래프에서의 마스킹 정보(특정 노드들 간의 연결을 마스킹하기 위한 정보)
                # pos_encoding : 위치 인코딩 정보
                node_inputs_batch = torch.stack(rollouts[0], dim=0) # (batch,sample_size+2,2)
                edge_inputs_batch = torch.stack(rollouts[1], dim=0) # (batch,sample_size+2,k_size)
                current_inputs_batch = torch.stack(rollouts[2], dim=0) # (batch,1,1)
                action_batch = torch.stack(rollouts[3], dim=0) # (batch,1,1)
                value_batch = torch.stack(rollouts[4], dim=0) # (batch,1,1)
                reward_batch = torch.stack(rollouts[5], dim=0) # (batch,1,1)
                value_prime_batch = torch.stack(rollouts[6], dim=0) # (batch,1,1)
                target_v_batch = torch.stack(rollouts[7])
                budget_inputs_batch = torch.stack(rollouts[8], dim=0)
                LSTM_h_batch = torch.stack(rollouts[9])
                LSTM_c_batch = torch.stack(rollouts[10])
                mask_batch = torch.stack(rollouts[11])
                pos_encoding_batch = torch.stack(rollouts[12])

                if device != local_device:
                    node_inputs_batch = node_inputs_batch.to(device)
                    edge_inputs_batch = edge_inputs_batch.to(device)
                    current_inputs_batch = current_inputs_batch.to(device)
                    action_batch = action_batch.to(device)
                    value_batch = value_batch.to(device)
                    reward_batch = reward_batch.to(device)
                    value_prime_batch = value_prime_batch.to(device)
                    target_v_batch = target_v_batch.to(device)
                    budget_inputs_batch = budget_inputs_batch.to(device)
                    LSTM_h_batch = LSTM_h_batch.to(device)
                    LSTM_c_batch = LSTM_c_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    pos_encoding_batch = pos_encoding_batch.to(device)



                ### PPO ###
                
                # 기존 정책의 로그 확률 및 가치 계산
                # logp_list : 현재 정책의 각 행동에 대한 로그 확률
                # value : 현재 상태의 가치 예측
                # old_logp : agent가 실제로 취한 액션의 로그 확률
                # advantage : 실제 보상과 가치 예측의 차이
                with torch.no_grad():
                    logp_list, value, _, _ = global_network(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, mask_batch)
                old_logp = torch.gather(logp_list, 1 , action_batch.squeeze(1)).unsqueeze(1) # (batch_size,1,1)
                advantage = (reward_batch + GAMMA*value_prime_batch - value_batch) # (batch_size, 1, 1)
                #advantage = target_v_batch - value_batch

                # 정책의 엔트로피로 정책이 얼마나 다양한 행동을 하는지 나타냄.
                entropy = (logp_list*logp_list.exp()).sum(dim=-1).mean()

                scaler = GradScaler()

                # PPO 업데이트 루프
                for i in range(8):
                    with autocast():
                        logp_list, value, _, _ = dp_model(node_inputs_batch, edge_inputs_batch, budget_inputs_batch, current_inputs_batch, LSTM_h_batch, LSTM_c_batch, pos_encoding_batch, mask_batch)
                        logp = torch.gather(logp_list, 1, action_batch.squeeze(1)).unsqueeze(1)
                        ratios = torch.exp(logp-old_logp.detach())
                        surr1 = ratios * advantage.detach()
                        surr2 = torch.clamp(ratios, 1-0.2, 1+0.2) * advantage.detach()
                        policy_loss = -torch.min(surr1, surr2)
                        policy_loss = policy_loss.mean()

                        # value_clipped = value + (target_v_batch - value).clamp(-0.2, 0.2)
                        # value_clipped_loss = (value_clipped-target_v_batch).pow(2)
                        # value_loss =(value-target_v_batch).pow(2).mean()
                        # value_loss = torch.max(value_loss, value_clipped_loss).mean()

                        mse_loss = nn.MSELoss()
                        value_loss = mse_loss(value, target_v_batch).mean()

                        entropy_loss = (logp_list * logp_list.exp()).sum(dim=-1).mean()

                        loss = policy_loss + 0.5*value_loss + 0.0*entropy_loss
                    global_optimizer.zero_grad()
                    # loss.backward()
                    scaler.scale(loss).backward()
                    scaler.unscale_(global_optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(global_network.parameters(), max_norm=10, norm_type=2)
                    # global_optimizer.step()
                    scaler.step(global_optimizer)
                    scaler.update()
                lr_decay.step()

                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward_batch.mean().item(), value_batch.mean().item(), policy_loss.item(), value_loss.item(),
                        entropy.item(), grad_norm.item(), target_v_batch.mean().item(), *perf_data]
                trainingData.append(data)

                #experience_buffer = []
                #for i in range(8):
                #    experience_buffer.append([])

            if len(trainingData) >= SUMMARY_WINDOW:
                writeToTensorBoard(writer, trainingData, curr_episode)
                trainingData = []

            # get the updated global weights
            if update_done == True:
                if device != local_device:
                    weights = global_network.to(local_device).state_dict()
                    global_network.to(device)
                else:
                    weights = global_network.state_dict()
            
            jobList = []                                                                                    
            for i, meta_agent in enumerate(meta_agents):                                                    
                jobList.append(meta_agent.job.remote(weights, curr_episode, BUDGET_RANGE, sample_size, SAMPLE_LENGTH))
                curr_episode += 1 
            
            if curr_episode % 32 == 0:
                print('Saving model', end='\n')
                checkpoint = {"model": global_network.state_dict(),
                              "optimizer": global_optimizer.state_dict(),
                              "episode": curr_episode,
                              "lr_decay": lr_decay.state_dict()}
                path_checkpoint = "./" + model_path + "/checkpoint.pth"
                torch.save(checkpoint, path_checkpoint)
                print('Saved model', end='\n')
                    
    
    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


if __name__ == "__main__":
    main()
