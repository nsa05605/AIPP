# Adaptive Informative Path Planning (AIPP) 코드 리뷰

## 1. CAtNIPP: Context-Aware Attention-based Network for Informative Path Planning
### 논문 소개
- 2차원 AIPP 연구
- GP 기반으로 환경 모델링 및 불확실성 감소를 보상으로 제공
- Attention과 LSTM 구조를 사용하여 그래프 기반 path planning 구조 사용
### 코드 리뷰 진행 상황
- 코드 각 부분에 대한 주석 작성 완료
- 코드 흐름 정리    
  - driver.py main() 함수에서 AttentionNet, RLRunner 클래스 호출
  - runner.py \__init__함수에서 AttentionNet 초기화 및 singleThrededJob() 함수에서 Worker 클래스 호출
  - worker.py \__init__함수에서 Env 클래스 초기화  
  - env.py reset() 함수에서 GaussianProcessForIPP 클래스 초기화 및 step() 함수에서 GP 업데이트  
    -> 이 과정에서 cov_trace 사용하여 reward 계산  
## 2. Deep Reinforcement Learning with Dynamic Graphs for Adaptive Informative Path Planning
### 논문 소개
- CAtNIPP 코드 기반으로 작성된 3차원 AIPP 연구
- 과수원 환경에서 GP 기반으로 환경 모델링
- 더 많은 정보(fruit)를 취득하는 방향으로 이동하도록 설계
### 코드 리뷰 진행 상황
- 전체적인 구조는 CAtNIPP와 유사해서 GP 기반 보상 설계하는 부분 확인 예정
## 3. Informative Path Planning for Mobile Sensing with Reinforcement Learning
### 논문 소개
