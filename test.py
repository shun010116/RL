import os
import torch
import gym
import numpy as np
from torch import nn
from torch.distributions import Normal
from gym.wrappers import RecordVideo

# 환경 이름
ENV_NAME = "BipedalWalker-v3"
VIDEO_DIR = "./videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

# 환경 설정 (시각화용)
env = gym.make(ENV_NAME, render_mode="rgb_array")
env = RecordVideo(env, VIDEO_DIR, episode_trigger=lambda e: True)

# Actor-Critic 정의
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        mean = self.actor(x)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist
    
# 환경 정보
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# 모델 초기화 및 가중치 로드
model = ActorCritic(obs_dim, act_dim)
model.load_state_dict(torch.load("model/ppo_bipedalwalker.pth"))
model.eval()

# 테스트 실행
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        dist = model(obs_tensor)
        action = dist.mean # deterministic 행동
    obs, reward, terminated, truncated, _ = env.step(action.numpy())
    done = terminated or truncated
    total_reward += reward 

print(f"테스트 완료, 총 보상: {total_reward:.2f}")
env.close()