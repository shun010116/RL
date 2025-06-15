import os
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.distributions import Normal

# Create directory for model and outputs
os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Hyperparameters
ENV_NAME = "BipedalWalker-v3"
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
UPDATE_EPOCHS = 10
BATCH_SIZE = 64
TIMESTEPS_PER_BATCH = 2048
TOTAL_TIMESTEPS = 1_000_000

# Actor-Critic Network
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
        value = self.critic(x)
        mean = self.actor(x)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist, value
    
# Buffer
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.acts = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.values = []

    def add(self, obs, act, reward, done, logprob, value):
        self.obs.append(obs)
        self.acts.append(act)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logprobs.append(logprob)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value):
        returns = []
        advantages = []
        gae = 0
        vals = self.values + [last_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + GAMMA * vals[t+1] * (1 - self.dones[t]) - vals[t]
            gae = delta + GAMMA * LAMBDA * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + vals[t])
        return (
            torch.tensor(np.array(self.obs), dtype=torch.float32),
            torch.tensor(self.acts, dtype=torch.float32),
            torch.tensor(self.logprobs, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32)
        )
        
# PPO Update
def ppo_update(model, optimizer_actor, optimizer_critic, obs, acts, old_logprobs, returns, advantages):
    for _ in range(UPDATE_EPOCHS):
        idxs = np.arange(len(obs))
        np.random.shuffle(idxs)
        for start in range(0, len(obs), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_idx = idxs[start:end]
            b_obs = obs[batch_idx]
            b_acts = acts[batch_idx]
            b_oldlog = old_logprobs[batch_idx]
            b_returns = returns[batch_idx]
            b_advs = advantages[batch_idx]

            dist, value = model(b_obs)
            logprob = dist.log_prob(b_acts).sum(axis=-1)
            ratio = (logprob - b_oldlog).exp()
            surr1 = ratio * b_advs
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * b_advs
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = ((value.squeeze() - b_returns) ** 2).mean()

            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

# Main Training
env = gym.make(ENV_NAME)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
model = ActorCritic(obs_dim, act_dim)
optimizer_actor = torch.optim.Adam(model.actor.parameters(), lr=ACTOR_LR)
optimizer_critic = torch.optim.Adam(model.critic.parameters(), lr=CRITIC_LR)

obs, _ = env.reset()
ep_rewards = []
avg_rewards = []

timestep = 0

print("Training started...")
while timestep < TOTAL_TIMESTEPS:
    buffer = RolloutBuffer()
    ep_reward = 0
    for _ in range(TIMESTEPS_PER_BATCH):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist, value = model(obs_tensor)
            action = dist.sample()
            logprob = dist.log_prob(action).sum()
        next_obs, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated
        buffer.add(obs, action.numpy(), reward, done, logprob.item(), value.item())
        obs = next_obs
        timestep += 1
        ep_reward += reward
        if done:
            obs, _ = env.reset()
            ep_rewards.append(ep_reward)
            ep_reward = 0

    with torch.no_grad():
        last_value = model(torch.tensor(obs, dtype=torch.float32))[1].item()

    obs_b, act_b, logprob_b, ret_b, adv_b = buffer.compute_returns_and_advantages(last_value)
    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8) # Normalize advantages
    ppo_update(model, optimizer_actor, optimizer_critic, obs_b, act_b, logprob_b, ret_b, adv_b)

    if len(ep_rewards) > 0:
        avg_reward = np.mean(ep_rewards[-10:])
        avg_rewards.append(avg_reward)
        print(f"Step: {timestep}, Average Reward (last 10): {avg_reward:.2f}")

# Plot rewards
plt.plot(avg_rewards)
plt.xlabel("Training Iteration")
plt.ylabel("Average Reward (Last 10 Episods)")
plt.title("PPO Training on BipedalWalker-v3")
plt.grid()
plt.savefig("outputs/reward_plot.png")
plt.show()

# Save model
torch.save(model.state_dict(), "model/ppo_bipedalwalker.pth")
print("모델 저장 완료: model/ppo_bipedalwalker.pth")

# Close environment
env.close()