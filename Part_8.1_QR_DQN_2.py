"""
paper https://arxiv.org/abs/1710.10044

Original source https://github.com/higgsfield/RL-Adventure/blob/master/8.quantile%20regression%20dqn.ipynb
"""

import math, random
import gym
import numpy as np
import torch
import torch.nn as nn

from utils.replay_memory import ReplayBuffer

class QRDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_quants):
        super(QRDQN, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_quants = num_quants

        self.features = nn.Sequential(
            nn.Linear(num_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions * self.num_quants)
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = self.features(x)
        x = x.view(batch_size, self.num_actions, self.num_quants)

        return x

    def q_values(self, x):
        x = self.forward(x)
        return x.mean(2)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(np.array(state, dtype=np.float32)).unsqueeze(0)
            qvalues = self.forward(state).mean(2)
            action = qvalues.max(1)[1]
            action = action.data.cpu().numpy()[0]
        else:
            action = random.randrange(self.num_actions)
        return action


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


if __name__ == "__main__":
    env_id = "CartPole-v0"
    env = gym.make(env_id)

    num_quant = 51
    Vmin = -10
    Vmax = 10

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    num_frames = 10000
    batch_size = 32
    gamma = 0.99

    losses = []
    all_rewards = []
    episode_reward = 0

    current_model = QRDQN(env.observation_space.shape[0], env.action_space.n, num_quant)
    target_model = QRDQN(env.observation_space.shape[0], env.action_space.n, num_quant)


    def compute_td_loss(batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(np.float32(done))

        # sample and get current state value
        dist = current_model(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_quant)
        dist = dist.gather(1, action).squeeze(1)
        # print(dist.size())

        # get next state value
        next_dist = target_model(next_state)
        next_action = next_dist.mean(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_quant)
        next_dist = next_dist.gather(1, next_action).squeeze(1).cpu().data
        dist_target = reward.unsqueeze(1) + 0.99 * next_dist * (1 - done.unsqueeze(1))

        quant_idx = torch.sort(dist, 1, descending=False)[1]
        tau_hat = torch.linspace(0.0, 1.0 - 1. / num_quant, num_quant) + 0.5 / num_quant  # Lemma2
        tau_hat = tau_hat.unsqueeze(0).repeat(batch_size, 1)
        quant_idx = quant_idx.cpu().data
        batch_idx = np.arange(batch_size)
        tau = tau_hat[:, quant_idx][batch_idx, batch_idx]

        k = 1

        u = dist_target - dist
        huber_loss = 0.5 * u.abs().clamp(min=0.0, max=k).pow(2)
        huber_loss += k * (u.abs() - u.abs().clamp(min=0.0, max=k))
        quantile_loss = (tau - (u < 0).float()).abs() * huber_loss
        loss = quantile_loss.sum() / num_quant

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(current_model.parameters(), 0.5)
        optimizer.step()

        return loss


    optimizer = torch.optim.Adam(current_model.parameters())

    replay_buffer = ReplayBuffer(10000)

    update_target(current_model, target_model)

    state = env.reset()
    for frame_idx in range(1, num_frames + 1):
        action = current_model.act(state, epsilon_by_frame(frame_idx))

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > batch_size:
            loss = compute_td_loss(batch_size)
            losses.append(loss.item())

        if frame_idx % 200 == 0:
            # plot(frame_idx, all_rewards, losses)
            print('frame %s. reward: %s' % (frame_idx, np.mean(all_rewards[-10:])))

        if frame_idx % 1000 == 0:
            update_target(current_model, target_model)
