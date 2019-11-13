"""
paper https://arxiv.org/abs/1710.10044

Original source https://github.com/higgsfield/RL-Adventure/blob/master/8.quantile%20regression%20dqn.ipynb
"""

import math, random
import gym
import numpy as np
import torch
import torch.nn as nn

from utils.wrappers import make_atari, wrap_deepmind, wrap_pytorch


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class QRCnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions, num_quants):
        super(QRCnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_quants = num_quants

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions * self.num_quants)
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = self.features(x)
        x = x.view(batch_size, -1)

        x = self.value(x)
        x = x.view(batch_size, self.num_actions, self.num_quants)

        return x

    def q_values(self, x):
        x = self.forward(x)
        return x.mean(2)

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

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
    env_id = "PongNoFrameskip-v4"
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    num_quant = 51
    Vmin = -10
    Vmax = 10

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    num_frames = 100000
    batch_size = 32
    gamma = 0.99

    losses = []
    all_rewards = []
    episode_reward = 0

    current_model = QRCnnDQN(env.observation_space.shape, env.action_space.n, num_quant)
    target_model = QRCnnDQN(env.observation_space.shape, env.action_space.n, num_quant)

    optimizer = torch.optim.Adam(current_model.parameters())

    replay_initial = 10000
    replay_buffer = ReplayBuffer(10000)

    update_target(current_model, target_model)


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
        tau_hat = torch.linspace(0.0, 1.0 - 1. / num_quant, num_quant) + 0.5 / num_quant
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
