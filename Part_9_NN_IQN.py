'''
Please refer to the paper for detail explanations in code comments
https://arxiv.org/abs/1806.06923

Original code source
https://github.com/senya-ashukha/quantile-regression-dqn-pytorch/blob/master/qr-dqn-solution-cool.ipynb
https://github.com/sungyubkim/Deep_RL_with_pytorch/blob/master/6_Uncertainty_in_RL/6_3_IQN_ver_A.ipynb

'''

import os
import sys
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

import pickle
import time
from collections import deque, OrderedDict
from time import gmtime, strftime
import tabulate

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

'''DQN settings'''
# sequential images to define state
STATE_LEN = 4
# target policy sync interval
TARGET_REPLACE_ITER = 1
# simulator steps for start learning
LEARN_START = int(1e+3)
# (prioritized) experience replay memory size
MEMORY_CAPACITY = int(1e+5)
# simulator steps for learning interval
LEARN_FREQ = 4
# quantile numbers for QR-DQN
N_QUANT = 64
# quantiles
QUANTS = np.linspace(0.0, 1.0, N_QUANT + 1)[1:]

'''Environment Settings'''
# openai gym env name
env = gym.make('MountainCar-v0')

N_ACTIONS = env.action_space.n
assert N_ACTIONS == 2 or 3  # go left or right

N_STATES = env.observation_space.shape
# Total simulation step
STEP_NUM = int(1e+7)
# gamma for MDP
GAMMA = 0.99
# visualize for agent playing
RENDERING = False

'''Training settings'''
# mini-batch size
BATCH_SIZE = 128
# learning rage
LR = 1e-4
# epsilon-greedy
EPSILON = 1.0

'''Save&Load Settings'''
# check save/load
SAVE = True
LOAD = False
# save frequency
SAVE_FREQ = int(1e+3)
# paths for predction net, target net, result log
PRED_PATH = './tmp/model/iqn_pred_net.pkl'
TARGET_PATH = './tmp/model/iqn_target_net.pkl'
RESULT_PATH = './tmp/plots/result.pkl'


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, state, action, next_state, reward, done):
        transition = torch.Tensor([state]), torch.Tensor([action]), torch.Tensor([next_state]), torch.Tensor(
            [reward]), torch.Tensor([done])
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*sample)

        batch_state = torch.cat(batch_state)
        batch_action = torch.LongTensor(batch_action)
        batch_reward = torch.cat(batch_reward)
        batch_done = torch.cat(batch_done)
        batch_next_state = torch.cat(batch_next_state)

        return batch_state, batch_action, batch_reward.unsqueeze(1), batch_next_state, batch_done.unsqueeze(1)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self, len_state, hsize):
        nn.Module.__init__(self)

        self.num_quant = N_QUANT
        self.num_actions = N_ACTIONS
        self.fc1 = nn.Linear(len_state, hsize)

        # Eq(4), additional function for computing
        # embedding for the sampled quantile \tau
        self.phi = nn.Linear(1, hsize, bias=False)
        self.phi_bias = nn.Parameter(torch.zeros(hsize))
        # mapping the embedding to a single value
        # as quantile regression
        self.fc2 = nn.Linear(hsize, hsize)
        self.fc3 = nn.Linear(hsize, self.num_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)

        tau = torch.rand(N_QUANT, 1)  # sample N quantiles
        quants = torch.arange(0, N_QUANT, 1.0)  # 0,1,...,N-1

        # Eq(4), parameterize sampled quantiles
        cos_trans = torch.cos(quants * tau * 3.141592).unsqueeze(2)  # (N_QUANT, N_QUANT, 1)
        rand_feat = torch.relu(self.phi(cos_trans).mean(dim=1) + self.phi_bias.unsqueeze(0)).unsqueeze(0)

        # Eq for Z above Eq(4) -  Z(x,a) = f( psi(x) dot phi(tau) )
        x = x.view(x.size(0), -1).unsqueeze(1)  # (m, 1, 7 * 7 * 64)
        x = x * rand_feat  # (m, N_QUANT, h_size)
        x = torch.relu(self.fc2(x))  # (m, N_QUANT, h_size)

        # note that output of IQN is quantile values of value distribution
        action_value = self.fc3(x).transpose(1, 2)  # (m, N_ACTIONS, N_QUANT)
        return action_value, tau

    # def select_action(self, state, eps):
    #     if not isinstance(state, torch.Tensor):
    #         state = torch.Tensor([state])
    #     action = torch.randint(0, self.num_actions, (1,))
    #     if random.random() > eps:
    #         action = self.forward(state).mean(2).max(1)[1]
    #     return int(action)

    def select_action(self, x, eps):
        x = torch.FloatTensor(x)

        if np.random.uniform() >= eps:
            # greedy case
            action_value, tau = self.forward(x)  # (N_ENVS, N_ACTIONS, N_QUANT)
            action_value = action_value.mean(dim=2)
            action = torch.argmax(action_value, dim=1).data.cpu().numpy()
        else:
            # random exploration case
            action = np.random.randint(0, N_ACTIONS, (x.size(0)))
        return action


if __name__ == "__main__":
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    memory = ReplayMemory(10000)
    # episode step for accumulate reward
    result = deque(maxlen=100)

    eps = 1.0

    Z = Network(len(env.reset()), 64)  # pred_net
    Ztgt = Network(len(env.reset()), 64)  # target_net
    optimizer = torch.optim.Adam(Z.parameters(), 1e-3)

    steps_done = 0
    learn_step_counter = 0
    running_reward = None
    gamma, batch_size = 0.99, BATCH_SIZE

    state = env.reset()

    for step in range(1, STEP_NUM):

        with torch.no_grad():
            action = Z.select_action(torch.Tensor([state]), eps)
        next_state, reward, done, _ = env.step(action[0])
        memory.push(state, action, next_state, reward, float(done))
        result.append(reward)

        # annealing the epsilon(exploration strategy)
        if step <= int(1e+3):
            # linear annealing to 0.9 until million step
            eps -= 0.9 / 1e+3
        elif step <= int(1e+4):
            # linear annealing to 0.99 until the end
            eps -= 0.09 / (1e+4 - 1e+3)

        if (LEARN_START <= len(memory)) and (len(memory) % LEARN_FREQ == 0):

            states, actions, rewards, next_states, dones = memory.sample(batch_size)

            q_eval, q_eval_tau = Z(states)  # (32, 2, 64) (64, 1)
            mb_size = q_eval.size(0)
            q_eval = q_eval[np.arange(batch_size), actions]  # (m, N_QUANT)
            q_eval = q_eval.unsqueeze(2)  # (m, N_QUANT, 1)

            # get next state value
            q_next, q_next_tau = Ztgt(next_states)  # (m, N_ACTIONS, N_QUANT), (N_QUANT, 1)
            best_actions = q_next.mean(dim=2).argmax(dim=1)  # (m)
            # q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1)
            q_next = q_next[np.arange(batch_size), best_actions] # (m, N_QUANT)
            # print q_next.size(), rewards.size(), dones.size() # (32, 64), (32, 1), (32, 1)
            q_target = rewards + GAMMA * (1. - dones) * q_next  # (32,64) = (m , N_QUANT)
            q_target = q_target.unsqueeze(1).detach()  # (m , 1, N_QUANT)

            # quantile Huber loss, in section 2.3
            u = q_target - q_eval  # (m, N_QUANT, N_QUANT)
            tau = q_eval_tau.unsqueeze(0)  # (1, N_QUANT, 1)
            # note that tau is for present quantile
            weight = torch.abs(tau - u.le(0.).float())  # (m, N_QUANT, N_QUANT)
            loss = F.smooth_l1_loss(q_eval, q_target.detach(), reduction='none')
            # (m, N_QUANT, N_QUANT)
            loss = torch.mean(weight * loss, dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_rate = 0.01
            learn_step_counter += 1
            if learn_step_counter % TARGET_REPLACE_ITER == 0:
                # update target network parameters using prediction network
                for target_param, pred_param in zip(Ztgt.parameters(), Z.parameters()):
                    target_param.data.copy_((1.0 - update_rate) \
                                            * target_param.data + update_rate * pred_param.data)

        # print log and save
        if step % SAVE_FREQ == 0:
            # calc mean return
            mean_100_ep_return = round(np.mean(result), 2)
            result.append(mean_100_ep_return)
            # print log
            print('Used Step:', len(memory),
                  'EPS: ', round(EPSILON, 3),
                  '| Mean ep 100 return: ', mean_100_ep_return)

        next_state = state
        #env.render()

    actions = {
        'CartPole-v0': ['Left', 'Right'],
        'MountainCar-v0': ['Left', 'Non', 'Right'],
    }


    def get_plot(q):
        eps, p = 1e-8, 0
        x, y = [q[0] - np.abs(q[0] * 0.2)], [0]
        for i in range(0, len(q)):
            x += [q[i] - eps, q[i]]
            y += [p, p + 1 / len(q)]
            p += 1 / len(q)
        x += [q[i] + eps, q[i] + np.abs(q[i] * 0.2)]
        y += [1.0, 1.0]
        return x, y


    state, done, steps = env.reset(), False, 0
    if not os.path.isdir('./img'):
        os.mkdir('./img')

    while True:
        plt.clf()
        steps += 1
        action = Z.select_action(torch.Tensor([state]), eps(steps_done))
        state, reward, done, _ = env.step(action)

        plt.subplot(1, 2, 1)
        plt.title('step = %s' % steps)
        plt.imshow(env.render(mode='rgb_array'))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        Zval = Z(torch.Tensor([state])).detach().numpy()
        for i in range(env.action_space.n):
            x, y = get_plot(Zval[0][i])
            plt.plot(x, y, label='%s Q=%.1f' % (actions[env_name][i], Zval[0][i].mean()))
            plt.legend(bbox_to_anchor=(1.0, 1.1), ncol=3, prop={'size': 6})

        if done: break
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.savefig('img/%s.png' % steps)
    plt.clf()
