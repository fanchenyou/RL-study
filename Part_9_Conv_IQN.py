import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random
import os
import pickle
import time
from collections import deque
from copy import deepcopy
from tqdm import tqdm

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from utils.replay_memory import ReplayBuffer, PrioritizedReplayBuffer
from utils.wrappers import wrap, wrap_cover, SubprocVecEnv

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
# number of environments for C51
N_ENVS = 16
# openai gym env name
ENV_NAME = 'BreakoutNoFrameskip-v4'
env = SubprocVecEnv([wrap_cover(ENV_NAME) for i in range(N_ENVS)])
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape
# Total simulation step
STEP_NUM = int(1e+7)
# gamma for MDP
GAMMA = 0.99
# visualize for agent playing
RENDERING = False

'''Training settings'''
# check GPU usage
USE_GPU = torch.cuda.is_available()
print('USE GPU: ' + str(USE_GPU))
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


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(STATE_LEN, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.phi = nn.Linear(1, 7 * 7 * 64, bias=False)
        self.phi_bias = nn.Parameter(torch.zeros(7 * 7 * 64))
        self.fc = nn.Linear(7 * 7 * 64, 512)

        # action value distribution
        self.fc_q = nn.Linear(512, N_ACTIONS)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x.size(0) : minibatch size
        mb_size = x.size(0)

        x = self.feature_extraction(x / 255.0)  # (m, 7 * 7 * 64)
        tau = torch.rand(N_QUANT, 1)  # (N_QUANT, 1)
        quants = torch.arange(0, N_QUANT, 1.0)
        if USE_GPU:
            tau = tau.cuda()
            quants = quants.cuda()
        cos_trans = torch.cos(quants * tau * 3.141592).unsqueeze(2)  # (N_QUANT, N_QUANT, 1)
        rand_feat = F.relu(self.phi(cos_trans).mean(dim=1) + self.phi_bias.unsqueeze(0)).unsqueeze(0)

        x = x.view(x.size(0), -1).unsqueeze(1)  # (m, 1, 7 * 7 * 64)
        x = x * rand_feat  # (m, N_QUANT, 7 * 7 * 64)
        x = F.relu(self.fc(x))  # (m, N_QUANT, 512)

        # note that output of IQN is quantile values of value distribution
        action_value = self.fc_q(x).transpose(1, 2)  # (m, N_ACTIONS, N_QUANT)

        return action_value, tau

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))


class DQN(object):
    def __init__(self):
        self.pred_net, self.target_net = ConvNet(), ConvNet()
        # sync eval target
        self.update_target(self.target_net, self.pred_net, 1.0)
        # use gpu
        if USE_GPU:
            self.pred_net.cuda()
            self.target_net.cuda()

        # simulator step conter
        self.memory_counter = 0
        # target network step counter
        self.learn_step_counter = 0

        # ceate the replay buffer
        self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)

        # define optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=LR)

    def update_target(self, target, pred, update_rate):
        # update target network parameters using predcition network
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_((1.0 - update_rate) \
                                    * target_param.data + update_rate * pred_param.data)

    def save_model(self):
        # save prediction network and target network
        self.pred_net.save(PRED_PATH)
        self.target_net.save(TARGET_PATH)

    def load_model(self):
        # load prediction network and target network
        self.pred_net.load(PRED_PATH)
        self.target_net.load(TARGET_PATH)

    def choose_action(self, x, EPSILON):
        x = torch.FloatTensor(x)
        if USE_GPU:
            x = x.cuda()

        if np.random.uniform() >= EPSILON:
            # greedy case
            action_value, tau = self.pred_net(x)  # (N_ENVS, N_ACTIONS, N_QUANT)
            action_value = action_value.mean(dim=2)
            action = torch.argmax(action_value, dim=1).data.cpu().numpy()
        else:
            # random exploration case
            action = np.random.randint(0, N_ACTIONS, (x.size(0)))
        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done))

    def learn(self):
        self.learn_step_counter += 1
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.update_target(self.target_net, self.pred_net, 1e-2)

        b_s, b_a, b_r, b_s_, b_d = self.replay_buffer.sample(BATCH_SIZE)
        b_w, b_idxes = np.ones_like(b_r), None

        b_s = torch.FloatTensor(b_s)
        b_a = torch.LongTensor(b_a)
        b_r = torch.FloatTensor(b_r)
        b_s_ = torch.FloatTensor(b_s_)
        b_d = torch.FloatTensor(b_d)

        if USE_GPU:
            b_s, b_a, b_r, b_s_, b_d = b_s.cuda(), b_a.cuda(), b_r.cuda(), b_s_.cuda(), b_d.cuda()

        # action value distribution prediction
        q_eval, q_eval_tau = self.pred_net(b_s)  # (m, N_ACTIONS, N_QUANT), (N_QUANT, 1)
        mb_size = q_eval.size(0)
        q_eval = torch.stack([q_eval[i].index_select(0, b_a[i]) for i in range(mb_size)]).squeeze(1)
        # (m, N_QUANT)
        q_eval = q_eval.unsqueeze(2)  # (m, N_QUANT, 1)
        # note that dim 1 is for present quantile, dim 2 si for next quantile

        # get next state value
        q_next, q_next_tau = self.target_net(b_s_)  # (m, N_ACTIONS, N_QUANT), (N_QUANT, 1)
        best_actions = q_next.mean(dim=2).argmax(dim=1)  # (m)
        q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1)
        # (m, N_QUANT)
        q_target = b_r.unsqueeze(1) + GAMMA * (1. - b_d.unsqueeze(1)) * q_next
        # (m, N_QUANT)
        q_target = q_target.unsqueeze(1).detach()  # (m , 1, N_QUANT)

        # quantile Huber loss
        u = q_target.detach() - q_eval  # (m, N_QUANT, N_QUANT)
        tau = q_eval_tau.unsqueeze(0)  # (1, N_QUANT, 1)
        # note that tau is for present quantile
        weight = torch.abs(tau - u.le(0.).float())  # (m, N_QUANT, N_QUANT)
        loss = F.smooth_l1_loss(q_eval, q_target.detach(), reduction='none')
        # (m, N_QUANT, N_QUANT)
        loss = torch.mean(weight * loss, dim=1).mean(dim=1)

        # calc importance weighted loss
        b_w = torch.Tensor(b_w)
        if USE_GPU:
            b_w = b_w.cuda()
        loss = torch.mean(b_w * loss)

        # backprop loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

# model load with check
if LOAD and os.path.isfile(PRED_PATH) and os.path.isfile(TARGET_PATH):
    dqn.load_model()
    pkl_file = open(RESULT_PATH,'rb')
    result = pickle.load(pkl_file)
    pkl_file.close()
    print('Load complete!')
else:
    result = []
    print('Initialize results!')

print('Collecting experience...')

# episode step for accumulate reward
epinfobuf = deque(maxlen=100)
# check learning time
start_time = time.time()

# env reset
s = np.array(env.reset())

for step in tqdm(range(1, STEP_NUM//N_ENVS+1)):
    a = dqn.choose_action(s, EPSILON)

    # take action and get next state
    s_, r, done, infos = env.step(a)
    # log arrange
    for info in infos:
        maybeepinfo = info.get('episode')
        if maybeepinfo: epinfobuf.append(maybeepinfo)
    s_ = np.array(s_)

    # clip rewards for numerical stability
    clip_r = np.sign(r)

    # store the transition
    for i in range(N_ENVS):
        dqn.store_transition(s[i], a[i], clip_r[i], s_[i], done[i])

    # annealing the epsilon(exploration strategy)
    if step <= int(1e+3):
        # linear annealing to 0.9 until million step
        EPSILON -= 0.9/1e+3
    elif step <= int(1e+4):
        # linear annealing to 0.99 until the end
        EPSILON -= 0.09/(1e+4 - 1e+3)

    # if memory fill 50K and mod 4 = 0(for speed issue), learn pred net
    if (LEARN_START <= dqn.memory_counter) and (dqn.memory_counter % LEARN_FREQ == 0):
        dqn.learn()

    # print log and save
    if step % SAVE_FREQ == 0:
        # check time interval
        time_interval = round(time.time() - start_time, 2)
        # calc mean return
        mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]),2)
        result.append(mean_100_ep_return)
        # print log
        print('Used Step:',dqn.memory_counter,
              'EPS: ', round(EPSILON, 3),
              '| Mean ep 100 return: ', mean_100_ep_return,
              '| Used Time:',time_interval)
        # save model
        dqn.save_model()
        pkl_file = open(RESULT_PATH, 'wb')
        pickle.dump(np.array(result), pkl_file)
        pkl_file.close()

    s = s_

    if RENDERING:
        env.render()