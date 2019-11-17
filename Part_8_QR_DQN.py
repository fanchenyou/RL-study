"""
paper https://arxiv.org/abs/1710.10044

Original source https://github.com/senya-ashukha/quantile-regression-dqn-pytorch/blob/master/qr-dqn-solution-cool.ipynb

MountainCar problem: https://gym.openai.com/envs/MountainCar-v0/   https://github.com/openai/gym/wiki/MountainCar-v0
"""

import torch
import torch.nn as nn
import gym
import os

import os
import sys
import random
import numpy as np

from collections import OrderedDict
from tabulate import tabulate
from time import gmtime, strftime

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from IPython import display


class Logger:
    def __init__(self, name='name', fmt=None):
        self.handler = True
        self.scalar_metrics = OrderedDict()
        self.fmt = fmt if fmt else dict()

        base = './logs'
        if not os.path.exists(base): os.mkdir(base)

        time = gmtime()
        hash = ''.join([chr(random.randint(97, 122)) for _ in range(5)])
        fname = '-'.join(sys.argv[0].split('/')[-3:])
        self.path = '%s/%s-%s-%s-%s' % (base, fname, name, strftime('%m-%d-%H-%M', time), hash)

        self.logs = self.path + '.csv'
        self.output = self.path + '.out'
        self.checkpoint = self.path + '.cpt'

    def print0(self, *args):
        str_to_write = ' '.join(map(str, args))
        # with open(self.output, 'a') as f:
        #     f.write(str_to_write + '\n')
        #     f.flush()

        print(str_to_write)
        sys.stdout.flush()

    def add_scalar(self, t, key, value):
        if key not in self.scalar_metrics:
            self.scalar_metrics[key] = []
        self.scalar_metrics[key] += [(t, value)]

    def add_dict(self, t, d):
        for key, value in d.iteritems():
            self.add_scalar(t, key, value)

    def add(self, t, **args):
        for key, value in args.items():
            self.add_scalar(t, key, value)

    def iter_info(self, order=None):
        names = list(self.scalar_metrics.keys())
        if order:
            names = order
        values = [self.scalar_metrics[name][-1][1] for name in names]
        t = int(np.max([self.scalar_metrics[name][-1][0] for name in names]))
        fmt = ['%s'] + [self.fmt[name] if name in self.fmt else '.1f' for name in names]

        if self.handler:
            self.handler = False
            self.print0(tabulate([[t] + values], ['step'] + names, floatfmt=fmt))
        else:
            self.print0(tabulate([[t] + values], ['step'] + names, tablefmt='plain', floatfmt=fmt).split('\n')[1])


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, state, action, next_state, reward, done):
        transition = torch.Tensor([state]), torch.Tensor([action]), torch.Tensor([next_state]), \
                     torch.Tensor([reward]), torch.Tensor([done])
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
    def __init__(self, len_state, num_quant, num_actions):
        nn.Module.__init__(self)

        self.num_quant = num_quant
        self.num_actions = num_actions

        self.layer1 = nn.Linear(len_state, 256)
        self.layer2 = nn.Linear(256, num_actions * num_quant)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        return x.view(-1, self.num_actions, self.num_quant)

    def select_action(self, state, eps):
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor([state])
        action = torch.randint(0, self.num_actions, (1,))
        if random.random() > eps:
            action = self.forward(state).mean(2).max(1)[1]
        return int(action)


if __name__ == "__main__":
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    memory = ReplayMemory(10000)
    logger = Logger('q-net', fmt={'loss': '.5f'})

    eps_start, eps_end, eps_dec = 0.9, 0.1, 500
    eps = lambda steps: eps_end + (eps_start - eps_end) * np.exp(-1. * steps / eps_dec)

    Z = Network(len_state=len(env.reset()), num_quant=8, num_actions=env.action_space.n)
    Ztgt = Network(len_state=len(env.reset()), num_quant=8, num_actions=env.action_space.n)
    optimizer = torch.optim.Adam(Z.parameters(), 1e-3)

    steps_done = 0
    running_reward = None
    gamma, batch_size = 0.99, 32
    tau = torch.Tensor((2 * np.arange(Z.num_quant) + 1) / (2.0 * Z.num_quant)).view(1, -1)


    def huber(x, k=1.0):
        return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


    for episode in range(501):
        sum_reward = 0
        state = env.reset()
        while True:
            steps_done += 1

            action = Z.select_action(torch.Tensor([state]), eps(steps_done))
            next_state, reward, done, _ = env.step(action)

            memory.push(state, action, next_state, reward, float(done))
            sum_reward += reward

            if len(memory) < batch_size: break
            states, actions, rewards, next_states, dones = memory.sample(batch_size)

            # q_val of current states
            theta = Z(states)[np.arange(batch_size), actions]

            # q_val of next states
            Znext = Ztgt(next_states).detach()
            Znext_max = Znext[np.arange(batch_size), Znext.mean(2).max(1)[1]]

            # temporal difference
            Ttheta = rewards + gamma * (1 - dones) * Znext_max
            diff = Ttheta.t().unsqueeze(-1) - theta

            # Eq(10) in paper
            loss = huber(diff) * (tau - (diff.detach() < 0).float()).abs()
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state

            if steps_done % 100 == 0:
                Ztgt.load_state_dict(Z.state_dict())

            if done and episode % 50 == 0:
                logger.add(episode, steps=steps_done, running_reward=running_reward, loss=loss.data.numpy())
                logger.iter_info()

            if done:
                running_reward = sum_reward if not running_reward else 0.2 * sum_reward + running_reward * 0.8
                break

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
