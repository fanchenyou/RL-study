'''
This tutorial is from https://github.com/awjuliani/DeepRL-Agents/blob/master/Contextual-Policy.ipynb
The original Tensorflow implementation is replaced with Pytorch
Related article is https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149
'''

import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')


class QNet(nn.Module):
    def __init__(self, state_size=16, out_size=4):
        super(QNet, self).__init__()
        self.state_size = state_size
        self.out_size = out_size
        self.W = nn.Linear(state_size, out_size)
        nn.init.uniform_(self.W.weight.data, 0, 0.1)
        # self.data1 = self.W.weight.data.clone()

    def forward(self, inputs):
        out = self.W(inputs)
        _, argmax = out.max(-1)
        # print(out, argmax)
        # print(self.W.weight.data-self.data1)
        return out, argmax


criterion = nn.MSELoss()
model = QNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

y = .99
e = 0.2
num_episodes = 2000

success_steps = []
rewards_list = []

table = np.identity(16)
table = torch.from_numpy(table).float()

for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    if i % 100 == 0:
        print '%d/%d' % (i, num_episodes)

    # The Q-Network
    while j < 99:
        j += 1
        # print j

        with torch.no_grad():
            # Choose an action by greedily (with e chance of random action) from the Q-network
            inputs = table[s:s + 1]

            allQ, a = model(inputs)
            a = a.numpy()
            # print a[0]
            # print a,
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()

            # print a[0]
            # Get new state and reward from environment

            s1, r, d, _ = env.step(a[0])

            # Obtain the Q' values by feeding the new state through our network
            inputs2 = table[s1:s1 + 1]
            Q1, _ = model(inputs2)
            # print(Q1.size())

            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = torch.max(Q1)
            # print(maxQ1, r + y * maxQ1)
            targetQ = allQ.detach()
            # print(targetQ)
            # print(targetQ[0, a[0]], r + y * maxQ1)
            targetQ[0, a[0]] = r + y * maxQ1

        # print(allQ, targetQ)
        inputs = table[s:s + 1]
        Qout, _ = model(inputs)
        loss = criterion(Qout, targetQ)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rAll += r
        s = s1
        if d:
            # Reduce chance of random action as we train the model.
            e = 1. / ((i / 50) + 10)
            break
    if i % 100 == 0:
        print '%f' % (rAll)
    success_steps.append(j)
    rewards_list.append(rAll)

print("Percent of succesful episodes: " + str(sum(rewards_list) / num_episodes) + "%")

# print(rewards_list)
# print(success_steps)

plt.plot(rewards_list)
plt.plot(success_steps)
