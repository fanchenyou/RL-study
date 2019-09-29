'''
Tutorial article is https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149
Original TF code is from https://github.com/awjuliani/DeepRL-Agents/blob/master/Contextual-Policy.ipynb

This code shows
    1. train a simple neural network in order to learn a policy for picking actions using feedback from the environment.
    2. use policy gradients to adjust NN's weights through gradient descent.
    3. e-greedy policy: most of the time our agent will choose the action that corresponds to the largest expected value,
       but occasionally, with e probability, it will choose randomly.
    4. policy loss equation: Loss = -log(pi)*A   (read tutorial for more information)
'''

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

'''
The Contextual Bandits
Here we define our contextual bandits. In this example, we are using three four-armed bandit. 
Each bandit has four arms that can be pulled. Each bandit has different success probabilities for each arm. 

The pullBandit function generates a random number from a normal distribution with a mean of 0. 
The lower the bandit number, the more likely a positive reward will be returned. We want our agent to learn to always 
choose the bandit-arm that will most often give a positive reward, depending on the Bandit presented.
'''


class ContextualBandit:
    def __init__(self):
        self.state = 0
        # List out our bandits. Currently arms 4, 2, and 1 (respectively) are the most optimal.
        self.bandits = np.array([[0.2, 0, -0.0, -5], [0.1, -5, 1, 0.25], [-5, 5, 5, 5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def getBandit(self):
        # Returns a random state for each episode, aka, which bandit machine (out of 3) we are using
        self.state = np.random.randint(0, len(self.bandits))
        return self.state

    def pullArm(self, action):
        # Get a random number.
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            # return a positive reward.
            return 1
        else:
            # return a negative reward.
            return -1


class QNet(nn.Module):
    def __init__(self, s_size, a_size):
        super(QNet, self).__init__()
        self.s_size = s_size
        self.fc1 = nn.Linear(s_size, a_size, bias=False)
        self.fc1.weight.data.fill_(1.0)

    def forward(self, state_in):
        state_in_one_hot = F.one_hot(state_in, num_classes=self.s_size)  # which bandit machine, encode in one-hot
        out = self.fc1(state_in_one_hot)
        # out = torch.sigmoid(out)
        out = out.view(-1)  # batch size is 1
        print(out.size())
        _, argmax = out.max(0)
        print(argmax)
        # print(out, argmax)
        # print(self.W.weight.data-self.data1)
        return out, argmax


class BanditLoss(nn.Module):
    # score is K, chose_action is scalar (1 out of K), rewards is scalar
    def forward(self, score, chosen_action, rewards):
        return - (F.logsigmoid(score[chosen_action]) * rewards)


cBandit = ContextualBandit()
s_size = cBandit.num_bandits
a_size = cBandit.num_actions
print(s_size, a_size)

criterion = BanditLoss()
myAgent = QNet(s_size, a_size)
optimizer = torch.optim.SGD(myAgent.parameters(), lr=0.001)

total_episodes = 10000  # Set total number of episodes to train agent on.
total_reward = np.zeros([cBandit.num_bandits, cBandit.num_actions])  # Set scoreboard for bandits to 0.
e = 0.1  # Set the chance of taking a random action.

exit()


i = 0
while i < total_episodes:
    s = cBandit.getBandit()  # Get a state from the environment.

    # Choose either a random action or one from our network.
    if np.random.rand(1) < e:
        action = np.random.randint(cBandit.num_actions)
    else:
        action = sess.run(myAgent.chosen_action, feed_dict={myAgent.state_in: [s]})

    reward = cBandit.pullArm(action)  # Get our reward for taking an action given a bandit.

    # Update the network.
    feed_dict = {myAgent.reward_holder: [reward], myAgent.action_holder: [action], myAgent.state_in: [s]}
    _, ww = sess.run([myAgent.update, weights], feed_dict=feed_dict)

    # Update our running tally of scores.
    total_reward[s, action] += reward
    if i % 500 == 0:
        print("Mean reward for each of the " + str(cBandit.num_bandits) + " bandits: " + str(
            np.mean(total_reward, axis=1)))
    i += 1



for a in range(cBandit.num_bandits):
    print("The agent thinks action " + str(np.argmax(ww[a]) + 1) + " for bandit " + str(
        a + 1) + " is the most promising....")
    if np.argmax(ww[a]) == np.argmin(cBandit.bandits[a]):
        print("...and it was right!")
    else:
        print("...and it was wrong!")