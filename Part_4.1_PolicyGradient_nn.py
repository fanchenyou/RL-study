# coding=utf-8
'''
Tutorial article is https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-3-model-based-rl-9a6fe0cce99
Original TF code is from https://github.com/awjuliani/DeepRL-Agents/blob/master/Model-Network.ipynb


Physical environments take time to navigate, and the real environment may not reset.
We can build a model of the environment. With such a model, an agent can ‘imagine’ what it might be like to move around
the real environment, and we can train a policy on this imagined environment in addition to the real one.
If we were given a good enough model of an environment, an agent could be trained entirely on that model,
and even perform well when placed into a real environment for the first time.

This code shows
    1. train a neural network in order to learn a policy for picking actions using feedback from the environment.
    2. train a separate neural network to learn the environment.
    3. use policy gradients to adjust NN's weights through gradient descent.
    4. alternate training policy and env network
    5. buffer gradients of policy network
'''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gym

env = gym.make('CartPole-v0')

# hyperparameters
H = 8  # number of hidden layer neurons
learning_rate = 1e-2
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?

model_bs = 3  # Batch size when learning from model
real_bs = 3  # Batch size when learning from real environment

# model initialization
D = 4  # input dimensionality


# Policy Network
class PolicyNet(nn.Module):
    def __init__(self, in_size, h_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(in_size, h_size, bias=False)
        self.fc2 = nn.Linear(h_size, 1, bias=False)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = torch.relu(x)
        score = self.fc2(x)
        prob = torch.sigmoid(score)
        return score, prob


class BCE_Weight_Loss(nn.Module):
    def forward(self, input, target, advantage):
        # return F.binary_cross_entropy_with_logits(input, target, weight=advantage)
        loglik = torch.log(target * (target - input) + (1 - target) * (target + input))
        loss = -torch.mean(loglik * advantage)
        return loss


criterion_policy = BCE_Weight_Loss()
policy_net = PolicyNet(D, H)
optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)


class EnvNet(nn.Module):
    def __init__(self, in_size, h_size):
        super(EnvNet, self).__init__()
        self.fc1 = nn.Linear(in_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.out_o = nn.Linear(h_size, 4)
        self.out_r = nn.Linear(h_size, 1)
        self.out_d = nn.Linear(h_size, 1)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        observation = self.out_o(x)
        reward = self.out_r(x)
        done = self.out_d(x)

        return observation, reward, done


criterion_observation = nn.MSELoss()
criterion_reward = nn.MSELoss()
criterion_done = nn.BCEWithLogitsLoss()

env_net = EnvNet(5, 256)  # 256 - model layer size
optimizer_env = torch.optim.Adam(env_net.parameters(), lr=learning_rate)


# Helper-functions

def resetGradBuffer(gradBuffer):
    for k, v in gradBuffer.iteritems():
        gradBuffer[k] = v * 0
    return gradBuffer


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# This function uses our model to produce a new state when given a previous state and action
def stepModel(xs, action):
    toFeed = np.reshape(np.hstack([xs[-1][0], np.array(action)]), [1, 5])
    # myPredict = sess.run([predicted_state], feed_dict={previous_state: toFeed})
    with torch.no_grad():
        toFeed = torch.from_numpy(toFeed).float()
        observation, reward, done = env_net(toFeed)

    # print reward
    reward = reward.item()
    # print('reward', reward)
    observation = observation.cpu().numpy()
    observation[:, 0] = np.clip(observation[:, 0], -2.4, 2.4)
    observation[:, 2] = np.clip(observation[:, 2], -0.4, 0.4)
    doneP = torch.sigmoid(done).item()
    # print('doneP', doneP)

    if doneP > 0.1 or len(xs) >= 300:
        done = True
    else:
        done = False
    return observation, reward, done


# Training the Policy and Model
xs, drs, ys, ds = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
real_episodes = 1
batch_size = real_bs

drawFromModel = False  # When set to True, will use model for observations
trainTheModel = True  # Whether to train the model
trainThePolicy = False  # Whether to train the policy
switch_point = 1

# Launch the graph
rendering = False
observation = env.reset()
x = observation

gradBuffer = {}
for name, param in policy_net.named_parameters():
    if param.requires_grad:
        gradBuffer[name] = torch.zeros_like(param)

print("================================================================")
print("Gradbuffer contains params from policy and environment networks")
print(gradBuffer.keys())
print

while episode_number <= 500:
    # Start displaying environment once performance is acceptably high.
    if (reward_sum / batch_size > 150 and not drawFromModel) or rendering:
        env.render()
        rendering = True

    with torch.no_grad():
        x = np.reshape(observation, [1, 4])
        x = torch.from_numpy(x).float()
        # score = sess.run(probability, feed_dict={observations: x})
        _, prob = policy_net(x)
        # print 'prob', prob.item()
        action = 1 if np.random.uniform() < prob.item() else 0

    # record various intermediates (needed later for backprop)
    xs.append(x)
    y = 1 if action == 0 else 0
    ys.append(y)

    # step the  model or real environment and get new measurements
    if not drawFromModel:
        observation, reward, done, info = env.step(action)
    else:
        observation, reward, done = stepModel(xs, action)

    reward_sum += reward

    ds.append(done * 1)
    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:
        if not drawFromModel:
            real_episodes += 1
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        epy = np.vstack(ys)
        epr = np.vstack(drs)
        epd = np.vstack(ds)
        xs, drs, ys, ds = [], [], [], []  # reset array memory

        if trainTheModel:
            actions = np.array([np.abs(y - 1) for y in epy][:-1])
            state_prevs = epx[:-1, :]
            state_prevs = np.hstack([state_prevs, actions])
            state_nexts = epx[1:, :]
            rewards = np.array(epr[1:, :])
            dones = np.array(epd[1:, :])
            state_nextsAll = np.hstack([state_nexts, rewards, dones])

            state_prevs = torch.from_numpy(state_prevs).float()
            state_nexts = torch.from_numpy(state_nexts).float()
            true_done = torch.from_numpy(dones).float()
            true_reward = torch.from_numpy(rewards).float()

            # feed_dict = {previous_state: state_prevs, true_observation: state_nexts, true_done: dones,
            #              true_reward: rewards}

            predicted_observation, predicted_reward, predicted_done = env_net(state_prevs)
            loss_o = criterion_observation(predicted_observation, state_nexts)
            loss_d = criterion_done(predicted_done, true_done)
            loss_r = criterion_reward(predicted_reward, true_reward)
            model_loss = loss_o + loss_d + loss_r

            # loss, pState, _ = sess.run([model_loss, predicted_state, updateModel], feed_dict)
            optimizer_env.zero_grad()
            model_loss.backward()
            optimizer_env.step()

        if trainThePolicy:
            discounted_epr = discount_rewards(epr).astype('float32')
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            # tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})

            ob_exp = torch.from_numpy(epx).float()
            input_y = torch.from_numpy(epy).float()
            discounted_epr = torch.from_numpy(discounted_epr).float()

            score, prob = policy_net(ob_exp)
            loss = criterion_policy(prob, input_y, discounted_epr)
            # print loss
            policy_net.zero_grad()
            loss.backward()

            # for ix, grad in enumerate(tGrad):
            #     gradBuffer[ix] += grad
            for name, param in policy_net.named_parameters():
                if param.requires_grad:
                    # print(name, param.grad.data)
                    gradBuffer[name] += param.grad.data

        if switch_point + batch_size == episode_number:
            switch_point = episode_number
            if trainThePolicy:
                # sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})

                # manually set gradient of policy_net trainable params
                for name, param in policy_net.named_parameters():
                    if param.requires_grad:
                        param.grad = gradBuffer[name]
                optimizer_policy.step()
                gradBuffer = resetGradBuffer(gradBuffer)

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            if not drawFromModel:
                if real_episodes % 10 == 0:
                    print('World Perf: Episode %f. Reward %f. action: %f. mean reward %f.' % (
                        real_episodes, reward_sum / real_bs, action, running_reward / real_bs))
                if reward_sum / batch_size > 200:
                    break
            reward_sum = 0

            # Once the model has been trained on 100 episodes, we start alternating between training the policy
            # from the model and training the model from the real environment.
            if episode_number > 100:
                drawFromModel = not drawFromModel
                trainTheModel = not trainTheModel
                trainThePolicy = not trainThePolicy

        if drawFromModel:
            observation = np.random.uniform(-0.1, 0.1, [4])  # Generate reasonable starting point
            batch_size = model_bs
        else:
            observation = env.reset()
            batch_size = real_bs

print(real_episodes)

plt.figure(figsize=(8, 12))
pState = torch.cat([predicted_observation, predicted_reward, predicted_done], dim=1)
pState = pState.detach().cpu().numpy()

for i in range(6):
    plt.subplot(6, 2, 2 * i + 1)
    plt.plot(pState[:, i])
    plt.subplot(6, 2, 2 * i + 1)
    plt.plot(state_nextsAll[:, i])
plt.tight_layout()
plt.show()
