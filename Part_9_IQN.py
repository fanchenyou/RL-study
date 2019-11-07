import os
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
from timeit import default_timer as timer
from datetime import timedelta
from collections import deque
import torch
import torch.nn as nn

np.random.seed(1212)

env = gym.make('CartPole-v1')
env.seed(1212)

MINIBATCH_SIZE = 32
TRAIN_START = 1000
TARGET_UPDATE = 25
MEMORY_SIZE = 20000
INPUT = env.observation_space.shape[0]
OUTPUT = env.action_space.n
LEARNING_RATE = 0.0001
DISCOUNT = 0.99
LOAD_model = True
SAVE_model = True
TRAIN = False
RENDER = False
VIEW_DIST = False

path = "./CartPole_iqn_test"


class IQNAgent:
    def __init__(self, a_dim, s_dim,
                 num_tau=32,
                 num_tau_prime=8,
                 learning_rate=1e-3,
                 gamma=0.99,
                 batch_size=32,
                 buffer_size=10000,
                 target_update_step=25,
                 e_greedy=True,
                 e_step=1000,
                 eps_max=0.9,
                 eps_min=0.01,
                 eta=0.2,
                 gradient_norm=None,
                 view_dist=True
                 ):
        self.memory = []
        self.iter = 0
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = int(batch_size)
        self.buffer_size = buffer_size
        self.num_tau = num_tau
        self.num_tau_prime = num_tau_prime
        self.e_greedy = e_greedy
        self.e_step = e_step
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps = 0.0 if not e_greedy else self.eps_max
        self.target_update_step = target_update_step
        self.gradient_norm = gradient_norm
        self.view_dist = view_dist
        self.eta = eta

        # self.a_dim, self.s_dim, = a_dim, s_dim
        # self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        # self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        # self.A = tf.placeholder(tf.float32, [None, 1], 'a')
        # self.A_ = tf.placeholder(tf.float32, [None, 1], 'a')
        # self.T = tf.placeholder(tf.float32, [None, None], 'theta_t')
        # self.tau = tf.placeholder(tf.float32, [None, None], 'tau')
        # self.tau_ = tf.placeholder(tf.float32, [None, None], 'tau_')

        self.q_theta_eval_train, self.q_mean_eval_train, self.q_theta_eval_test, self.q_mean_eval_test = self._build_net(
            self.S, self.tau,
            scope='eval_params',
            trainable=True)
        self.q_theta_next_train, self.q_mean_next_train, _, _ = self._build_net(self.S_, self.tau_,
                                                                                scope='target_params',
                                                                                trainable=False)

        self.qe_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_params')
        self.qt_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_params')

        self.params_replace = [tf.assign(qt, qe) for qt, qe in zip(self.qt_params, self.qe_params)]

        a_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), tf.squeeze(tf.to_int32(self.A))], axis=1)
        self.q_theta_eval_a = tf.gather_nd(params=self.q_theta_eval_train, indices=a_indices)
        a_next_max_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32),
                                       tf.squeeze(tf.to_int32(self.A_))], axis=1)
        self.q_theta_next_a = tf.gather_nd(params=self.q_theta_next_train, indices=a_next_max_indices)



    def _build_net(self, s, tau, scope, trainable):

        s_tiled = tf.tile(s, [1, tf.shape(tau)[1]])
        s_reshaped = tf.reshape(s_tiled, [-1, self.s_dim])
        tau_reshaped = tf.reshape(tau, [-1, 1])

        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            pi_mtx = tf.constant(np.expand_dims(np.pi * np.arange(0, 64), axis=0), dtype=tf.float32)

            #net_psi = tf.layers.dense(s_reshaped, 16, activation=tf.nn.selu,
            #                          kernel_initializer=init_w, bias_initializer=init_b, name='psi',
            #                          trainable=trainable)
            cos_tau = tf.cos(tf.matmul(tau_reshaped, pi_mtx))
            net_phi = tf.layers.dense(cos_tau, 4, activation=tf.nn.relu,
                                      kernel_initializer=init_w, bias_initializer=init_b, name='phi',
                                      trainable=trainable)

            #joint_term = tf.multiply(net_psi, net_phi)
            #joint_term = tf.multiply(s_reshaped, net_phi)
            joint_term = s_reshaped+tf.multiply(s_reshaped, net_phi)

            q_net = tf.layers.dense(joint_term, 32, activation=tf.nn.selu,
                                    kernel_initializer=init_w, bias_initializer=init_b, name="layer1",
                                    trainable=trainable)

            q_net = tf.layers.dense(q_net, 32, activation=tf.nn.selu,
                                    kernel_initializer=init_w, bias_initializer=init_b, name="layer2",
                                    trainable=trainable)

            q_flat = tf.layers.dense(q_net, self.a_dim, activation=None,
                                     kernel_initializer=init_w, bias_initializer=init_b, name="theta",
                                     trainable=trainable)

            q_re_train = tf.transpose(tf.split(q_flat, self.batch_size, axis=0), perm=[0, 2, 1])

            q_re_test = tf.transpose(tf.split(q_flat, 1, axis=0), perm=[0, 2, 1])

            q_mean_train = tf.reduce_mean(q_re_train, axis=2)

            q_mean_test = tf.reduce_mean(q_re_test, axis=2)

        return q_re_train, q_mean_train, q_re_test, q_mean_test

    def update_target_net(self):
        self.sess.run(self.params_replace)

    def quantile_huber_loss(self):
        q_theta_expand = tf.tile(tf.expand_dims(self.q_theta_eval_a, axis=2), [1, 1, self.num_tau_prime])
        T_theta_expand = tf.tile(tf.expand_dims(self.T, axis=1), [1, self.num_tau_prime, 1])

        u_theta = T_theta_expand - q_theta_expand

        rho_u_tau = self._rho_tau(u_theta, tf.tile(tf.expand_dims(self.tau, axis=2), [1, 1, self.num_tau_prime]))

        qr_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(rho_u_tau, axis=2), axis=1))

        return qr_loss

    def memory_add(self, state, action, reward, next_state, done):
        self.memory += [(state, action, reward, next_state, done)]
        if len(self.memory) > self.buffer_size:
            self.memory = self.memory[-self.buffer_size:]

    def learn(self):
        if self.iter % self.target_update_step:
            self.update_target_net()

        minibatch = np.vstack(random.sample(self.memory, self.batch_size))
        bs = np.vstack(minibatch[:, 0])
        ba = np.vstack(minibatch[:, 1])
        br = np.vstack(minibatch[:, 2])
        bs_ = np.vstack(minibatch[:, 3])
        bd = np.vstack(minibatch[:, 4])

        tau = np.random.rand(self.batch_size, self.num_tau_prime)
        tau_ = np.random.rand(self.batch_size, self.num_tau_prime)
        tau_beta_ = self.conditional_value_at_risk(self.eta, np.random.rand(self.batch_size, self.num_tau))


        # TODO
        T_mean_K = self.sess.run(self.q_mean_next_train, feed_dict={self.S_: bs_, self.tau_: tau_beta_})
        ba_ = np.expand_dims(np.argmax(T_mean_K, axis=1), axis=1)

        T_theta_ = self.sess.run(self.q_theta_next_a, feed_dict={self.S_: bs_, self.A_: ba_, self.tau_: tau_})

        T_theta = br + (1 - bd) * self.gamma * T_theta_
        T_theta = T_theta.astype(np.float32)

        loss, _ = self.sess.run([self.loss, self.train_op], {self.S: bs, self.A: ba, self.T: T_theta, self.tau: tau})








        self.iter += 1

        if self.eps > self.eps_min:
            self.eps -= self.eps_max / self.e_step

        return loss

    @staticmethod
    def _rho_tau(u, tau, kappa=1):
        # delta = tf.cast(u < 0, 'float')
        delta = (u < 0).float()
        if kappa == 0:
            return (tau - delta) * u
        else:
            # return tf.abs(tau - delta) * tf.where(tf.abs(u) <= kappa, 0.5 * tf.square(u),
            #                                       kappa * (tf.abs(u) - kappa / 2))

            # https://pytorch.org/docs/stable/torch.html#torch.where
            return torch.abs(tau - delta) * torch.where(torch.abs(u) <= kappa,
                                                        0.5 * torch.mul(u, u),
                                                        kappa * (torch.abs(u) - kappa / 2))
    @staticmethod
    def conditional_value_at_risk(eta, tau):
        return eta * tau

    def choose_action(self, state):
        state = state[np.newaxis, :]
        tau_K = np.random.rand(1, self.num_tau)
        tau_beta = self.conditional_value_at_risk(self.eta, tau_K)
        if np.random.uniform() > self.eps:

            # TODO
            actions_value, q_dist = self.sess.run([self.q_mean_eval_test, self.q_theta_eval_test],
                                                  feed_dict={self.S: state, self.tau: tau_beta})
            action = np.argmax(actions_value)

        else:
            action = np.random.randint(0, self.a_dim)
            actions_value, q_dist = 0, 0

        return action, actions_value, q_dist, tau_beta



def plot_cdf(actions_value, q_dist, tau_beta):
    y = np.repeat(np.sort(tau_beta), 2, 0)
    plt.ylim([0, 1])
    plt.xlim([np.max(actions_value)-1, np.max(actions_value)+1])
    plt.xlabel('Q')
    plt.ylabel('CDF')
    plt.step(np.transpose(np.sort(q_dist[0])), np.transpose(y))
    plt.legend(labels=('Left', 'Right'))
    plt.draw()
    plt.pause(0.00001)
    plt.clf()


def train():
    IQNbrain = IQNAgent(OUTPUT, INPUT,
                        gamma=DISCOUNT,
                        batch_size=MINIBATCH_SIZE,
                        buffer_size=MEMORY_SIZE,
                        target_update_step=TARGET_UPDATE,
                        e_greedy=not LOAD_model,
                        e_step=1000,
                        )

    loss = self.quantile_huber_loss()



    optim = torch.optim.Adam(IQNbrain.parameters(), lr=LEARNING_RATE)


    all_rewards = []
    frame_rewards = []
    loss_list = []
    loss_frame = []
    recent_rlist = deque(maxlen=15)
    recent_rlist.append(0)
    episode, epoch, frame = 0, 0, 0
    start = timer()

    while np.mean(recent_rlist) < 499:
        episode += 1

        rall, count = 0, 0
        done = False
        s = env.reset()

        while not done:
            if RENDER:
                env.render()

            frame += 1
            count += 1

            action, actions_value, q_dist, tau_beta = IQNbrain.choose_action(s)
            if VIEW_DIST:
                plot_cdf(actions_value, q_dist, tau_beta)

            s_, r, done, l = env.step(action)

            if done and count >= 500:
                reward = 1
            elif done and count < 500:
                reward = -10
            else:
                reward = 0

            IQNbrain.memory_add(s, float(action), reward, s_, int(done))
            s = s_

            rall += r

            if frame > TRAIN_START and TRAIN:
                loss = IQNbrain.learn()
                loss_list.append(loss)
                loss_frame.append(frame)

        recent_rlist.append(rall)
        all_rewards.append(rall)
        frame_rewards.append(frame)

        print("Episode:{} | Frames:{} | Reward:{} | Recent reward:{}".format(episode, frame, rall,
                                                                                          np.mean(recent_rlist)))

    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join(path, 'IQN.ckpt')
    if SAVE_model:
        IQNbrain.save_model(ckpt_path)

    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.title('Episode %s. Recent_reward: %s. Time: %s' % (
        len(all_rewards), np.mean(recent_rlist), timedelta(seconds=int(timer() - start))))
    plt.plot(frame_rewards, all_rewards)
    plt.ylim(0, 510)
    plt.subplot(212)
    plt.title('Loss')
    plt.plot(loss_frame, loss_list)
    #plt.ylim(0, 20)
    plt.show()
    plt.close()

#
# def test():
#         IQNbrain = IQNAgent(sess, OUTPUT, INPUT,
#                             learning_rate=LEARNING_RATE,
#                             gamma=DISCOUNT,
#                             batch_size=MINIBATCH_SIZE,
#                             buffer_size=MEMORY_SIZE,
#                             target_update_step=TARGET_UPDATE,
#                             e_greedy=not LOAD_model,
#                             e_step=1000,
#                             gradient_norm=None,
#                             )
#
#         IQNbrain.load_model(tf.train.latest_checkpoint(path))
#
#         masspole_list = np.arange(0.01, 0.21, 0.025)
#         length_list = np.arange(0.5, 3, 0.25)
#
#         performance_mtx = np.zeros([masspole_list.shape[0], length_list.shape[0]])
#
#         for im in range(masspole_list.shape[0]):
#             for il in range(length_list.shape[0]):
#                 env.env.masspole = masspole_list[im]
#                 env.env.length = length_list[il]
#
#                 all_rewards = []
#
#                 for episode in range(5):
#
#                     rall, count = 0, 0
#                     done = False
#                     s = env.reset()
#
#                     while not done:
#                         if RENDER:
#                             env.render()
#
#                         action, actions_value, q_dist, tau_beta = IQNbrain.choose_action(s)
#
#                         s_, r, done, _ = env.step(action)
#
#                         s = s_
#
#                         rall += r
#
#                     all_rewards.append(rall)
#
#                     print("Episode:{} | Reward:{} ".format(episode, rall))
#
#                 performance_mtx[im, il] = np.mean(all_rewards)
#
#         fig, ax = plt.subplots()
#         ims = ax.imshow(performance_mtx, cmap=cm.gray, interpolation=None, vmin=0, vmax=500)
#         ax.set_xticks(np.arange(0, length_list.shape[0], length_list.shape[0] - 1))
#         ax.set_xticklabels(['0.5', '3'])
#         ax.set_yticks(np.arange(0, masspole_list.shape[0], masspole_list.shape[0] - 1))
#         ax.set_yticklabels(['0.01', '0.20'])
#         ax.set_xlabel('Pole length')
#         ax.set_ylabel('Pole mass')
#         ax.set_title('Robustness test: IQN')
#         fig.colorbar(ims, ax=ax)
#
#         plt.show()
#         plt.close()


def main():
    if TRAIN:
        train()
    else:
        test()


if __name__ == "__main__":
    main()