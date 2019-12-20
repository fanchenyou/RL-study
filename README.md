## Several Reinforcement models
![rl_methods](/pics/rl.png)
#### 0. Q-tables, [ref](https://github.com/awjuliani/DeepRL-Agents/blob/master/Q-Table.ipynb), [code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter01/tic_tac_toe.py)
    * Learn Q-tables with policy gradient
    * Add tic_tac_toe examples
 
#### 1. MDP, [code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter03/grid_world.py)
    * MDP assumes a finite number of states (env) and actions (agent). 
    * Agent observes a state and executes an action, which incurs intermediate costs to be minimized.
    * The goal is to maximize the (expected) accumulated rewards.
    * Use Bellman equation to update each state value.
    
#### 2. Bandit problem, [ref](https://github.com/awjuliani/DeepRL-Agents/blob/master/Contextual-Policy.ipynb)
    * Bandit problem
    * Train a neural network to learn a policy for picking actions using feedback from the environment
    * Use policy gradients to adjust NN's weights through gradient descent
    * Re-written in Pytorch 1.2

#### 3. Modeling environment, [ref](https://github.com/awjuliani/DeepRL-Agents/blob/master/Model-Network.ipynb), [tutorial](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-3-model-based-rl-9a6fe0cce99)
    * Add separate network to model physical environment
    * Use policy gradients to adjust NN's weights through gradient descent
    * Re-written in Pytorch 1.2
    
#### 4. DQN, [ref](https://github.com/seungeunrho/minimalRL/blob/master/a2c.py)
    * Deep Q Network
    * Use separate networks for 1) updating weights and 2) generate current decisions
    * Use replay memory to sample from so that the training process is stable
    * Use epsilon-greedy to sample actions from predicted distribution

#### 5. A2C, [ref](https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/405_DQN_Reinforcement_learning.py), [tutorial](https://www.freecodecamp.org/news/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d/)
    * Advantage Actor-Critic RL
    * Train Actor and Critic networks
    * Define worker function which has independent gym environment, and simulates CartPole
    * Creates multiple processes for workers to update networks
    
#### 6. Continuous A3C, [ref](https://github.com/MorvanZhou/pytorch-A3C/blob/master/continuous_A3C.py), [tutorial](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
    * Continuous Asynchronized Actor Critic

#### 7. Discrete A3C, [ref](https://github.com/MorvanZhou/pytorch-A3C/blob/master/discrete_A3C.py)
    * Discrete Asynchronized Actor Critic
    
#### 8. Quantile-DQN, [paper](https://arxiv.org/pdf/1710.10044.pdf), [ref](https://github.com/senya-ashukha/quantile-regression-dqn-pytorch)
    * Distributional Reinforcement Learning with Quantile Regression
    * Add NN and CNN examples
    
#### 9. Implicit-Quantile-DQN (IQN), [paper](https://arxiv.org/pdf/1806.06923.pdf), [ref](https://github.com/sjYoondeltar/myRL_example)
  ![Network](/pics/iqn.png)
  
    * Implicit Quantile Networks for Distributional Reinforcement Learning
    * Add NN and CNN examples
    
#### 10. Deep Deterministic Policy Gradient (DDPG), [tutorial](https://spinningup.openai.com/en/latest/algorithms/ddpg.html), [ref](https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py)  
    * Concurrently learn a Q-function and a policy. 
    * DDPG interleaves learning an approximator to Q(s,a) with learning an approximator to a(s).
    * DDPG explores action space by noise at training time.