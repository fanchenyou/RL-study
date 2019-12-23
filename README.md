## Several Reinforcement models
![rl_methods](/pics/rl.png)
#### 0. Q-tables, [ref](https://github.com/awjuliani/DeepRL-Agents/blob/master/Q-Table.ipynb), [code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter01/tic_tac_toe.py)
    * Learn Q-tables with policy gradient
    * Bandit, tic_tac_toe examples
 
#### 1.1 Markov Decision Process, [code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter03/grid_world.py)
    * MDP assumes a finite number of states (env) and actions (agent). 
    * Agent observes a state and executes an action, which incurs intermediate costs to be minimized.
    * The goal is to maximize the (expected) accumulated rewards.
    * Use Bellman equation to update each state value.
    
#### 1.2 Dynamic Programming, [code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/tree/master/chapter04), [tutorial](https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/)    
    * Dynamic programming algorithms solve MDP as planning problems. 
    * Need to know entire env, can start at any state and learn next state.
    * Given a MDP, find an optimal policy for the agent to follow. It contains two main steps:
    * a. Break the problem into subproblems and solve it
    * b. Find overall optimal solution to the problem at hand
    * Usually contains 1) policy iteration 2) value iteration
    
#### 1.3 Monte Carlo, [code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter05/blackjack.py), [tutorial](https://oneraynyday.github.io/ml/2018/05/24/Reinforcement-Learning-Monte-Carlo/)    
    * MC samples from the experience to estimate the env.
    * MC can also start at any state, but need take all possible actions at every start of an episode.
    * MC learns episode-by-episode.
    * Use Importance Sampling to learn the sampling strategy.
    * MC needs to wait until the final reward before any state-action pair values can be updated.
    * Once the final reward was received, the path taken to reach the final state would need to be traced back and each value updated accordingly.

#### 2. Temporal Difference, [code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/tree/master/chapter06), [tutorial](https://www.cse.unsw.edu.au/~cs9417ml/RL1/tdlearning.html)
    * Temporal Difference (TD) Learning is used to estimate value functions. 
    * Unlike MC, TD estimates the final reward at each state and the state-action value updates for every step. 
    * TD is a combination of DP and MC.
    * On-Policy v.s. Off-Policy Learning:
    * On-Policy TD can learn the value of the policy that is used to make decisions. 
    * Off-Policy TD can learn different policies for behaviour and estimation. 
    * Behaviour policy is usually "soft" so there is sufficient exploration going on.
    * TD(0) - learn state-value function v(s)
    # Sarsa(On-policy) - learn an action-value function q(a,s)
    # Q-learning(Off-policy) - learn action-value regardless of selected action
    # Act-critic - critic is mearsuing V(s), actor is an independent policy. Use actor to choose action, and use TD to update critic. 
    
#### 3. Modeling environment, [ref](https://github.com/awjuliani/DeepRL-Agents/blob/master/Model-Network.ipynb), [tutorial](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-3-model-based-rl-9a6fe0cce99)
    * Add separate network to model physical environment
    * Use policy gradients to adjust NN's weights through gradient descent
    * Re-written in Pytorch 1.2
    
#### 4. Q-learning
    * Dyna-Q
    *   a) Integrating planning, acting, and learning
    * DQN
    *   a) Use separate networks for 1) updating weights and 2) generate current decisions
    *   b) Use replay memory to sample from so that the training process is stable
    *   c) Use epsilon-greedy to sample actions from predicted distribution

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