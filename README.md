# DRL-ProjectTwo-Continuous-Control
Udacity Nanodegree on DRL - Project 2

# Goal of the Project

In this project is asked to train a double-joined arm agent able to tackle a target the Unity's Reacher environment. As instructed a reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
The observation space consists of 33 variables corresponding to the position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
 
# Learning Algorithm
### Deep Deterministic Policy Gradient (DDPG)
Considering the goal of the project and the environment provided, the Agent will be trained using a DDPG algorithm which tackles exactly the needs of this problem of continuous control as stated in this paper

The implementation of the algorithm is based on the template provided by Udacity at https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum. A key factor of the implementation is the usage of an Actor-Critic Agent which enhances the estimation of an optimal policy by maximizing separately cumulative state-action values applying a gradient ascent approach.
In this project, the actor-critic agent has been implemented in the DDPG_Agent.py file, while the actor-critic models are in the model.py file.

As stated in the DDPG paper stated above, to balance continuous space exploitation, the Agent has been provided with a Ornstein-Uhlenbeck process which is responsible for adding noise according to the provided parameters: 

```python
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter
EPSILON = 1.0           # explore->exploit noise process added to act step
EPSILON_DECAY = 1e-6    # decay rate for noise process
```


