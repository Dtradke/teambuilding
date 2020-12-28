
import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from del_environment import SampleEnvironment
from agency import PPOAgent

import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

#
# arm_totals = np.zeros((3,))
# pulls = np.zeros((3,))
# state = np.zeros((3,))
# arm_means = np.random.normal(loc=0, scale=1.0, size=(3))
#
# for i in range(500):
#     for j in range(arm_means.shape[0]):
#         sample = np.random.normal(arm_means[j], scale=1.0)
#         arm_totals[j] += sample
#         pulls[j]+=1
#         state[j] = arm_totals[j] / pulls[j]
#
# print(arm_means)
# print(state)
# exit()




tf.compat.v1.enable_v2_behavior()

tf.version.VERSION



# class Agent(object):
#     def __init__(self, id):
#         self._id = id



"""## Hyperparameters"""

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

# amount of data sampled from the replay buffer to compute the gradient
batch_size = 30  # @param {type:"integer"}

learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

# target network is updated every "target_update_period" iterations
target_update_period = 10 # @param {type:"integer"}

temp_rewards_arr = []
arms = 3
agent_amt = 3

env = SampleEnvironment(arms)


agent = PPOAgent(env)

# agent.Actor.summary()
# exit()

EPISODES = 100
LEN_OF_EPISODE = 300
# def run(self):
for e in range(EPISODES):
    # state = self.reset(self.env)
    state = env.reset()
    agent.reset()
    done, score, SAVING = False, 0, ''
    # Instantiate or reset games memory
    # states, actions, rewards, predictions = [], [], [], []
    # while not done:
    for i in range(LEN_OF_EPISODE):
        # if i == 30: exit()
        #self.env.render()
        # Actor picks an action
        action, prediction = agent.act(agent._state)
        # Retrieve new state, reward, and whether the state is terminal
        next_state, reward, done = env.step(action, agent)
        # Memorize (state, action, reward) for training
        agent.states.append(agent._state)
        action_onehot = np.zeros([env.action_space])
        action_onehot[action] = 1
        agent.actions.append(action_onehot)
        agent.rewards.append(reward)
        agent.predictions.append(prediction)
        # Update current state
        agent._state = next_state
        agent.score += reward
        # if i == (LEN_OF_EPISODE-1): done = 1
        # if done == 1:
    # average = score / LEN_OF_EPISODE
    average = agent.PlotModel(e)
    # saving best models
    if average >= agent.max_average:
        agent.max_average = average
        agent.save()
        SAVING = "SAVING"
    else:
        SAVING = ""
    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, EPISODES, agent.score, average, SAVING))
    diff = env._arm_means - agent._state
    print("diff: ", diff, " = ", np.sum(np.absolute(diff)))
    print("perc: ", agent.pulls / LEN_OF_EPISODE)
    print()

    # agent.replay(states, actions, rewards, predictions)
    agent.replay()

# self.env.close()
