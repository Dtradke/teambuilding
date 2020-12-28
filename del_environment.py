import abc
import tensorflow as tf
import numpy as np
import copy

# from tf_agents.environments import py_environment
# from tf_agents.environments import tf_environment
# from tf_agents.environments import tf_py_environment
# from tf_agents.environments import utils
# from tf_agents.specs import array_spec
# from tf_agents.environments import wrappers
# from tf_agents.environments import suite_gym
# from tf_agents.trajectories import time_step as ts

# tf.compat.v1.enable_v2_behavior()



class SampleEnvironment(object): #py_environment.PyEnvironment
    def __init__(self, arms=3, mean=0, sd=1, max_timesteps=300):
        self.name = "normal_temporal"
        self._arms = arms
        self._mean = mean
        self._sd = sd
        self.action_space = arms
        self.state_space = np.zeros(self._arms,)
        # self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=arms, name='action')
        # self._observation_spec = array_spec.BoundedArraySpec(shape=(arms,), dtype=np.float32, name='observation')
        self._state = np.random.normal(loc=self._mean, scale=self._sd, size=(self._arms)) #np.zeros(self._arms,)
        self._episode_ended = False
        # self._pulls = np.zeros(self._arms,)
        # self._arm_totals = np.zeros(self._arms,)
        self._arm_means = np.random.normal(loc=self._mean, scale=self._sd, size=(self._arms)) #this should be an (arms,) array of arm means
        self._timestep = 0
        self.last_diff = 0
        self._max_timesteps = max_timesteps
        self._current_time_step = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def reset(self):
        self._state = np.random.normal(loc=self._mean, scale=self._sd, size=(self._arms)) #np.zeros((self._arms,))
        self._episode_ended = False
        self._arm_means = np.random.normal(loc=self._mean, scale=self._sd, size=(self._arms))
        self._pulls = np.zeros(self._arms,)
        self._arm_totals = np.zeros(self._arms,)
        self.last_diff = 0
        # print(">>>>>>>>>>1", ts.restart(np.array(self._state, dtype=np.float32)))
        # return ts.restart(np.array(self._state, dtype=np.float32))
        return self._state

    def step(self, action, agent):
        state = copy.deepcopy(agent._state)
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        if action == self._arms+1: # THIS WILL NOT HAPPEN RN                    # episode ends when agent takes max action (self._arms) or max time
            self._episode_ended = True
            done = 1
        else:
            sample = np.random.normal(self._arm_means[action], scale=self._sd)
            agent.arm_totals[action] += sample
            agent.pulls[action]+=1
            state[action] = agent.arm_totals[action] / agent.pulls[action]
            done = 0

        # cur_diff = tfp.distributions.kl_divergence(self._arm_means, self._state)
        # cur_diff = np.sum(np.absolute(self._arm_means - agent._state))         #FIXME: this prob doesnt work - some samples bring mean further away
        # r = self.last_diff - cur_diff
        # print("last: ", round(self.last_diff,4), " - cur: ", round(cur_diff,4), " r: ", round(r,4))
        # self.last_diff = cur_diff


        r = 0
        diff = np.absolute(self._arm_means - state)
        # r = 1 / np.sum(diff)
        for i in diff:
            r += 1 / i      # this relies on i never being the true mean... this could break
        r = r / self._arms
        # print(r)

        # r = 0
        # diff = np.absolute(self._arm_means - self._state)
        # for i in diff:
        #     r += 1 / i      # this relies on i never being the true mean... this could break
        # r = r / self._arms
        # reward = np.absolute(self.last_r_tot - r)
        # self.last_r_tot = r
        # print(r)





        return state, r, done
