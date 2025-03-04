import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils import EzPickle, seeding

import numpy as np



class RLEnv(gym.Env):

    def __init__(self, factory, alpha_value,render_mode='human'):
  
        self.factory = factory
        self.alpha_value = alpha_value
        self.max_time_step = 24  
        self.num_resources = self.factory.num_resources
        self.num_equipment_agent = self.factory.num_equipment_agent
        self.num_ess = self.factory.num_ess
        self.possible_agents = self.factory.equipment_agents + self.factory.energy_storage_name_list
        self.primary_agents = self.factory.primary_agents
        self.auxillary_agents = self.factory.auxillary_agents
        self.ess_agents = self.factory.energy_storage_name_list
        self.render_mode=render_mode

        self.observation_space = spaces.Box(low=np.zeros((self.num_resources + self.num_ess + 2,)), high=np.iinfo(np.int32).max, shape=(self.num_resources + self.num_ess + 2,), dtype=np.float32)

        self.action_space = spaces.MultiDiscrete(np.array([2, 2, 2, 3, 3, 2, 3, 3, 3, 3, 5,3]))

    def reset(self, *, seed=None, options=None):

        self.factory.reset_env()
        observations = self.factory.get_new_state()
        infos = {}
        return observations, infos

    def step(self, action):

        observation, production_rewards, consumption_rewards = self.factory.run(action)

        reward = sum([(1-self.alpha_value) * consumption_rewards[a] + self.alpha_value * production_rewards[a] for a in self.possible_agents]) 

        terminated = self.factory.current_timestep >= self.max_time_step

        info = {}

        return observation, reward, terminated, False, info

    def render(self, mode='human', close=False):
        if self.render_mode == 'human':
          print(f'Time Step #{self.factory.current_timestep}')
          values = self.factory.print_table()
          return values