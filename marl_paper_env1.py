import xml.etree.ElementTree as ET
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import pandas as pd
import numpy as np
from prettytable import PrettyTable
import functools
import random
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete , Box
from pettingzoo import ParallelEnv

import numpy as np
import os
import ast
from gymnasium.utils import EzPickle, seeding


class MRLEnvironment(ParallelEnv,EzPickle):

    metadata = {
        "name": "steel_environment_v0"
    }

    def __init__(self,factory,alpha_value):

        EzPickle.__init__(self,factory,alpha_value)
    
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
        self.render_mode = None


    def reset(self, seed=None, options=None):
     
        self.agents = copy(self.possible_agents)
        self.factory.reset_env()
        observations = self.factory.get_new_state()
        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos
    

    def step(self, actions):

        self.agents = copy(self.possible_agents)

        observations,production_rewards,consumption_rewards = self.factory.run(actions)       

        # rewards = {a : 0.3*consumption_rewards[a] + 0.7*production_rewards[a] for a in self.agents}
      
        global_reward = sum([(1-self.alpha_value) * consumption_rewards[a] + self.alpha_value * production_rewards[a] for a in self.agents])
        rewards = {a:global_reward for a in self.agents}

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        infos = {a: {} for a in self.agents}

        if self.factory.current_timestep >= self.max_time_step:
            truncations = {a: True for a in self.agents}
            terminations = {a: True for a in self.agents}
        

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""
        values = self.factory.print_table()

        return values

    # If your spaces change over time, remove this line (disable caching).
    # @functools.lru_cache(maxsize=None)
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):

        observation_dict = {}

        for equipment in self.primary_agents:
            observation_dict[equipment]= Box(low=np.zeros((2 + self.num_ess + 2,)), 
                            high=np.iinfo(np.int32).max, 
                            shape=(2+ self.num_ess  + 2,), 
                            dtype=np.float32)
        for equipment in self.auxillary_agents:
            observation_dict[equipment]= Box(low=np.zeros((1+ self.num_ess + 2,)), 
                            high=np.iinfo(np.int32).max, 
                            shape=(1+ self.num_ess  + 2,), 
                            dtype=np.float32)
        for ess in self.ess_agents:
            observation_dict[ess]= Box(low=np.zeros((self.num_ess + 2,)), 
                            high=np.iinfo(np.int32).max, 
                            shape=(self.num_ess  + 2,), 
                            dtype=np.float32)
            
        return observation_dict[agent]
   
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        spaces_def = self.factory.set_action_spaces()
        action_dict = { a : spaces_def[a] for a in self.possible_agents }
        return action_dict[agent]