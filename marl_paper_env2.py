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

    def __init__(self,factory):

        EzPickle.__init__(self,factory)
    
        self.factory = factory
        self.max_time_step = 23

        
        self.num_resources = self.factory.num_resources
        self.num_equipment_agent = self.factory.num_equipment_agent
        self.possible_agents = ['primary','auxillary']
        self.render_mode = None


    def reset(self, seed=None, options=None):
     
        self.agents = copy(self.possible_agents)
        self.factory.reset_env()

        self.ob = np.zeros(self.num_resources + 3, dtype=np.float32)
     
        observations = {
            a: self.ob
            for a in self.agents
        }

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):

        self.agents = copy(self.possible_agents)

        next_state,production_rewards,consumption_rewards = self.factory.run(actions)        

        #rewards
      
        rewards = {a: (0.5)*consumption_rewards[a] + (0.5)*production_rewards[a] for a in self.agents}

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        
        if self.factory.current_timestep >= self.max_time_step:
            truncations = {a: True for a in self.agents}
            terminations = {a: True for a in self.agents}

        
       
        # Get observations
        observations = {
            a : next_state
            for a in self.agents
        }


        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""
        self.factory.print_table()

    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        primary_space = Box(low=np.zeros((self.num_resources+ 3,)), 
                            high=np.iinfo(np.int32).max, 
                            shape=(self.num_resources+ 3,), 
                            dtype=np.float32)
        observation_dict = {'primary':primary_space,'auxillary':primary_space}
        return observation_dict[agent]
   
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        primary_actions = MultiDiscrete([2,2,2,3,3,2,3,3,3])
        auxillary_actions = MultiDiscrete([3,5])
        action_dict = {'primary':primary_actions,'auxillary':auxillary_actions}
        return action_dict[agent]