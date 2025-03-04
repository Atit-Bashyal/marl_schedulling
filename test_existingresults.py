import os
import re
import xml.etree.ElementTree as ET

from factory_individual_agents import Factory
from marl_paper_env1 import MRLEnvironment
import utilities  as util

import copy
import csv
import numpy as np


from ray.rllib.policy.policy import Policy

import ray
import matplotlib.pyplot as plt
import ray
from ray import air, tune
from ray.tune.registry import get_trainable_cls, register_env
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.logger import pretty_print
import pandas as pd 

price_signal = [107.21, 101.1, 96.73, 82.34, 81.93, 84.03, 97.25, 106.51, 
                120.7, 116.86,112.85, 113.37, 115, 109.78, 119.71, 125.83, 
                139.57, 155.93, 163.5,146.74, 139.38, 131.8, 129.94, 121.3,121.3]

# price_signal =  [25.65, 23.51, 22.07, 21.13, 20.96, 21.81, 23.61, 26.64, 31.22,
#                   36.63, 46.57 , 50.51, 55.06, 67.33, 77.31, 93.53, 108.87, 114.03,
#                     81.57, 59.12, 52.01, 48.01, 36.79, 31.23,31.23]

pv_signal = [0.0,0.0,0.0,0.0,0.0,0.0,0.9,93.25,374.775,611.95,789.875,927.025,
             942.875,865.15,791.05,615.9,388.025,116.575,5.925,0.0,0.0,0.0,0.0,0.0,0.0]

# Load and parse the XML file
tree = ET.parse('/Users/sakshisharma/Desktop/rl_scheduling/steel_manufacturing_configuration.xml')
root = tree.getroot()

factory = Factory(root,price_signal,pv_signal)

marl_env = MRLEnvironment(factory,0.2)

agents_ = factory.equipment_agents + factory.energy_storage_name_list


policies_ = {name: (None, marl_env.observation_space(name), marl_env.action_space(name), {}) for name in agents_}

ray.shutdown()
# Initialize Ray
ray.init(ignore_reinit_error=True)


# Register the environment
register_env(
    "MRLEnvironment",
    lambda _: ParallelPettingZooEnv(marl_env),
)


# Use the adjusted operating points of single agent
# operating_points = util.convert_csv_to_list('/Users/sakshisharma/Desktop/rl_scheduling/actions_test_multiagent(paper).csv')
operating_points = util.convert_csv_to_list2('/Users/sakshisharma/Desktop/rl_scheduling/actions_test.csv')
adjusted_operating_points = np.array(operating_points)
# Testing loop with predefined actions
for episode in range(1):  # Test for one episode
    print(f"Testing Episode: {episode + 1}")
    
    obs, info = marl_env.reset()  # Reset the environment
    done = {a: False for a in marl_env.agents}


    individual_rewards = {} 
    cumulative_rewards = {}
    container_states = {}
    experience = []
    # Iterate over timesteps based on the length of the adjusted actions
    for timestep in range(adjusted_operating_points.shape[0]):
        if all(done.values()):
            break

        # Use predefined actions from the table for this timestep
        action = {}
        current_obs =  marl_env.factory.get_new_state()
        for agent_idx, agent in enumerate(agents_):
            action[agent] = adjusted_operating_points[timestep][agent_idx]

                

        # Step the environment
        next_obs, rewards, terminations, truncations, infos = marl_env.step(action)

        experience.append((current_obs,action,rewards,next_obs,terminations, truncations, infos))


        # Render and log environment state
        print('Timestep:', marl_env.factory.current_timestep)
        individual_rewards[timestep] = [(key, value) for key, value in rewards.items()]
        cumulative_rewards[timestep] = sum([value for key, value in rewards.items()])
       

        v = marl_env.render()
        container_states[factory.current_timestep] = copy.deepcopy(v)

        # Update observations
        obs = next_obs

        # Check for termination or truncation
        done = {k: terminations[k] or truncations[k] for k in marl_env.agents}

    print(f"Episode {episode + 1} finished.")


util.compute_and_save_total_cost(factory.consumption_dict,[price/1000 for price in price_signal[:-1]],output_filename="total_cost_singleenv.csv")

util.plot_stacked_bar(factory.consumption_dict,price_signal[:-1])
util.plot_storage_charging_discharging(factory.energy_storage_dict,price_signal[:-1])
util.plot_2d_heatmap_over_time(container_states)

# util.plot_reward_distribution_individual(individual_rewards)

# util.plot_cumulative_rewards(cumulative_rewards)
