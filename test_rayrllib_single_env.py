import os
import re
import xml.etree.ElementTree as ET
import copy
import pandas as pd

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils import EzPickle, seeding

import numpy as np

from factory_single_agent import Factory
from rl_paper_env1 import RLEnv
import utilities as util
from ray.rllib.policy.policy import Policy
import ray
from ray import air, tune
# from ray.rllib.connectors.env_to_module import FlattenObservations
# from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
# from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
# from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.logger import UnifiedLogger
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env
from ray.tune.logger import JsonLoggerCallback, CSVLoggerCallback, TBXLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import warnings



warnings.filterwarnings("ignore", category=DeprecationWarning)



price_signal = [107.21, 101.1, 96.73, 82.34, 81.93, 84.03, 97.25, 106.51, 
                120.7, 116.86,112.85, 113.37, 115, 109.78, 119.71, 125.83, 
                139.57, 155.93, 163.5,146.74, 139.38, 131.8, 129.94,121.3,121.3]

# price_signal =  [25.65, 23.51, 22.07, 21.13, 20.96, 21.81, 23.61, 26.64, 31.22,
#                   36.63, 46.57 , 50.51, 55.06, 67.33, 77.31, 93.53, 108.87, 114.03,
#                     81.57, 59.12, 52.01, 48.01, 36.79, 31.23]

pv_signal = [0.0,0.0,0.0,0.0,0.0,0.0,0.9,93.25,374.775,611.95,789.875,927.025,
             942.875,865.15,791.05,615.9,388.025,116.575,5.925,0.0,0.0,0.0,0.0,0.0,0.0]

# Load and parse the XML file
tree = ET.parse('/Users/sakshisharma/Desktop/rl_scheduling/steel_manufacturing_configuration.xml')
root = tree.getroot()

alpha_value = 0.7

factory = Factory(root,price_signal,pv_signal)

single_agent_env = RLEnv(factory,alpha_value)


# Register the single-agent environment
register_env(
    "SingleAgentEnvironment",
    lambda _: single_agent_env
)

# Define PPO configuration for a single-agent setup
ppo_config = (
    PPOConfig()
    .environment("SingleAgentEnvironment")  # Use the registered environment name
    .env_runners(
        num_env_runners=1,  # Single environment
        num_envs_per_env_runner=1,
        batch_mode="complete_episodes"
    )
    .framework("torch")  # Use PyTorch
    .training(
        train_batch_size=240,  # Example setting
        minibatch_size=48,
        gamma=0.2,
        lr=5e-4,
        use_critic=True,
        use_gae=True,
        lambda_=0.5,
        entropy_coeff_schedule=[
            (0, 0.1), 
            (30000, 0.07), 
            (50000, 0.05), 
            (100000, 0.01)
        ],
        vf_loss_coeff=2.0,
        grad_clip=0.5,
    )
)


# /Users/sakshisharma/Desktop/rl_scheduling/tensorboard_logs/PPO_Training_Experiment/PPO_MRLEnvironment_d16c2_00000_0_2024-11-14_14-52-05
# Load the checkpoint path from the training output directory
# checkpoint_dir = "/Users/sakshisharma/Desktop/rl_scheduling/tensorboard_logs_env1/PPO_Training_Experiment/PPO_MRLEnvironment_11acf_00000_0_2025-01-03_19-32-21/checkpoint_000071"  # Update this path to match your saved experiment
checkpoint_dir = "/Users/sakshisharma/Desktop/rl_scheduling/tensorboard_logs/single_agent/PPO_Training_Experiment/PPO_SingleAgentEnvironment_03000_00000_0_2025-02-05_13-30-41/checkpoint_000199" 
checkpoint_path = "file://" + os.path.abspath(checkpoint_dir)
# Restore the model from the checkpoint
policy = Policy.from_checkpoint(checkpoint_path)
print(policy)

for episode in range(1):  # You can change the number of episodes for testing
    print(f"Testing Episode: {episode + 1}")
    
    obs,info = single_agent_env.reset()  # Reset the environment at the beginning of the episode
    done = False
    # episode_rewards = {"Primary": 0, "Auxillary": 0}

    rewards = {} 
    container_states = {}
    actions = {}
    
    
    while not done:
        # Get actions from the trained model

        action = policy['default_policy'].compute_single_action(obs)[0]
            

        actions[single_agent_env.factory.current_timestep] = action

        # Step the environment    
        next_obs, reward, done, _ , infos = single_agent_env.step(action)

        # render the environment state
        print('factory timestep',single_agent_env.factory.current_timestep)
        rewards[single_agent_env.factory.current_timestep] = reward
       



        v = single_agent_env.render()
        container_states[factory.current_timestep] = copy.deepcopy(v)
       
    

        # Update observations
        obs = next_obs


    # Print final episode reward
    print(f"Episode {episode + 1} finished.")

 
    

   

# Shutdown Ray after testing is done
ray.shutdown()


action_df = pd.DataFrame.from_dict(actions, orient='index')
action_df.to_csv('actions_test.csv')



util.compute_and_save_total_cost(factory.consumption_dict,[price/1000 for price in price_signal[:-1]],output_filename="total_cost_singleenv.csv")

util.plot_stacked_bar(factory.consumption_dict,price_signal[:-1], pv_signal = pv_signal[:-1])
util.plot_storage_charging_discharging(factory.energy_storage_dict,price_signal[:-1],pv_signal = pv_signal[:-1])
util.compute_and_plot_energy_cost(factory.consumption_dict,price_signal[:-1])
util.plot_total_energy_cost(factory.consumption_dict,price_signal[:-1])

util.plot_container_value_over_time(container_states,'Steelpowder')

util.plot_container_comparison_over_time(container_states,'Coolwater','Nitrogen')

util.plot_3d_heatmap_over_time(container_states)

# util.plot_reward_distribution_individual(individual_rewards)

# util.plot_cumulative_rewards(cumulative_rewards)




