import os
import re
import xml.etree.ElementTree as ET

from factory_individual_agents import Factory
from marl_paper_env1 import MRLEnvironment
import utilities  as util

import copy

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

marl_env = MRLEnvironment(factory,0.8)

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

# Define the PPO configuration
ppo_config = (
    PPOConfig()
    .environment("MRLEnvironment")  # Use the registered environment name
    .env_runners(
        num_env_runners=4,  # Replaces num_rollout_workers
        num_envs_per_env_runner=2,  # Replaces num_envs_per_worker
        batch_mode="complete_episodes",  # Ensure full episodes are processed
        rollout_fragment_length='auto',  # Length of an episode
    )
    .framework("torch")  # Use PyTorch
    .multi_agent(
        policies=policies_,
        policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id)  # Maps agent IDs directly to policies
    
    .training(
        model = {'use_attention': True,
          # Coefficient for value function loss
        'vf_share_layers':True,},
        train_batch_size=240,  # Collect data from 10 episodes per iteration (24 * 10)
        minibatch_size=48,
        gamma=0.2,
        use_critic=True,  # Use a value function for advantage estimation
        use_gae=True,  # Generalized Advantage Estimation
        lambda_=0.5,  # GAE parameter for bias-variance trade-off
        entropy_coeff_schedule=[
            (0, 0.1),       # Start with high exploration
            (50000, 0.1),   # Keep high entropy until iteration 50
            (75000, 0.05),  # Start decreasing entropy around iteration 75
            (100000, 0.01), # Reach low entropy by iteration 100
        ],  # Encourage exploration with lower entropy cost
        vf_loss_coeff=2.0,  # Coefficient for value function loss
        vf_share_layers=True,  # Separate layers for value function
        grad_clip=0.5,  # Clip gradients to stabilize training
    )
)

# /Users/sakshisharma/Desktop/rl_scheduling/tensorboard_logs/PPO_Training_Experiment/PPO_MRLEnvironment_d16c2_00000_0_2024-11-14_14-52-05
# Load the checkpoint path from the training output directory
# checkpoint_dir = "/Users/sakshisharma/Desktop/rl_scheduling/tensorboard_logs_env1/PPO_Training_Experiment/PPO_MRLEnvironment_11acf_00000_0_2025-01-03_19-32-21/checkpoint_000071"  # Update this path to match your saved experiment
checkpoint_dir = "/Users/sakshisharma/Desktop/rl_scheduling/tensorboard_logs/env1/PPO_Training_Experiment/PPO_MRLEnvironment_52ab2_00000_0_2025-02-05_14-23-01/checkpoint_000127" 
checkpoint_path = "file://" + os.path.abspath(checkpoint_dir)
# Restore the model from the checkpoint
policy = Policy.from_checkpoint(checkpoint_path)


tag = "ray/tune/env_runners/hist_stats/episode_reward"  # The tag you're interested in

# bin_edges, bin_values = util.extract_histogram_from_tensorboard(checkpoint_dir, tag)
# util.plot_histogram(bin_edges, bin_values)

# Run the environment with the restored model
# Test the model for one episode or multiple episodes



for episode in range(1):  # You can change the number of episodes for testing
    print(f"Testing Episode: {episode + 1}")
    
    obs,info = marl_env.reset()  # Reset the environment at the beginning of the episode
    done = {a: False for a in marl_env.agents}
    # episode_rewards = {"Primary": 0, "Auxillary": 0}

    individual_rewards = {} 
    cumulative_rewards = {}
    container_states = {}
    actions = {}
    
    
    while not all(done.values()):
        # Get actions from the trained model
        action = {}
        
        for agent in marl_env.agents:
            action[agent] = policy[agent].compute_single_action(obs[agent])[0]
            

        actions[marl_env.factory.current_timestep] = action

        # Step the environment    
        next_obs, rewards, terminations, truncations, infos = marl_env.step(action)

        # render the environment state
        print('factory timestep',marl_env.factory.current_timestep)
        individual_rewards[marl_env.factory.current_timestep] = [(key, value) for key, value in rewards.items()]
        cumulative_rewards[marl_env.factory.current_timestep] = sum([value for key, value in rewards.items()])
       



        v = marl_env.render()
        container_states[factory.current_timestep] = copy.deepcopy(v)
       
        
       


        # Update observations
        obs = next_obs

        # Check for termination or truncation
        done = {k: terminations[k] or truncations[k] for k in marl_env.agents}

    # Print final episode reward
    print(f"Episode {episode + 1} finished.")

 
    

   

# Shutdown Ray after testing is done
ray.shutdown()


action_df = pd.DataFrame.from_dict(actions, orient='index')
action_df.to_csv('actions_test_multiagent.csv')

util.compute_and_save_total_cost(factory.consumption_dict,[price/1000 for price in price_signal[:-1]],output_filename="total_cost_env1.csv")

util.plot_stacked_bar(factory.consumption_dict,price_signal[:-1], pv_signal = pv_signal[:-1])
util.plot_storage_charging_discharging(factory.energy_storage_dict,price_signal[:-1],pv_signal = pv_signal[:-1])
util.compute_and_plot_energy_cost(factory.consumption_dict,price_signal[:-1])
util.plot_total_energy_cost(factory.consumption_dict,price_signal[:-1])

util.plot_container_value_over_time(container_states,'Steelpowder')

util.plot_container_comparison_over_time(container_states,'Coolwater','Nitrogen')

util.plot_3d_heatmap_over_time(container_states)

util.plot_reward_distribution_individual(individual_rewards)

util.plot_cumulative_rewards(cumulative_rewards)

# util.plot_value_vs_step_mean('/Users/sakshisharma/Desktop/rl_scheduling/0.8_mean.csv','/Users/sakshisharma/Desktop/rl_scheduling/0.8_max.csv','/Users/sakshisharma/Desktop/rl_scheduling/0.8_min.csv')


