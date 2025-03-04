import os
import re
import xml.etree.ElementTree as ET

from factory_individual_agents import Factory
from marl_paper_env1 import MRLEnvironment

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

# alpha_value_list = [0.2,0.4,0.5,0.6,0.8]


alpha_value = 0.8

# for alpha_value in alpha_value_list:

factory = Factory(root,price_signal,pv_signal)

marl_env = MRLEnvironment(factory,alpha_value)

agents_ = factory.primary_agents + factory.auxillary_agents + factory.energy_storage_name_list
policies_ = {name: (None, marl_env.observation_space(name), marl_env.action_space(name), {}) for name in agents_}



register_env(
    "MRLEnvironment",
    lambda _: ParallelPettingZooEnv(marl_env),
)

ppo_config = (
    PPOConfig()
    .environment("MRLEnvironment")  # Use the registered environment name
    .env_runners(
        num_env_runners=1,  # Replaces num_rollout_workers
        num_envs_per_env_runner=1,  # Replaces num_envs_per_worker
        batch_mode="complete_episodes",  # Ensure full episodes are processed
        rollout_fragment_length='auto',  # Length of an episode
    )
    .framework("torch")  # Use PyTorch
    .multi_agent(
        policies=policies_,
        policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id)  # Maps agent IDs directly to policies
    
    .training(
        model = {
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
            (75000, 0.075),  # Start decreasing entropy around iteration 75
            (100000, 0.05), # Reach low entropy by iteration 100
        ],  # Encourage exploration with lower entropy cost
        vf_loss_coeff=5.0,  # Coefficient for value function los
        grad_clip=0.5,  # Clip gradients to stabilize training
    )
)

# Initialize Ray
ray.init()

# Define the stop criteria and logging settings
stop_criteria = {
    'training_iteration':1000 # Specify based on your needs
}

# Define the logging directory
log_dir = f"./tensorboard_logs/env1"
storage_path="file://" + os.path.abspath(log_dir)

# Create a Tuner for training with TensorBoard logging and checkpointing
tuner = tune.Tuner(
    "PPO",  # Specify the RLlib algorithm
    param_space=ppo_config.to_dict(),
    run_config=air.RunConfig(
        stop=stop_criteria,
        verbose=1,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=5,  # Frequency of checkpoints (e.g., every iteration)
            checkpoint_at_end=True  # Ensure final checkpoint at end
        ),
        storage_path=storage_path,  # Directory for TensorBoard logs
        name="PPO_Training_Experiment"
    ),
)

# Run the training
results = tuner.fit()
# Shutdown Ray
ray.shutdown()

