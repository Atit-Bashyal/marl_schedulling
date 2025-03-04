
import os
import re
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils import EzPickle, seeding

import numpy as np

from factory_single_agent import Factory
from rl_paper_env1 import RLEnv

import ray
from ray import air, tune
# from ray.rllib.connectors.env_to_module import FlattenObservations
# from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
# from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
# from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.logger import UnifiedLogger
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.impala import IMPALAConfig
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

alpha_value = 0.8

factory = Factory(root,price_signal,pv_signal)

single_agent_env = RLEnv(factory,alpha_value)


# Register the single-agent environment
register_env(
    "SingleAgentEnvironment",
    lambda _: single_agent_env
)

# Define PPO configuration for a single-agent setup
impala_config = (
    IMPALAConfig()
    .environment("SingleAgentEnvironment")  # Use the registered environment name
    .env_runners(
        num_env_runners=4,  # Single environment
        num_envs_per_env_runner=1,
        batch_mode="complete_episodes"
    )
    .framework("torch")  # Use PyTorch
    .training(lr=tune.grid_search([0.0001, 0.0002]), grad_clip=20.0)
    .learners(num_learners=1)
        )

# Initialize Ray
ray.init()

# Define the stop criteria and logging settings
stop_criteria = {
    'training_iteration': 1000  # Adjust as needed
}

# Define the logging directory
log_dir = "./tensorboard_logs/single_agent"
storage_path = "file://" + os.path.abspath(log_dir)

# Create a Tuner for training
tuner = tune.Tuner(
    "IMPALA",
    param_space=impala_config.to_dict(),
    run_config=air.RunConfig(
        stop=stop_criteria,
        verbose=1,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=5,
            checkpoint_at_end=True
        ),
        storage_path=storage_path,
        name="IMPALA_Training_Experiment"
    ),
)

# Run the training
results = tuner.fit()

# Shutdown Ray
ray.shutdown()