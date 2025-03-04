import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2



class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Define dimensions
        input_dim = obs_space.shape[0]
        hidden_dim = 56

        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input to first hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # First to second hidden layer
        self.policy_output = nn.Linear(hidden_dim, num_outputs)  # Final policy layer
        self.value_output = nn.Linear(hidden_dim, 1)  # Final value layer

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # Store the last feature vector for value_function
        self._last_features = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]  # Observation input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Save the features for value_function
        self._last_features = x
        
        # Policy output
        policy = self.tanh(self.policy_output(x))
        return policy, state

    def value_function(self):
        if self._last_features is None:
            raise ValueError("No features computed in forward pass before calling value_function.")
        # Value function output
        value = self.value_output(self._last_features)
        return value.squeeze(-1) 
