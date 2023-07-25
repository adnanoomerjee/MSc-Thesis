"""Architecture config for behavioral cloning"""
import functools
from typing import Literal

from flax import struct, linen

from hct.training import train as ppo

"""Default network configurations"""
DEFAULT_MLP_CONFIGS = {
    'policy_hidden_layer_sizes': (1024,) * 4,
    'value_hidden_layer_sizes': (1024,) * 5,
    'activation': linen.relu,
    'dropout_rate': 0.2
}

DEFAULT_TRANSFORMER_CONFIGS = {
    'num_layers': 3,
    'd_model': 256,
    'num_heads': 2,
    'dim_feedforward': 512,
    'dropout_rate': 0.2,
    'transformer_norm': True,
    'condition_decoder': True
}

"""Env training run parameters"""
LOW_LEVEL_ENV_PARAMETERS = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'difference', 'goal_obs': 'concatenate'},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'difference', 'goal_obs': 'concatenate'},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'difference', 'goal_obs': 'concatenate'},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'difference', 'goal_obs': 'concatenate'}
]

"""train_fn training run parameters"""
TEST_TRAINING_PARAMETERS = {
    'num_timesteps':512, 
    'episode_length':50, 
    'action_repeat':1, 
    'num_envs':2, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'seed':1,
    'unroll_length':2,
    'batch_size':2, 
    'num_minibatches':2,
    'num_updates_per_batch':4,
    'num_evals':4, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}
    
@struct.dataclass
class NetworkArchitecture:
    name: str
    configs: dict

    @classmethod
    def create(cls, name: Literal['MLP', 'Transformer'], **new_configs):
        if name == 'MLP':
            configs = DEFAULT_MLP_CONFIGS.copy()
        elif name == 'Transformer':
            configs = DEFAULT_TRANSFORMER_CONFIGS.copy()
        else:
            raise ValueError(f"Unsupported architecture name: {name}. Choose either 'MLP' or 'Transformer'.")
        configs.update(new_configs)
        return NetworkArchitecture(name=name, configs=configs)
        

