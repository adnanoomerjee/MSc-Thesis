"""Architecture config for behavioral cloning"""
import functools
from typing import Literal

from flax import struct, linen

from hct.training import train as ppo

"""Default network configurations"""
DEFAULT_MLP_CONFIGS = {
    'policy_hidden_layer_sizes': (32,) * 4,
    'value_hidden_layer_sizes': (256,) * 5
}

DEFAULT_TRANSFORMER_CONFIGS = {
    'policy_num_layers': 2,
    'policy_d_model': 16,
    'policy_num_heads': 2,
    'policy_dim_feedforward': 32,
    'policy_dropout_rate': 0.0,
    'policy_transformer_norm': True,
    'policy_condition_decoder': True,
    'value_num_layers': 4,
    'value_d_model': 32,
    'value_num_heads': 2,
    'value_dim_feedforward': 64,
    'value_dropout_rate': 0.0,
    'value_transformer_norm': True,
    'value_condition_decoder': True
}


DEFAULT_TRANSFORMER_CONFIGS_ALTERNATE = {
    'policy_num_layers': 4,
    'policy_d_model': 16,
    'policy_num_heads': 2,
    'policy_dim_feedforward': 32,
    'policy_dropout_rate': 0.0,
    'policy_transformer_norm': True,
    'policy_condition_decoder': True,
    'value_num_layers': 4,
    'value_d_model': 32,
    'value_num_heads': 2,
    'value_dim_feedforward': 64,
    'value_dropout_rate': 0.0,
    'value_transformer_norm': True,
    'value_condition_decoder': True
}


VVSMALL_TRANSFORMER_CONFIGS = {
    'policy_num_layers': 4,
    'policy_d_model': 8,
    'policy_num_heads': 2,
    'policy_dim_feedforward': 32,
    'policy_dropout_rate': 0.0,
    'policy_transformer_norm': True,
    'policy_condition_decoder': True,
    'value_num_layers': 4,
    'value_d_model': 16,
    'value_num_heads': 2,
    'value_dim_feedforward': 32,
    'value_dropout_rate': 0.0,
    'value_transformer_norm': True,
    'value_condition_decoder': True
}

VSMALL_TRANSFORMER_CONFIGS = {
    'num_layers': 2,
    'd_model': 32,
    'num_heads': 2,
    'dim_feedforward': 64,
    'dropout_rate': 0.0,
    'transformer_norm': True,
    'condition_decoder': True
}

SMALL_TRANSFORMER_CONFIGS = {
    'num_layers': 3,
    'd_model': 64,
    'num_heads': 2,
    'dim_feedforward': 128,
    'dropout_rate': 0.1,
    'transformer_norm': True,
    'condition_decoder': True
}

MID_TRANSFORMER_CONFIGS = {
    'num_layers': 2,
    'd_model': 128,
    'num_heads': 2,
    'dim_feedforward': 256,
    'dropout_rate': 0.1,
    'transformer_norm': True,
    'condition_decoder': True
}

MID2_TRANSFORMER_CONFIGS = {
    'num_layers': 8,
    'd_model': 64,
    'num_heads': 2,
    'dim_feedforward': 128,
    'dropout_rate': 0.1,
    'transformer_norm': True,
    'condition_decoder': True
}

MID3_TRANSFORMER_CONFIGS = {
    'num_layers': 8,
    'd_model': 128,
    'num_heads': 2,
    'dim_feedforward': 256,
    'dropout_rate': 0.1,
    'transformer_norm': True,
    'condition_decoder': True
}

LARGE_TRANSFORMER_CONFIGS = {
    'num_layers': 2,
    'd_model': 128,
    'num_heads': 2,
    'dim_feedforward': 256,
    'dropout_rate': 0.1,
    'transformer_norm': True,
    'condition_decoder': True
}



"""Env training run parameters"""
LOW_LEVEL_ENV_PARAMETERS = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate'},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate'},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate'},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate'}
]

LOW_LEVEL_ENV_PARAMETERS_V3 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_V4 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_V6 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID3_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID3_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID3_TRANSFORMER_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID3_TRANSFORMER_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_V7 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID3_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID3_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID3_TRANSFORMER_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID3_TRANSFORMER_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_V8 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': SMALL_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': SMALL_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': SMALL_TRANSFORMER_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': SMALL_TRANSFORMER_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_V9 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': SMALL_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': SMALL_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': SMALL_TRANSFORMER_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': SMALL_TRANSFORMER_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_V10 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': SMALL_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': SMALL_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': SMALL_TRANSFORMER_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': SMALL_TRANSFORMER_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_V11 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_V12 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_V13 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': MID2_TRANSFORMER_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_V14 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_V15 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_V16 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'difference', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_V17 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': VSMALL_TRANSFORMER_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_VEL = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': VVSMALL_TRANSFORMER_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': VVSMALL_TRANSFORMER_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': VVSMALL_TRANSFORMER_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_MLP_1 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_MLP_2 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS, 'terminate_when_unhealthy':False, 'ctrl_cost':0.5, 'reward_goal_reached': 0},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_MLP_3 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_MLP_4 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_TRANSFORMER_CONFIGS_ALTERNATE},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_MLP_5 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_MLP_6 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS}
]

LOW_LEVEL_ENV_PARAMETERS_MLP_7 = [
   {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS, 'reward_goal_reached': 500, 'unhealthy_cost': 1.0},
   {'position_goals': True, 'velocity_goals': 'root', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': True, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS},
   {'position_goals': False, 'velocity_goals': 'full', 'distance_reward': 'absolute', 'goal_obs': 'concatenate', 'architecture_configs': DEFAULT_MLP_CONFIGS}
]

LOW_LEVEL_POS_GOALS_MLP = [
    {}
]
"""train_fn training run parameters"""

V1_PARAMETERS = { 
    'num_timesteps':50_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.9, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':256, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':512, 
    'normalize_observations':False,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# small without PE
V2_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':1e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':512, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':512, 
    'normalize_observations':False,
    'reward_scaling':1,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# small with PE
V3_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':512, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':512, 
    'normalize_observations':False,
    'reward_scaling':1,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# with PE and mid-size model
V4_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':1e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':1024, 
    'num_minibatches':16,
    'num_updates_per_batch':4,
    'num_evals':256, 
    'normalize_observations':False,
    'reward_scaling':1,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# with PE and mid2-size model
V5_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':1024, 
    'num_minibatches':16,
    'num_updates_per_batch':4,
    'num_evals':256, 
    'normalize_observations':False,
    'reward_scaling':1,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# with PE and mid3-size model
V6_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':1e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':1024, 
    'num_minibatches':16,
    'num_updates_per_batch':4,
    'num_evals':256, 
    'normalize_observations':False,
    'reward_scaling':1,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# with PE1D and mid3-size model
V7_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':1e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':1024, 
    'num_minibatches':16,
    'num_updates_per_batch':4,
    'num_evals':256, 
    'normalize_observations':False,
    'reward_scaling':1,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# with PE1D and small model
V8_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':1e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':512, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':256, 
    'normalize_observations':False,
    'reward_scaling':1,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# with PE1D and small model, no rot dist
V9_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':1e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':512, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':256, 
    'normalize_observations':False,
    'reward_scaling':1,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# with PE1D and small model, no rot dist, no unhealthy, bigger pos range 
V10_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':1e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':512, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':256, 
    'normalize_observations':False,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# reduce entropy cost
V11_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':1e-4, 
    'entropy_cost':1e-3,
    'discounting':0.97, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':512, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':256, 
    'normalize_observations':False,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# increase entropy cost, deterministic eval
V12_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':1e-4, 
    'entropy_cost':1e-1,
    'discounting':0.97, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':512, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':256, 
    'normalize_observations':False,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':True,
    'normalize_advantage':True
}

# decrease entropy cost, deterministic eval False, increase discount, difference reward, increase lr
V13_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.99, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':512, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':256, 
    'normalize_observations':False,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# increase reward scaling, reduce lr, vsmall model
V14_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':1e-4, 
    'entropy_cost':1e-2,
    'discounting':0.99, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':512, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':256, 
    'normalize_observations':False,
    'reward_scaling':100,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# reduce reward scaling, increase lr, increase batch size
V15_PARAMETERS = {
    'num_timesteps':500_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.99, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':256, 
    'normalize_observations':False,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# decrease lr
V16_PARAMETERS = {
    'num_timesteps':500_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':1e-4, 
    'entropy_cost':1e-2,
    'discounting':0.99, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':256, 
    'normalize_observations':False,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# absolutte reward, batch size 
V17_PARAMETERS = {
    'num_timesteps':500_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':1e-4, 
    'entropy_cost':1e-2,
    'discounting':0.99, 
    'gradient_clipping':0.1,
    'seed':1,
    'unroll_length':5,
    'batch_size':4096, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':256, 
    'normalize_observations':False,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

 
VEL_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':1000000,
    'seed':5,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':10, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# Default MLP
V1_MLP_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':1000000,
    'seed':5,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':10, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}
 
# remove terminate when unhealthy, add ctrl cost
V2_MLP_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':1000000,
    'seed':5,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':10, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# reward goal reached, more steps
V3_MLP_PARAMETERS = {
    'num_timesteps':500_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':1000000,
    'seed':5,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':10, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# remove ctrl cost, V2 comparison
V4_MLP_PARAMETERS = {
    'num_timesteps':500_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':1000000,
    'seed':5,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':10, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# goal_reached_reward
V5_MLP_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':1000000,
    'seed':5,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':10, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

# unhealthy cost 50
V6_MLP_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':1000000,
    'seed':5,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':10, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}


# unhealthy cost reduced to 1
V7_MLP_PARAMETERS = {
    'num_timesteps':100_000_000, 
    'episode_length':1000, 
    'action_repeat':1, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':1000000,
    'seed':5,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':100, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}


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
        

