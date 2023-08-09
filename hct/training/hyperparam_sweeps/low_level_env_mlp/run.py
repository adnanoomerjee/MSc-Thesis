import functools
from typing import Literal
import itertools

import json

import jax
from flax import struct, linen

from hct.training import train as ppo
from hct.training.configs import *
from hct.io.model import load, save

from brax.envs.base import Env

import pandas as pd

from pathlib import Path

"""Low level training run parameters"""

filepath = Path('/nfs/nhome/live/aoomerjee/MSc-Thesis/hct/training/hyperparam_sweeps/low_level_env_mlp/runs2')
'''
LOW_LEVEL_ENV_PARAMETERS = {
    'position_goals': (True, False),
    'velocity_goals': (None, 'root', 'full'),
    'reward_goal_reached': (0, 50, 100),
    'unhealthy_cost': (0, -1),
    'air_probability': (0.1, 0.3),
    'distance_reward': ('absolute', 'relative')
    }'''

ADDITIONAL_LOW_LEVEL_ENV_PARAMETERS = {
    'position_goals': True,
    'velocity_goals': 'root',
    'goal_root_pos_masked': [True],
    'reward_goal_reached': (0, 50, 100),
    'unhealthy_cost': (0, -1),
    'air_probability': (0.1, 0.3),
    'distance_reward': ('absolute', 'relative')
    }

ADDITIONAL_LOW_LEVEL_ENV_PARAMETERS_2 = {
    'position_goals': [True, False],
    'velocity_goals': (None, 'root', 'full'),
    'reward_goal_reached': [0],
    'unhealthy_cost': [-1],
    'air_probability': [0.1],
    'distance_reward': ['absolute']
    }

LOW_LEVEL_ENV_PARAMETERS = {
    'position_goals': (True),
    'velocity_goals': (None),
    'reward_goal_reached': (0, 50, 100),
    'unhealthy_cost': (0, -1),
    'distance_reward': ('absolute', 'relative'),
    'architecture_configs': [DEFAULT_MLP_CONFIGS]
    }

LOW_LEVEL_TRAINING_PARAMETERS = {
    'num_timesteps':300_000_000, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, # trial 0.97 vs 0.99, 0.97 better
    'gradient_clipping':1.0, 
    'seed':5,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':102, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

FINAL_LOW_LEVEL_ENV_PARAMETERS = {
    'position_goals': True,
    'velocity_goals': None,
    'reward_goal_reached': 0,
    'unhealthy_cost': -1,
    'distance_reward': 'absolute',
    'architecture_configs': LARGE_MLP_CONFIGS
    }

FINAL2_LOW_LEVEL_ENV_PARAMETERS = {
    'position_goals': True,
    'velocity_goals': None,
    'reward_goal_reached': 0,
    'unhealthy_cost': -1,
    'distance_reward': 'absolute',
    'architecture_configs': LARGE_MLP_CONFIGS
    }

'''
def hyperparameter_sweep():
    # Special handling for 'position_goals' and 'velocity_goals'
    pos_vel_combinations = [(True, v) for v in LOW_LEVEL_ENV_PARAMETERS['velocity_goals']] + [(False, 'full')]

    # Get other parameters
    other_params = {key: value for key, value in LOW_LEVEL_ENV_PARAMETERS.items() if key not in ['position_goals', 'velocity_goals']}
    other_param_names = other_params.keys()
    other_param_values = other_params.values()

    # Generate all combinations of the other parameters
    other_param_combinations = list(itertools.product(*other_param_values))

    # Combine 'pos_vel_combinations' with 'other_param_combinations'
    env_parameters = []
    for pos_vel in pos_vel_combinations:
        for other in other_param_combinations:
            combined_dict = {'position_goals': pos_vel[0], 'velocity_goals': pos_vel[1], 'architecture_configs': DEFAULT_MLP_CONFIGS}
            for i, name in enumerate(other_param_names):
                combined_dict[name] = other[i]
            env_parameters.append(combined_dict)
    
    # additional runs
    other_params = {key: value for key, value in ADDITIONAL_LOW_LEVEL_ENV_PARAMETERS.items() if key not in ['position_goals', 'velocity_goals']}
    other_param_names = other_params.keys()
    other_param_values = other_params.values()

    # Generate all combinations of the other parameters
    other_param_combinations = list(itertools.product(*other_param_values))
    
    for other in other_param_combinations:
        combined_dict = {'position_goals': ADDITIONAL_LOW_LEVEL_ENV_PARAMETERS['position_goals'], 'velocity_goals': ADDITIONAL_LOW_LEVEL_ENV_PARAMETERS['velocity_goals'], 'architecture_configs': DEFAULT_MLP_CONFIGS}
        for i, name in enumerate(other_param_names):
            combined_dict[name] = other[i]
        env_parameters.append(combined_dict)   
    
    other_params = {key: value for key, value in ADDITIONAL_LOW_LEVEL_ENV_PARAMETERS_2.items() if key not in ['position_goals', 'velocity_goals']}
    other_param_names = other_params.keys()
    other_param_values = other_params.values()

    # Generate all combinations of the other parameters
    other_param_combinations = list(itertools.product(*other_param_values))

    # Combine 'pos_vel_combinations' with 'other_param_combinations'
    for pos_vel in pos_vel_combinations:
        for other in other_param_combinations:
            combined_dict = {'position_goals': pos_vel[0], 'velocity_goals': pos_vel[1], 'architecture_configs': LARGE_MLP_CONFIGS}
            for i, name in enumerate(other_param_names):
                combined_dict[name] = other[i]
            env_parameters.append(combined_dict)
    

    training_parameters = [LOW_LEVEL_TRAINING_PARAMETERS for p in env_parameters]
    return env_parameters, training_parameters
'''

def hyperparameter_sweep():
    # Special handling for 'position_goals' and 'velocity_goals'
    params = LOW_LEVEL_ENV_PARAMETERS
    param_names = params.keys()
    param_values = params.values()
    combinations = list(itertools.product(*param_values))
    env_parameters = [dict(zip(param_names, combination)) for combination in combinations]
    training_parameters = [LOW_LEVEL_TRAINING_PARAMETERS for p in env_parameters]
    return env_parameters, training_parameters

def generate_data_tables():

    env_params, training_params = hyperparameter_sweep()

    data = []
    
    for run in filepath.iterdir():

        training_metrics = load(f'{run}/training_metrics')
        model_variant = int(training_metrics['model_variant_name'])

        env_params[model_variant].pop('architecture_configs')

        final_metrics = {'Model Variant ID': model_variant}
        final_metrics.update(env_params[model_variant])
        final_metrics.update(training_metrics[list(training_metrics.keys())[-1]])
        final_metrics = {key: f"{value['mean']:.4f} ± {value['std']:.4f}" if isinstance(value, dict) else value for key, value in final_metrics.items()}
    
        data.append(final_metrics)

    data = pd.DataFrame(data)
    data.sort_values(by='Model Variant ID', inplace=True, ascending=True)

    savepath = f"{filepath.parent}/low_level_experimental_results.csv"
    data.to_csv(savepath, index=False)

    return data

