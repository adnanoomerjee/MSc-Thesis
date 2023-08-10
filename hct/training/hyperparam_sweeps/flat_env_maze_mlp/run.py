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

filepath = Path('/nfs/nhome/live/aoomerjee/MSc-Thesis/hct/training/hyperparam_sweeps/flat_env_maze_mlp/runs')

FLAT_ENV_PARAMETERS = {
    'reward_sparsity': [1, 2, 4, 8, 12, 22],
    'architecture_configs': [DEFAULT_MLP_CONFIGS]
    }

FLAT_ENV_PARAMETERS_LARGE = {
    'reward_sparsity': [1, 2, 4, 8, 12, 22],
    'architecture_configs': [LARGE_MLP_CONFIGS]
    }

FLAT_ENV_DENSE = {
    'reward_type': ['dense'],
    'architecture_configs': [DEFAULT_MLP_CONFIGS, LARGE_MLP_CONFIGS, VLARGE_MLP_CONFIGS]
}

FLAT_ENV_DENSEVEL = {
    'reward_type': ['dense'],
    'reward_movement': 'velocity',
    'architecture_configs': [DEFAULT_MLP_CONFIGS, LARGE_MLP_CONFIGS, VLARGE_MLP_CONFIGS]
}



FLAT_TRAINING_PARAMETERS = {
    'num_timesteps':700_000_000, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, # trial 0.97 vs 0.99, 0.97 better
    'gradient_clipping':1.0, # trial 1.0 vs 1000000, 1.0 better
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


def hyperparameter_sweep():
    # Get parameters
    params = FLAT_ENV_PARAMETERS
    param_names = params.keys()
    param_values = params.values()
    combinations = list(itertools.product(*param_values))
    env_parameters = [dict(zip(param_names, combination)) for combination in combinations]

    params = FLAT_ENV_PARAMETERS_LARGE
    param_names = params.keys()
    param_values = params.values()
    combinations = list(itertools.product(*param_values))
    env_parameters += [dict(zip(param_names, combination)) for combination in combinations]

    params = FLAT_ENV_DENSE
    param_names = params.keys()
    param_values = params.values()
    combinations = list(itertools.product(*param_values))
    env_parameters += [dict(zip(param_names, combination)) for combination in combinations]
    training_parameters = [FLAT_TRAINING_PARAMETERS for p in env_parameters]

    params = FLAT_ENV_DENSEVEL
    param_names = params.keys()
    param_values = params.values()
    combinations = list(itertools.product(*param_values))
    env_parameters += [dict(zip(param_names, combination)) for combination in combinations]
    training_parameters = [FLAT_TRAINING_PARAMETERS for p in env_parameters]

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

    savepath = f"{filepath.parent}/flat_experimental_results.csv"
    data.to_csv(savepath, index=False)

    return data

