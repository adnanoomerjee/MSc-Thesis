import functools
from typing import Literal
import itertools

import json
import sys
import os
sys.path.append("/nfs/nhome/live/aoomerjee/MSc-Thesis/")

import training_run

import jax
from flax import struct, linen

from hct.training.training_run import training_run
from hct.training.configs import *
from hct.io.model import load, save

from brax.envs.base import Env

import pandas as pd

from pathlib import Path

from absl import app, logging, flags
import jax.numpy as jp

env_name = 'MidLevelV2'
filedir = os.path.dirname(os.path.abspath(__file__))
filepath = f"{filedir}/runs/"

FLAGS = flags.FLAGS

flags.DEFINE_integer('config', 1, 'run config')

logging.set_verbosity(logging.INFO)
jp.set_printoptions(precision=4, linewidth=1000)

low_level_modelpath = "/nfs/nhome/live/aoomerjee/MSc-Thesis/hct/training/hyperparam_sweeps/low_level_env_mlp_v2/runs/3"

"""Low level training run parameters"""

MID_LEVEL_ENV_PARAMETERS = {
    'distance_reward': ['relative','absolute'],
    'low_level_modelpath': [low_level_modelpath],
    'architecture_configs': [DEFAULT_MLP_CONFIGS, LARGE_MLP_CONFIGS, VLARGE_MLP_CONFIGS],
    'action_repeat': [10, 20],
    'root_velocity_goals': [False]
    }


MID_LEVEL_TRAINING_PARAMETERS = {
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

def hyperparameter_sweep():

    params = MID_LEVEL_ENV_PARAMETERS
    param_names = params.keys()
    param_values = params.values()
    combinations = list(itertools.product(*param_values))
    env_parameters = [dict(zip(param_names, combination)) for combination in combinations]
    training_parameters = [MID_LEVEL_TRAINING_PARAMETERS for p in env_parameters]
    
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

    savepath = f"{filepath.parent}/experimental_results.csv"
    data.to_csv(savepath, index=False)

    return data


def main(argv):

  config = FLAGS.config

  env_params, training_params = hyperparameter_sweep()

  env_p = env_params[config]
  train_p = training_params[config]

  training_run(env_name=env_name, env_parameters=env_p, train_parameters=train_p, variant_name=f'{config}', filepath=filepath)

if __name__== '__main__':
  app.run(main)

