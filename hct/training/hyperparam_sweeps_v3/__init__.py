import functools
from typing import Literal
import itertools

import json

import traceback
import os

import sys
sys.path.append("/nfs/nhome/live/aoomerjee/MSc-Thesis/")

import jax
from flax import struct, linen

from hct.training import train as ppo
from hct.training.configs import *
from hct.io.model import load, save

from brax.envs.base import Env

import pandas as pd

from pathlib import Path
from absl import logging, flags, app

from hct.training.training_run import training_run
import jax.numpy as jp




TASK_TRAINING_PARAMETERS = {
    'num_timesteps':50_000_000, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.99, # trial 0.97 vs 0.99, 0.97 better
    'gradient_clipping':0, # trial 1.0 vs 1000000, 1.0 better
    'seed':5,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':42, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True,
    'num_eval_envs': 128
}

HMA_PRETRAIN_TRAINING_PARAMETERS = {
    'num_timesteps':150_000_000, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, 
    'gradient_clipping':0, 
    'seed':5,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':42, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True,
}

action_repeats = [1, 5] 
action_repeats2 = [1, 5] 

low_level_modelpath = "/nfs/nhome/live/aoomerjee/MSc-Thesis/hct/training/hyperparam_sweeps_v3/hma/low_level/runs"
low_level_model_ids = [i for i in range(3)]
low_level_models = [f"{low_level_modelpath}/{i}" for i in low_level_model_ids]
low_level_models2 = [f"{low_level_modelpath}/{i}" for i in [3, 4, 5]]

mid_level_modelpath = "/nfs/nhome/live/aoomerjee/MSc-Thesis/hct/training/hyperparam_sweeps_v3/hma/mid_level/runs"
mid_level_model_ids = [i for i in range(6)]
mid_level_models = [f"{mid_level_modelpath}/{i}" for i in mid_level_model_ids]

architecture_configs = [VLARGE_MLP_CONFIGS]

FLAT_PARAMS = {
    'reward_type': ['dense', 'sparse'],
    'task_information': [True, False],
    'architecture_configs': architecture_configs
    }

HMA2_PARAMS = {
    'state_below': [True],
    'low_level_modelpath': low_level_models,
    'reward_type': ['dense', 'sparse'],
    'task_information': [True],
    'reward_movement': ['velocity'],
    'action_repeat': action_repeats, 
    'architecture_configs': architecture_configs
}

HMA3_PARAMS = {
    'state': ['below'],
    'mid_level_modelpath': mid_level_models,
    'reward_type': ['dense', 'sparse'],
    'task_information': [True],
    'reward_movement': ['velocity'],
    'action_repeat': action_repeats2,
    'architecture_configs': architecture_configs
}

def generate_data_tables(filepath, hyperparameter_sweep):

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

def make_hyperparameter_sweep(env_params, training_params):

    def hyperparameter_sweep():
        # Get parameters
        params = env_params
        param_names = params.keys()
        param_values = params.values()
        combinations = list(itertools.product(*param_values))
        env_parameters = [dict(zip(param_names, combination)) for combination in combinations]

        training_parameters = [training_params for p in env_parameters]
        return env_parameters, training_parameters
    
    return hyperparameter_sweep




def make_main(hyperparameter_sweep, env_name, filepath):

    def main(argv):
        FLAGS = flags.FLAGS

        flags.DEFINE_bool('distributed', False, 'initialise distributed.')
        flags.DEFINE_integer('config', 0, 'run config')

        logging.set_verbosity(logging.INFO)
        jp.set_printoptions(precision=4)

        # Set up the absl logging and verbosity level
        logging.set_verbosity(logging.INFO)

        try:

            if FLAGS.distributed:
                jax.distributed.initialize()

            config = FLAGS.config

            env_params, training_params = hyperparameter_sweep()
            logging.info(env_params)

            env_p = env_params[config]
            train_p = training_params[config]
            training_run(env_name=env_name, env_parameters=env_p, train_parameters=train_p, variant_name=f'{config}', filepath=filepath)
        
        except Exception as e:
            tb = traceback.format_exc()
            logging.info(tb)

    return main
