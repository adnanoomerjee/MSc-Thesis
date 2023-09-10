import functools
from typing import Literal
import itertools

import json
import sys
import os
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

FLAGS = flags.FLAGS

flags.DEFINE_bool('distributed', False, 'initialise distributed.')
flags.DEFINE_integer('config', 1, 'run config')

logging.set_verbosity(logging.INFO)
jp.set_printoptions(precision=4)

env_name = 'AntTest'
filedir = os.path.dirname(os.path.abspath(__file__))
filepath = f"{filedir}/runs/"

"""Low level training run parameters"""


ENV_P = {'backend':'positional', 'architecture_configs':DEFAULT_MLP_CONFIGS}

TRAIN_P = {
    'num_timesteps':50_000_000, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, # trial 0.97 vs 0.99, 0.97 better
    'gradient_clipping':0, 
    'seed':5,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':12, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}
'''
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
'''
def main(argv):
  
  if FLAGS.distributed:
    jax.distributed.initialize()

  training_run(env_name=env_name, env_parameters=ENV_P, train_parameters=TRAIN_P, variant_name=f'test', filepath=filepath)

if __name__== '__main__':
  app.run(main)
