import functools
import jax
import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/nfs/nhome/live/aoomerjee/MSc-Thesis/")
from hct.training.configs import *

from datetime import datetime
from jax import numpy as jp
import matplotlib.pyplot as plt

from IPython.display import HTML, clear_output

from hct import training
from hct import envs

import flax

from hct.io import model
from brax.io import json
from brax.io import html

from absl import app, logging, flags
import importlib
import shutil

def training_run(env_name, env_parameters, train_parameters, variant_name, filepath):
    
    current_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    env_parameters_name = env_parameters.copy()
    env_parameters_name.pop('architecture_configs')

    training_run_name = f'{variant_name}'
    filepath = f'{filepath}{training_run_name}/'
    
    if os.path.exists(filepath):
      shutil.rmtree(filepath)
    os.makedirs(os.path.dirname(filepath))      

    logging.get_absl_handler().use_absl_log_file('log', filepath)

    env = envs.get_environment(
        env_name=env_name,
        **env_parameters)
    
    train_fn = training.get_train_fn(
        env=env,
        **train_parameters)

    training_run_metrics = {
      'model_variant_name': training_run_name
    }
    training_parameters = train_fn.keywords.copy()
    training_parameters.pop('environment')
    training_parameters['env_name'] = env_name

    model.save(obj=training_parameters, path=f'{filepath}/training_params')
    model.save(obj=env_parameters, path=f'{filepath}/env_params')
    model.save(obj=env, path=f'{filepath}/env')

    def progress(num_steps, metrics):
      training_run_metrics.update({str(num_steps): metrics})

    def save(current_step, make_policy, params, make_inference_fn, network):
      model.save(obj=training_run_metrics, path=f'{filepath}/training_metrics')
      model.save(obj=params, path=f'{filepath}/model_params')
      model.save(obj=make_inference_fn, path=f'{filepath}/make_inference_fn')
      model.save(obj=network, path=f'{filepath}/network')
      
    make_policy, params, metrics = train_fn(
        progress_fn=progress, 
        policy_params_fn=save
    )


