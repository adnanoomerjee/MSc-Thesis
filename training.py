import functools
import jax
import os

from hct.training.configs import *

cwd = os.getcwd()

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

FLAGS = flags.FLAGS
flags.DEFINE_bool('distributed', False, 'initialise distributed.')
flags.DEFINE_integer('config', 0, 'run config')
logging.set_verbosity(logging.INFO)

def training_run(env_name, env_parameters, train_parameters):
    
    current_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    env_parameters_name = env_parameters.copy()
    env_parameters_name.pop('architecture_configs')
    training_run_name = f'{env_name}, {env_parameters_name}, {current_datetime}'
    filepath = f'{cwd}/hct/training_runs_mlp/{training_run_name}/'
    os.makedirs(os.path.dirname(filepath))

    logging.get_absl_handler().use_absl_log_file('log', filepath)

    env = envs.get_environment(
        env_name=env_name,
        **env_parameters)
    
    train_fn = training.get_train_fn(
        env=env,
        **train_parameters)

    training_run_metrics = {}
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

def main(argv):
  
  if FLAGS.distributed:
    jax.distributed.initialize()
  config_int = FLAGS.config
  config = LOW_LEVEL_ENV_PARAMETERS_MLP_4[config_int]
  training_run(env_name='LowLevel', env_parameters=config, train_parameters=V4_MLP_PARAMETERS)

if __name__== '__main__':
  app.run(main)


