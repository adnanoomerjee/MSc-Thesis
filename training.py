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
flags.DEFINE_bool('distribute', True, 'initialise distribute.')
logging.set_verbosity(logging.INFO)

def training_run(env_name, env_parameters, train_parameters):
    
    current_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    training_run_name = f'{env_name}, {env_parameters}, {current_datetime}'
    filepath = f'{cwd}/hct/training_runs/{training_run_name}/'
    os.makedirs(os.path.dirname(filepath))

    logging.get_absl_handler().use_absl_log_file('log', filepath)

    env = envs.get_environment(
        env_name=env_name,
        **env_parameters)
    
    train_fn = training.get_train_fn(
        env=env,
        **train_parameters)

    training_run_metrics = {}
    training_parameters = env.parameters | train_fn.keywords
    training_parameters.pop('environment')
    training_parameters['env name'] = env_name

    model.save(obj=training_parameters, path=f'{filepath}/training_params')

    def progress(num_steps, metrics):
      training_run_metrics.update({str(num_steps): metrics})

    def save(current_step, make_policy, params):
      model.save(obj=training_run_metrics, path=f'{filepath}/training_metrics')
      model.save(obj=params, path=f'{filepath}/model_params')
      model.save(obj=make_policy, path=f'{filepath}/make_policy')
      
    make_policy, params, metrics = train_fn(
        progress_fn=progress, 
        policy_params_fn=save
    )

def main(argv):

  if FLAGS.distribute:
    jax.distributed.initialize()

  for config in LOW_LEVEL_ENV_PARAMETERS:
    training_run(env_name='LowLevel', env_parameters=config, train_parameters={})

if __name__== '__main__':
  app.run(main)


