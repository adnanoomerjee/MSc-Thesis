import functools
import jax
import os

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
import importlib

func = 'run.hyperparameter_sweep'
savedir = 'runs'

default = f'hct/training/hyperparam_sweeps/low_level_env_mlp/'

FLAGS = flags.FLAGS

flags.DEFINE_bool('distributed', False, 'initialise distributed.')
flags.DEFINE_integer('config', 1, 'run config')
flags.DEFINE_string('hyperparam_func_path', default, 'hyperparam_sweep_function_path')

logging.set_verbosity(logging.INFO)
jp.set_printoptions(precision=4)

def training_run(env_name, env_parameters, train_parameters, variant_name, filepath):
    
    current_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    env_parameters_name = env_parameters.copy()
    env_parameters_name.pop('architecture_configs')

    training_run_name = f'{variant_name}'
    filepath = f'{filepath}{training_run_name}/'
    if not os.path.exists(filepath):
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

def function_from_path(func_path):
  module_path, func_name = func_path.rsplit('.', 1)
  module = importlib.import_module(module_path)
  func = getattr(module, func_name)
  return func

def main(argv):
  
  if FLAGS.distributed:
    jax.distributed.initialize()

  filepath = FLAGS.hyperparam_func_path
  savepath = filepath + 'runs/'
  hyperparam_func = filepath.replace('/','.') + func
  config = FLAGS.config
  hyperparam_sweep_fn = function_from_path(hyperparam_func) 

  env_params, training_params = hyperparam_sweep_fn()
  training_run(env_name='LowLevel', env_parameters=env_params[config], train_parameters=training_params[config], variant_name=f'{config}', filepath=savepath)

if __name__== '__main__':
  app.run(main)


