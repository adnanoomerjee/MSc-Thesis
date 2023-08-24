from hct.io.model import load, save
from hct.training.acting import Evaluator
from hct.envs.goal import Goal

from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper, VmapWrapper
from brax.io import json, html
import brax
from brax import base

from absl import logging

import seaborn as sns

import jax
import jax.numpy as jp

import sys
import functools

from absl import app

import os
import shutil

from typing import Literal

import matplotlib.pyplot as plt

path = "/nfs/nhome/live/aoomerjee/MSc-Thesis/hct/training_runs_mlp/LowLevel, {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate'}, 2023-07-29, 16:52:30"

def aggregate(x, axis=0):
  mean = jp.mean(x, axis=axis)
  std = jp.std(x, axis=axis)
  stderr = std / jp.sqrt(x.shape[axis])
  return {'mean': mean, 'std': std, 'stderr': stderr}

def get_training_metrics(modelpath):
    
    training_metrics = load(f"{modelpath}/training_metrics")
    training_metrics.pop('model_variant_name')

    training_steps = []
    metrics_per_training_step = []

    for k, v in training_metrics.items():
        training_steps.append(float(k))
        metrics_per_training_step.append(v)

    metrics_per_training_step[0].update(
      {'training/sps': 0.0, 
      'training/walltime': 0.0, 
      'training/entropy_loss': 0.0, 
      'training/policy_loss': 0.0, 
      'training/total_loss': 0.0, 
      'training/v_loss': 0.0
      }
    )

    return (training_steps, metrics_per_training_step)


def testrun(
    modelpath, 
    seed=1,
    num_eval_envs=128,
    deterministic_eval=False,
    episode_length=None,
    ):
  
  key = jax.random.PRNGKey(seed=seed)

  training_metrics = load(f"{modelpath}/training_metrics")
  model_variant_name = training_metrics['model_variant_name']

  training_params = load(f"{modelpath}/training_params")
  env_params = load(f"{modelpath}/env_params")

  env = load(f"{modelpath}/env")
  action_repeat = env.action_repeat
  episode_length = env.episode_length if episode_length is None else episode_length
  env = AutoResetWrapper(VmapWrapper(EpisodeWrapper(env, episode_length, action_repeat)))
  
  network = load(f"{modelpath}/network")
  params = load(f"{modelpath}/model_params")
  make_inference_fn = load(f"{modelpath}/make_inference_fn")
  
  make_policy = functools.partial(
      make_inference_fn(network),
      obs_mask=env.obs_mask, 
      action_mask=env.action_mask,
      non_actuator_nodes=env.non_actuator_nodes,
      deterministic=deterministic_eval
      )
  
  evaluator = Evaluator(
    eval_env=env,
    eval_policy_fn=make_policy, 
    num_eval_envs=num_eval_envs,
    episode_length=episode_length, 
    action_repeat=action_repeat, 
    key=key,
    extra_fields=['eval_metrics', 'goal', 'steps', 'truncation'],
    save_state=True)
  
  metrics, data = evaluator.run_evaluation(params)
  data = jax.tree_util.tree_map(lambda x: jp.swapaxes(x, 0, 1), data)

  state_extras = data.extras['state_extras']

  episode_metrics = {
      f'eval/episode_{name}': value for name, value 
      in state_extras['eval_metrics'].episode_metrics.items()
  }
  '''
  episode_metrics.update(
      proportion_goal_reached=jp.abs(1-state_extras['eval_metrics'].active_episodes))'''

  #ordered_goal_difficulty = jp.argsort(episode_metrics['eval/episode_goal_distance_world_frame'][:, -1], axis=0)

  #rollouts_to_save = jp.linspace(0, num_eval_envs - 1, 10, dtype=int)

  #goals = jax.tree_map(lambda x: x[ordered_goal_difficulty[rollouts_to_save], -1, ...], 
  #                    state_extras['goal'])

  #rollouts = jax.tree_map(lambda x: x[ordered_goal_difficulty[rollouts_to_save], ...], 
  #                    data.nstate)
  
  episode_metrics = jax.tree_map(lambda x: aggregate(x), episode_metrics)
  episode_steps = state_extras['steps'][0,:]

  experimental_data = {
    'model_variant_name': model_variant_name,
    'env': env,
    'training_params': training_params,
    'env_params': env_params,
    'final_metrics': metrics,
    'testing_data': (episode_steps, episode_metrics),
    'training_data': gettraining_metrics(modelpath),
    'data': data
  }

  return experimental_data


def plot(experimental_data, metric, timeframe: Literal['training', 'testing'], error: Literal[None, 'std', 'stderr'] = None, label = None, err_label = None):
  
  if label is None:
    label = experimental_data['model_variant_name']
    
  steps, metrics = experimental_data[f"{timeframe}_data"]
  max_steps = steps[-1]

  if timeframe=='testing':
    max_steps = 1000
    values = metrics[metric]['mean']
    lower_error = values - metrics[metric][error]
    upper_error = values + metrics[metric][error]

  else:

    values = []
    lower_error = []
    upper_error = []

    for m in metrics:
      value = m[metric]['mean']
      err = m[metric][error]
      values.append(value)
      lower_error.append(value - err)
      upper_error.append(value + err)

  plt.xlabel('Timesteps')
  plt.ylabel(metric)

  plt.xlim(0, max_steps)
  plt.plot(steps, values, linewidth=0.5, label=label)
  if error is not None:
    plt.fill_between(steps, lower_error , upper_error , alpha=0.3, label=err_label) 

def display(experimental_data, rollout_id, append_goal = True):

  select_id = lambda x, y: x[y]

  env = experimental_data['env']
  rollout_states = jax.tree_map(functools.partial(select_id, y=rollout_id), experimental_data['test_rollouts'])
  goal = jax.tree_map(functools.partial(select_id, y=rollout_id), experimental_data['test_goals'])

  rollout = []

  for i in range(rollout_states.q.shape[0]):
    
    rollout_i = jax.tree_map(functools.partial(select_id, y=i), rollout_states)
    rollout.append(rollout_i)

  goal = brax.positional.base.State(
  q=goal.q,
  qd=goal.qd,
  x=goal.x,
  xd=goal.xd,
  contact=None,
  x_i=rollout[-1].x_i,
  xd_i=rollout[-1].xd_i, 
  j=rollout[-1].j, 
  jd=rollout[-1].jd,
  a_p=rollout[-1].a_p, 
  a_c=rollout[-1].a_c, 
  mass=rollout[-1].mass,
)
  if append_goal:
    for i in range(50):
      rollout.append(goal)
      rollout.insert(0, goal)

  return html.render(env.sys.replace(dt=env.dt), rollout)

def jump_rollout(modelpath, seed=0):
  #path = "/nfs/nhome/live/aoomerjee/MSc-Thesis/hct/training/hyperparam_sweeps/low_level_env_mlp/runs/6"
  network = load(f"{path}/network")
  params = load(f"{path}/model_params")
  make_inference_fn = load(f"{path}/make_inference_fn")
  env = load(f"{path}/env")
  make_policy = make_inference_fn(network)
  policy = make_policy(params, non_actuator_nodes=0, deterministic=True)
  jit_inference_fn = jax.jit(policy)

  jit_env_reset = jax.jit(env.reset)
  jit_env_step = jax.jit(env.step)

  rng = jax.random.PRNGKey(seed=seed)

  rollout = []
  states=[]
  action = []
  ext = []
  metrics = []
  reward = []

  state = jit_env_reset(rng=rng)

  goal = state.info['goal']
  root = goal.x.take(0).__add__(base.Transform(jp.array([0,0,0.5]), jp.array([0,0,0,0])))
  goal = Goal(
    q=goal.q,
    qd=goal.qd,
    x_rel=goal.x_rel.index_set(0, root),
    x=goal.x.__add__(base.Transform(jp.array([0,0,0.8]), jp.array([0,0,0,0]))),
    xd_rel=goal.xd_rel,
    xd=goal.xd
  )
  state.info.update(goal=goal)

  for _ in range(1000):
    states.append(state)
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(rng)
    act, extras = jit_inference_fn(state.obs, act_rng)
    action.append(act)
    ext.append(extras)
    metrics.append(state.metrics)
    state = jit_env_step(state, act)
    reward.append(state.metrics['reward'])

  goal = brax.positional.base.State(
    q=goal.q,
    qd=goal.qd,
    x=goal.x,
    xd=goal.xd,
    contact=None,
    x_i=rollout[-1].x_i,
    xd_i=rollout[-1].xd_i, 
    j=rollout[-1].j, 
    jd=rollout[-1].jd,
    a_p=rollout[-1].a_p, 
    a_c=rollout[-1].a_c, 
    mass=rollout[-1].mass,
  )
  for i in range(50):
      rollout.append(goal)
  return html.render(env.sys.replace(dt=env.dt), rollout)
'''
def testing_plots(modelpath, metric, error: Literal[None, 'std', 'stderr'] = None):
  metrics = load(f"{modelpath}/training_metrics")
  x = []
  y = []
  y_upper = []
  y_lower = []
  metrics.pop('model_variant_name')
  for k, v in metrics.items():
      x.append(float(k))
      y.append(v[f'{metric}']['mean'])
      if error is not None:
        y_upper.append(v[f'{metric}']['mean'] + v[f'{metric}']['mean'][f'{error}'])
        y_lower.append(v[f'{metric}']['mean'] - v[f'{metric}']['mean'][f'{error}'])

  plt.plot(x,y, linewidth=0.5)
  if error is not None:
    plt.fill_between(x, y_lower, y_upper, alpha=0.3, label='Stderr Shading') '''


def main(argv):
  logging.get_absl_handler().use_absl_log_file('log', "hct")
  logging.set_verbosity(logging.INFO)
  jp.set_printoptions(precision=4)

  modelpath = "hct/training/hyperparam_sweeps/low_level_env_mlp_v2/runs"
  savepath = "hct/training/hyperparam_sweeps/low_level_env_mlp_v2/experimental_data"

  for i in range(8):
    logging.info(f"{i+1}/8")
    output = testrun(f"{modelpath}/{i}", seed=8, episode_length=500)  
    save(f"{savepath}/{i}", output)

if __name__== '__main__':
  app.run(main)


