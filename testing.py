from hct.io.model import load
from hct.training.acting import Evaluator
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper, VmapWrapper
import seaborn as sns

import jax
import jax.numpy as jp

import functools

from absl import app, main

path = "/nfs/nhome/live/aoomerjee/MSc-Thesis/hct/training_runs_mlp/LowLevel, {'position_goals': True, 'velocity_goals': None, 'distance_reward': 'absolute', 'goal_obs': 'concatenate'}, 2023-07-29, 16:52:30"

def aggregate(x, axis=-1):
  mean = jp.mean(x, axis=axis)
  std = jp.std(x, axis=axis)
  stderr = std / jp.sqrt(x.shape[axis])
  return {'mean': mean, 'std': std, 'stderr': stderr}

def testrun(
    modelpath, 
    seed=1,
    num_eval_envs=128,
    deterministic_eval=False,
    **kwargs
    ):
  
  key = jax.random.PRNGKey(seed=seed)
  
  training_params = load(f"{modelpath}/training_params")

  env = load(f"{modelpath}/env")
  action_repeat = env.action_repeat
  episode_length = env.episode_length
  env = VmapWrapper(EpisodeWrapper(env, episode_length, action_repeat))
  
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
  episode_metrics.update(
      truncation=state_extras['truncation'])

  ordered_goal_difficulty = jp.argsort(episode_metrics['eval/episode_goal_distance_world_frame'][:, -1], axis=0)

  goals = jax.tree_map(lambda x: x[ordered_goal_difficulty, -1, ...], 
                      state_extras['goal'])

  rollouts = jax.tree_map(lambda x: x[ordered_goal_difficulty, ...], 
                      state_extras['state'])

  episode_metrics = jax.tree_map(lambda x: aggregate(x), episode_metrics)


  steps = state_extras['steps'][0,:]
  episode_metrics = jax.tree_map(lambda x: aggregate(x), episode_metrics)

  return {
    'steps': steps,
    'episode_metrics': episode_metrics,
    'final_metrics': metrics,
    'rollouts': rollouts,
    'goals': goals
  }

if __name__== '__main__':
  app.run(main)


