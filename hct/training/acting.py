# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Brax training acting functions."""

import time
from typing import Callable, Sequence, Tuple, Union

from hct.envs.wrappers.training import EvalWrapper

from brax import envs
from hct.training.types import Metrics
from hct.training.types import Policy
from hct.training.types import PolicyParams
from hct.training.types import PRNGKey
from hct.training.types import Transition
from brax.v1 import envs as envs_v1
import jax
import jax.numpy as jp

State = Union[envs.State, envs_v1.State]
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]

def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    save_state: bool = False,
    extra_fields: Sequence[str] = ()
) -> Tuple[State, Transition]:
  """Collect data."""
  actions, policy_extras = policy(env_state.obs, key)
  nstate = env.step(env_state, actions)
  state_extras = {x: nstate.info[x] for x in extra_fields}
  if save_state:
    nstate_var = nstate
  else:
    nstate_var = None
  return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.obs,
      action=actions,
      reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation=nstate.obs,
      nstate=nstate_var,
      extras={
          'policy_extras': policy_extras,
          'state_extras': state_extras
      })


def generate_unroll(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
    save_state: bool = False
) -> Tuple[State, Transition]:
  """Collect trajectories of given unroll_length."""

  @jax.jit
  def f(carry, unused_t):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, transition = actor_step(
        env, state, policy, current_key, extra_fields=extra_fields, save_state=save_state)
    return (nstate, next_key), transition

  (final_state, _), data = jax.lax.scan(
      f, (env_state, key), (), length=unroll_length)
  #if save_state:
  #  rollout = jax.tree_util.tree_map(lambda x: jp.swapaxes(x, 0, 1), data.extras['state_extras']['state'])
  #  data.extras['state_extras'].update(state=rollout)
  return final_state, data


# TODO: Consider moving this to its own file.
class Evaluator:
  """Class to run evaluations."""

  def __init__(self, 
               eval_env: envs.Env,
               eval_policy_fn: Callable[[PolicyParams], Policy], 
               num_eval_envs: int,
               episode_length: int, 
               action_repeat: int, 
               key: PRNGKey,
               extra_fields: Sequence[str] = (),
               save_state: bool = False):
    """Init.

    Args:
      eval_env: Batched environment to run evals on.
      eval_policy_fn: Function returning the policy from the policy parameters.
      num_eval_envs: Each env will run 1 episode in parallel for each eval.
      episode_length: Maximum length of an episode.
      action_repeat: Number of physics steps per env step.
      key: RNG key.
    """
    self._key = key
    self._eval_walltime = 0.
    self._num_eval_envs = num_eval_envs

    eval_env = EvalWrapper(eval_env)

    def generate_eval_unroll(policy_params: PolicyParams,
                             key: PRNGKey) -> State:
      reset_keys = jax.random.split(key, num_eval_envs)
      eval_first_state = eval_env.reset(reset_keys)
      #if save_state:
      #    eval_first_state.info.update(state=jax.tree_map(lambda x: jp.stack([x]*1001, axis=1), eval_first_state.pipeline_state))
      return generate_unroll(
          eval_env,
          eval_first_state,
          eval_policy_fn(policy_params),
          key,
          unroll_length=episode_length // action_repeat,
          extra_fields=extra_fields,
          save_state=save_state)
    
    self._generate_eval_unroll = jax.jit(generate_eval_unroll)
    self._steps_per_unroll = episode_length * num_eval_envs
    
  def mean_std_stderr(self, values, weights=None):
    """
    Compute the weighted standard deviation.

    :param values: jnp.array of data values
    :param weights: jnp.array of weights for the values
    :return: weighted standard deviation
    """
    weighted_mean = jp.average(values, weights=weights)
    std = jp.sqrt(jp.average((values - weighted_mean) ** 2, weights=weights))
    stderr = std/jp.sum(weights) if weights is not None else std/jp.sum(self._num_eval_envs)
    out = {
      'mean': weighted_mean, 
      'std': std, 
      'stderr' :stderr
    }
    return jax.tree_map(lambda a: jp.where(jp.isnan(a).item(), x=0.0, y=a), out)


  def run_evaluation(self,
                     policy_params: PolicyParams,
                     training_metrics: Metrics = {},
                     aggregate_episodes: bool = True) -> Metrics:
    """Run one epoch of evaluation."""
    self._key, unroll_key = jax.random.split(self._key)

    t = time.time()
    eval_state, data = self._generate_eval_unroll(policy_params, unroll_key)
    eval_metrics = eval_state.info['eval_metrics']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t

    '''
    if 'episode_weights' in eval_metrics.episode_metrics.keys():
      mean_weights = eval_metrics.episode_metrics['episode_weights']
    else:
      mean_weights = None'''

    metrics = {
        f'eval/episode_{name}': self.mean_std_stderr(value) if aggregate_episodes and ('cumulative' not in name and 'weight' not in name) else jp.sum(value).item() if aggregate_episodes else value 
        for name, value in eval_metrics.episode_metrics.items()
    }

    metrics['eval/avg_episode_length'] = self.mean_std_stderr(eval_metrics.episode_steps, None)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics
    }

    return metrics, data  # pytype: disable=bad-return-type  # jax-ndarray
