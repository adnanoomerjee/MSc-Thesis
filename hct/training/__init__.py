
import functools

from hct.training import train as ppo

from brax.envs.base import Env

"""Default training configurations"""
train_fn = functools.partial(ppo.train,  
    num_timesteps=50_000_000, 
    episode_length=1000, 
    action_repeat=1, 
    num_envs=1024, 
    max_devices_per_host=None,
    learning_rate=3e-4, 
    entropy_cost=1e-2,
    discounting=0.97, 
    seed=1,
    unroll_length=5,
    batch_size=2048, 
    num_minibatches=32,
    num_updates_per_batch=4,
    num_evals=512, 
    normalize_observations=True,
    reward_scaling=10,
    clipping_epsilon=.3,
    gae_lambda=.95,
    deterministic_eval=False,
    normalize_advantage=True
)


def get_train_fn(env: Env, **kwargs):
  """Returns a train function.

  Args:
    env: environment
    **kwargs: keyword arguments that get passed to the Env class constructor

  Returns:
    train_fn
  """
  return functools.partial(train_fn, environment=env, **kwargs)