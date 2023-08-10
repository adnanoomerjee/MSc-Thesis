from hct.envs.ant_test import AntTest
from hct.envs.low_level_env import LowLevelEnv
from hct.envs.mid_level_env import MidLevelEnv
from hct.envs.maze.flat_env import FlatMazeEnv
from hct.envs.hurdles_gaps.flat_env import FlatHurdlesGapsEnv
from hct.envs.hurdles_gaps.high_level_env import HighLevelHurdlesGapsEnv
from hct.envs.maze.high_level_env import HighLevelMazeEnv


from hct.envs.test import test

from brax.envs.base import Env

import functools


_envs = {
    'AntTest': AntTest,
    'FlatMazeEnv': FlatMazeEnv,
    'FlatHurdlesEnv': functools.partial(FlatHurdlesGapsEnv, task='ant_hurdles'),
    'FlatGapsEnv': functools.partial(FlatHurdlesGapsEnv, task='ant_gaps'),
    'HMAHurdlesEnv': functools.partial(HighLevelHurdlesGapsEnv, task='ant_hurdles'),
    'HMAGapsEnv': functools.partial(HighLevelHurdlesGapsEnv, task='ant_gaps'),
    'HMAMazeEnv': HighLevelMazeEnv,
    'LowLevel': LowLevelEnv,
    'MidLevel': MidLevelEnv,
}

def get_environment(env_name: str, **kwargs) -> Env:
  """Returns an environment from the environment registry.

  Args:
    env_name: environment name string
    **kwargs: keyword arguments that get passed to the Env class constructor

  Returns:
    env: an environment
  """
  return _envs[env_name](**kwargs)