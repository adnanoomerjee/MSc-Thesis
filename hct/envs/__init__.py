from hct.envs.ant_test import AntTest
from hct.envs.low_level_env import LowLevelEnv
from hct.envs.mid_level_env import MidLevelEnv
from hct.envs.maze.flat_env import FlatMazeEnv
from hct.envs.hurdles_gaps.flat_env import FlatHurdlesGapsEnv
from hct.envs.hurdles_gaps.hma_env import HMA2HurdlesGapsEnv, HMA3HurdlesGapsEnv
from hct.envs.maze.hma_env import HMA2Maze, HMA3Maze



from hct.envs.test import test

from brax.envs.base import Env

import functools


_envs = {
    'AntTest': AntTest,
    'FlatHurdlesEnv': functools.partial(FlatHurdlesGapsEnv, task='ant_hurdles'),
    'HMA2HurdlesEnv': functools.partial(HMA2HurdlesGapsEnv, task='ant_hurdles'),
    'HMA3HurdlesEnv': functools.partial(HMA3HurdlesGapsEnv, task='ant_hurdles'),
    'FlatGapsEnv': functools.partial(FlatHurdlesGapsEnv, task='ant_gaps'),
    'HMA2GapsEnv': functools.partial(HMA2HurdlesGapsEnv, task='ant_gaps'),
    'HMA3GapsEnv': functools.partial(HMA3HurdlesGapsEnv, task='ant_gaps'),
    'FlatMazeEnv': FlatMazeEnv,
    'HMA2Maze': HMA2Maze,
    'HMA3Maze': HMA3Maze,
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