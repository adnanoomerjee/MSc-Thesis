from hct.envs.ant_test import AntTest
from hct.envs.low_level_env import LowLevelEnv
from hct.envs.mid_level_env import MidLevelEnv
from hct.envs.flat_env import FlatEnv

from hct.envs.test import test

from brax.envs.base import Env


_envs = {
    'AntTest': AntTest,
    'FlatEnv': FlatEnv,
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