import functools
from typing import Literal
import itertools

import json

from flax import struct, linen

from hct.training import train as ppo
from hct.training.configs import *

from brax.envs.base import Env

"""Low level training run parameters"""

LOW_LEVEL_ENV_PARAMETERS = {
    'position_goals': (True, False),
    'velocity_goals': (None, 'root', 'full'),
    'reward_goal_reached': (0, 50, 100),
    'unhealthy_cost': (0, -1),
    'air_probability': (0.1, 0.3),
    'distance_reward': ('absolute', 'relative')
    }

LOW_LEVEL_TRAINING_PARAMETERS = {
    'num_timesteps':500_000_000, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.97, # trial 0.97 vs 0.99, 0.97 better
    'gradient_clipping':1.0, # trial 1.0 vs 1000000, 1.0 better
    'seed':5,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':256, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

def hyperparameter_sweep():
    # Special handling for 'position_goals' and 'velocity_goals'
    pos_vel_combinations = [(True, v) for v in LOW_LEVEL_ENV_PARAMETERS['velocity_goals']] + [(False, 'full')]

    # Get other parameters
    other_params = {key: value for key, value in LOW_LEVEL_ENV_PARAMETERS.items() if key not in ['position_goals', 'velocity_goals']}
    other_param_names = other_params.keys()
    other_param_values = other_params.values()

    # Generate all combinations of the other parameters
    other_param_combinations = list(itertools.product(*other_param_values))

    # Combine 'pos_vel_combinations' with 'other_param_combinations'
    env_parameters = []
    for pos_vel in pos_vel_combinations:
        for other in other_param_combinations:
            combined_dict = {'position_goals': pos_vel[0], 'velocity_goals': pos_vel[1], 'architecture_configs': DEFAULT_MLP_CONFIGS}
            for i, name in enumerate(other_param_names):
                combined_dict[name] = other[i]
            env_parameters.append(combined_dict)
    
    training_parameters = [LOW_LEVEL_TRAINING_PARAMETERS for p in env_parameters]
    return env_parameters, training_parameters