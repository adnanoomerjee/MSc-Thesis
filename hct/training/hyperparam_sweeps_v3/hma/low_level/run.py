import sys
sys.path.append("/nfs/nhome/live/aoomerjee/MSc-Thesis/")

from hct.training.hyperparam_sweeps_v3 import *

env_name = 'LowLevel'
filedir = os.path.dirname(os.path.abspath(__file__))
filepath = f"{filedir}/runs/"

"""Low level training run parameters"""

'''

LOW_LEVEL_ENV_PARAMETERS = {
    'position_goals': (True, False),
    'velocity_goals': (None, 'root', 'full'),
    'distance_reward': ['relative'],
    'architecture_configs': [LARGE_MLP_CONFIGS, VLARGE_MLP_CONFIGS, VVLARGE_MLP_CONFIGS, MAX_MLP_CONFIGS],
    'goal_importance': [True, False]
    }'''

LOW_LEVEL_ENV_PARAMETERS = {
    'goal_dist': ['relative', 'absolute'],
    'position_goals': [True],
    'velocity_goals': [None],
    'distance_reward': ['relative'],
    'architecture_configs': [VLARGE_MLP_CONFIGS],
    'goal_importance': [None, 'continuous', 'discrete']
    }


def hyperparameter_sweep():

    pos_vel_combinations = [(True, v) for v in LOW_LEVEL_ENV_PARAMETERS['velocity_goals']] + [(False, 'full')]

    # Get other parameters
    other_params = {key: value for key, value in LOW_LEVEL_ENV_PARAMETERS.items() if key not in ['position_goals', 'velocity_goals']}
    other_param_names = other_params.keys()
    other_param_values = other_params.values()

    # Generate all combinations of the other parameters
    other_param_combinations = list(itertools.product(*other_param_values))

    env_parameters = []
    # Combine 'pos_vel_combinations' with 'other_param_combinations'
    for pos_vel in pos_vel_combinations:
        for other in other_param_combinations:
            combined_dict = {'position_goals': pos_vel[0], 'velocity_goals': pos_vel[1]}
            for i, name in enumerate(other_param_names):
                combined_dict[name] = other[i]
            env_parameters.append(combined_dict)

    training_parameters = [HMA_PRETRAIN_TRAINING_PARAMETERS for p in env_parameters]
    
    return env_parameters, training_parameters

def data_tables():
   return generate_data_tables(filepath, hyperparameter_sweep)


main = make_main(hyperparameter_sweep, env_name, filepath)

if __name__== '__main__':
  app.run(main)
