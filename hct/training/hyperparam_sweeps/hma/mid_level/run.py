import sys
sys.path.append("/nfs/nhome/live/aoomerjee/MSc-Thesis/")

from hct.training.hyperparam_sweeps import *


"""Low level training run parameters"""
env_name = 'MidLevel'
filedir = os.path.dirname(os.path.abspath(__file__))
filepath = f"{filedir}/runs/"

MID_LEVEL_ENV_PARAMETERS = {
    'goal_dist': ['absolute', 'relative'],
    'low_level_modelpath': low_level_models,
    'action_repeat': action_repeats,
    'architecture_configs': architecture_configs
    }

hyperparameter_sweep = make_hyperparameter_sweep(MID_LEVEL_ENV_PARAMETERS, HMA_PRETRAIN_TRAINING_PARAMETERS)

def generate_data_tables():
    return generate_data_tables(filepath, hyperparameter_sweep)

main = make_main(hyperparameter_sweep, env_name, filepath)

if __name__== '__main__':
  app.run(main)