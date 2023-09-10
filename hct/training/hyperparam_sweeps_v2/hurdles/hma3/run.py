import sys
sys.path.append("/nfs/nhome/live/aoomerjee/MSc-Thesis/")

from hct.training.hyperparam_sweeps import *

env_name = 'HMA3HurdlesEnv'
env_params = HMA3_PARAMS

filedir = os.path.dirname(os.path.abspath(__file__))
filepath = f"{filedir}/runs/"

hyperparameter_sweep = make_hyperparameter_sweep(env_params, TASK_TRAINING_PARAMETERS)

def data_tables():
   return generate_data_tables(filepath, hyperparameter_sweep)

main = make_main(hyperparameter_sweep, env_name, filepath)

if __name__== '__main__':
  app.run(main)