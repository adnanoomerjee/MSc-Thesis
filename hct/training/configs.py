"""General configs"""
import functools
from typing import Literal

from flax import struct, linen


"""Default network configurations"""
DEFAULT_MLP_CONFIGS = {
    'policy_hidden_layer_sizes': (32,) * 4,
    'value_hidden_layer_sizes': (256,) * 5
}

LARGE_MLP_CONFIGS = {
    'policy_hidden_layer_sizes': (64,) * 4,
    'value_hidden_layer_sizes': (512,) * 5
}

VLARGE_MLP_CONFIGS = {
    'policy_hidden_layer_sizes': (128,) * 4,
    'value_hidden_layer_sizes': (1024,) * 5
}

VVLARGE_MLP_CONFIGS = {
    'policy_hidden_layer_sizes': (256,) * 4,
    'value_hidden_layer_sizes': (2048,) * 5
}

MAX_MLP_CONFIGS = {
    'policy_hidden_layer_sizes': (1024,) * 4,
    'value_hidden_layer_sizes': (1024,) * 5
}

DEFAULT_TRANSFORMER_CONFIGS = {
    'policy_num_layers': 4,
    'policy_d_model': 64,
    'policy_num_heads': 2,
    'policy_dim_feedforward': 128,
    'policy_dropout_rate': 0.1,
    'policy_transformer_norm': True,
    'policy_condition_decoder': True,
    'value_num_layers': 4,
    'value_d_model': 128,
    'value_num_heads': 2,
    'value_dim_feedforward': 256,
    'value_dropout_rate': 0.1,
    'value_transformer_norm': True,
    'value_condition_decoder': True
}

TEST_TRAINING_PARAMETERS = {
    'num_timesteps':10_000_000, 
    'num_envs':2, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.99, # trial 0.97 vs 0.99, 0.97 better
    'gradient_clipping':0, # trial 1.0 vs 1000000, 1.0 better
    'seed':5,
    'unroll_length':5,
    'batch_size':4, 
    'num_minibatches':2,
    'num_updates_per_batch':4,
    'num_evals':3, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True
}

DEFAULT_TRANSFORMER_CONFIGS_ALTERNATE = {
    'policy_num_layers': 4,
    'policy_d_model': 16,
    'policy_num_heads': 2,
    'policy_dim_feedforward': 32,
    'policy_dropout_rate': 0.0,
    'policy_transformer_norm': True,
    'policy_condition_decoder': True,
    'value_num_layers': 4,
    'value_d_model': 32,
    'value_num_heads': 2,
    'value_dim_feedforward': 64,
    'value_dropout_rate': 0.0,
    'value_transformer_norm': True,
    'value_condition_decoder': True
}


VVSMALL_TRANSFORMER_CONFIGS = {
    'policy_num_layers': 4,
    'policy_d_model': 8,
    'policy_num_heads': 2,
    'policy_dim_feedforward': 32,
    'policy_dropout_rate': 0.0,
    'policy_transformer_norm': True,
    'policy_condition_decoder': True,
    'value_num_layers': 4,
    'value_d_model': 16,
    'value_num_heads': 2,
    'value_dim_feedforward': 32,
    'value_dropout_rate': 0.0,
    'value_transformer_norm': True,
    'value_condition_decoder': True
}

VSMALL_TRANSFORMER_CONFIGS = {
    'num_layers': 2,
    'd_model': 32,
    'num_heads': 2,
    'dim_feedforward': 64,
    'dropout_rate': 0.0,
    'transformer_norm': True,
    'condition_decoder': True
}

SMALL_TRANSFORMER_CONFIGS = {
    'num_layers': 3,
    'd_model': 64,
    'num_heads': 2,
    'dim_feedforward': 128,
    'dropout_rate': 0.1,
    'transformer_norm': True,
    'condition_decoder': True
}

MID_TRANSFORMER_CONFIGS = {
    'num_layers': 2,
    'd_model': 128,
    'num_heads': 2,
    'dim_feedforward': 256,
    'dropout_rate': 0.1,
    'transformer_norm': True,
    'condition_decoder': True
}

MID2_TRANSFORMER_CONFIGS = {
    'num_layers': 8,
    'd_model': 64,
    'num_heads': 2,
    'dim_feedforward': 128,
    'dropout_rate': 0.1,
    'transformer_norm': True,
    'condition_decoder': True
}

MID3_TRANSFORMER_CONFIGS = {
    'num_layers': 8,
    'd_model': 128,
    'num_heads': 2,
    'dim_feedforward': 256,
    'dropout_rate': 0.1,
    'transformer_norm': True,
    'condition_decoder': True
}

LARGE_TRANSFORMER_CONFIGS = {
    'num_layers': 2,
    'd_model': 128,
    'num_heads': 2,
    'dim_feedforward': 256,
    'dropout_rate': 0.1,
    'transformer_norm': True,
    'condition_decoder': True
}
    
@struct.dataclass
class NetworkArchitecture:
    name: str
    configs: dict

    @classmethod
    def create(cls, name: Literal['MLP', 'Transformer'], **new_configs):
        if 'policy_d_model' in new_configs.keys():
            name = 'Transformer'
        else:
            name = 'MLP'
        if name == 'MLP':
            configs = DEFAULT_MLP_CONFIGS.copy()
        elif name == 'Transformer':
            configs = DEFAULT_TRANSFORMER_CONFIGS.copy()
        else:
            raise ValueError(f"Unsupported architecture name: {name}. Choose either 'MLP' or 'Transformer'.")
        configs.update(new_configs)
        return NetworkArchitecture(name=name, configs=configs)
        

