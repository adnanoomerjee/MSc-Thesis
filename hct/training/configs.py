"""General configs"""
import functools
from typing import Literal

from flax import struct, linen


"""Default network configurations"""
DEFAULT_MLP_CONFIGS = {
    'policy_hidden_layer_sizes': (32,) * 4,
    'value_hidden_layer_sizes': (256,) * 5
}

DEFAULT_TRANSFORMER_CONFIGS = {
    'policy_num_layers': 2,
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
        if name == 'MLP':
            configs = DEFAULT_MLP_CONFIGS.copy()
        elif name == 'Transformer':
            configs = DEFAULT_TRANSFORMER_CONFIGS.copy()
        else:
            raise ValueError(f"Unsupported architecture name: {name}. Choose either 'MLP' or 'Transformer'.")
        configs.update(new_configs)
        return NetworkArchitecture(name=name, configs=configs)
        

