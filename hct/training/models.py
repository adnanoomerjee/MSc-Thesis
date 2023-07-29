# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Network definitions."""

import functools
from typing import Callable, Literal, Optional, Sequence, Tuple

from hct.training.transformer.modules import TransformerEncoder, PositionalEncoding, PositionalEncoding1D

from brax.training.types import PreprocessObservationFn
from brax.training.spectral_norm import SNDense
from brax.training.networks import ActivationFn, FeedForwardNetwork, Initializer

from flax import linen
import jax
import jax.numpy as jp


class MLP(linen.Module):
  """MLP module."""
  layer_sizes: Sequence[int]
  activate_final: bool = False
  bias: bool = True
  dropout_rate: float = 0.0

  @linen.compact
  def __call__(self, 
               data: jp.ndarray,
               action_mask: jp.ndarray = None):
    deterministic = not self.has_rng('dropout')
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = linen.Dense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=jax.nn.initializers.lecun_uniform(),
          use_bias=self.bias)(
              hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = linen.swish(hidden)
      if i != len(self.layer_sizes) - 1:
        hidden = linen.Dropout(
            rate=self.dropout_rate,
            deterministic=deterministic)(hidden)
    hidden = jp.squeeze(hidden, axis=-1) if hidden.shape[-1] == 1 else hidden
    if action_mask is not None:
      hidden = hidden * jp.repeat(action_mask, 2, axis=-1) 
    return hidden, None


class Transformer(linen.Module):
  """Transformer Network"""
  network_type: Literal['policy', 'value']
  policy_params_size: int
  num_layers: int
  d_model: int
  num_heads: int
  dim_feedforward: int
  dropout_rate: float
  transformer_norm: bool 
  condition_decoder: bool 

  @linen.compact
  def __call__(self, 
               data: jp.ndarray, 
               src_mask: jp.ndarray = None,
               action_mask: jp.ndarray = None,
               non_actuator_nodes: jp.ndarray = jp.empty(0, dtype=jp.int32)):
    # (B, L, O) O: observation size
    input_size = data.shape[-1]
    seq_len = data.shape[-2]
    # encoder
    output = linen.Dense(
      self.d_model,
      kernel_init=jax.nn.initializers.uniform(scale=0.1),
      bias_init=linen.initializers.zeros)(
        data) * jp.sqrt(input_size)
    output = PositionalEncoding1D(
      d_model=self.d_model, seq_len=seq_len, dropout_rate=self.dropout_rate)(output)
    output, attn_weights = TransformerEncoder(
      num_layers=self.num_layers,
      norm=linen.LayerNorm if self.transformer_norm else None,
      d_model=self.d_model,
      num_heads=self.num_heads,
      dim_feedforward=self.dim_feedforward,
      dropout_rate=self.dropout_rate)(output, src_mask)
    if self.condition_decoder:
      output = jp.concatenate([output, data], axis=-1)
    # decoder
    if self.network_type == 'policy':
      output = linen.DenseGeneral( # (B, L, P) P: number of distribution parameters
        self.policy_params_size,
        axis = -1, 
        kernel_init=jax.nn.initializers.uniform(scale=0.1),
        bias_init=linen.initializers.zeros)(output)
      if action_mask is not None:
        output = output * jp.repeat(action_mask, 2, axis=-1) 
      output = jp.delete(output, non_actuator_nodes, axis=-2)
    else:
      output = linen.DenseGeneral( # (B, 1) 
        1,
        axis = (-2, -1), 
        kernel_init=jax.nn.initializers.uniform(scale=0.1),
        bias_init=linen.initializers.zeros)(output) 
      output = jp.squeeze(output, axis=-1)
    return output, attn_weights
  

def make_mlp_model(
    obs_size: int,
    policy_params_size: int,
    preprocess_observations_fn: PreprocessObservationFn,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5
  ) -> Tuple[FeedForwardNetwork, FeedForwardNetwork]:
  """Creates MLP policy and value modules"""

  policy_module = MLP(
      layer_sizes=list(policy_hidden_layer_sizes) + [policy_params_size])
  
  value_module = MLP(
      layer_sizes=list(value_hidden_layer_sizes) + [1])
  
  dummy_obs = jp.zeros((1, obs_size))

  def make_policy_network() -> FeedForwardNetwork:
    """Creates a policy network."""

    def apply(processor_params, 
              policy_params, 
              obs, 
              obs_mask, 
              action_mask, 
              non_actuator_nodes, 
              dropout_rng=None):
      obs = preprocess_observations_fn(obs, processor_params)
      if dropout_rng is not None:
        apply = policy_module.apply(policy_params, obs, action_mask, rngs={'dropout': dropout_rng})
      else:
        apply = policy_module.apply(policy_params, obs, action_mask)
      return apply
    
    return FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), apply=apply)

  def make_value_network() -> FeedForwardNetwork:
    """Creates a value network."""

    def apply(processor_params, 
              policy_params, 
              obs, 
              obs_mask, 
              action_mask, 
              non_actuator_nodes, 
              dropout_rng=None):
      obs = preprocess_observations_fn(obs, processor_params)
      if dropout_rng is not None:
        apply = value_module.apply(policy_params, obs, action_mask, rngs={'dropout': dropout_rng})
      else:
        apply = value_module.apply(policy_params, obs, action_mask)
      return apply

    return FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_obs, None), apply=apply)

  return make_policy_network(), make_value_network()

  
def make_transformer_model(
  obs_size: int,
  policy_params_size: int,
  preprocess_observations_fn: PreprocessObservationFn,
  num_nodes: int,
  policy_num_layers: int = 3,
  policy_d_model: int = 256,
  policy_num_heads: int = 2,
  policy_dim_feedforward: int = 512,
  policy_dropout_rate: float = 0.2,
  policy_transformer_norm: bool = True,
  policy_condition_decoder: bool = True,
  value_num_layers: int = 3,
  value_d_model: int = 256,
  value_num_heads: int = 2,
  value_dim_feedforward: int = 512,
  value_dropout_rate: float = 0.2,
  value_transformer_norm: bool = True,
  value_condition_decoder: bool = True
) -> Tuple[FeedForwardNetwork, FeedForwardNetwork]: 
  """Creates Transformer policy/value networks
  Args:
    obs_size: size of an observation (last dim of input)
    policy_params_size: number of params that a policy network should generate
    preprocess_observations_fn: function for preprocessing observations
    max_num_nodes: maximum number of nodes (limbs) that input can take
    obs_mask: input observation mask
    num_layers: number of layers in TransformerEncoder
    d_model: size of an input for TransformerEncoder
    num_heads: number of heads in the multiheadattention
    dim_feedforward: the dimension of the feedforward network model
    dropout_rate: the dropout value
    transformer_norm: whether to use a layer normalization
    condition_decoder: whether to add skip connection from input to TransformerEncoder output
  Returns:
    A Transformer policy and value network
  """

  transformer = functools.partial(
    Transformer,
    policy_params_size=policy_params_size)
  
  policy_module = transformer(
    network_type = 'policy',
    num_layers=policy_num_layers,
    d_model=policy_d_model,
    num_heads=policy_num_heads,
    dim_feedforward=policy_dim_feedforward,
    dropout_rate=policy_dropout_rate,
    transformer_norm=policy_transformer_norm,
    condition_decoder=policy_condition_decoder
  )

  value_module = transformer(
    network_type = 'value',
    num_layers=value_num_layers,
    d_model=value_d_model,
    num_heads=value_num_heads,
    dim_feedforward=value_dim_feedforward,
    dropout_rate=value_dropout_rate,
    transformer_norm=value_transformer_norm,
    condition_decoder=value_condition_decoder
  )
  
  dummy_obs = jp.zeros((1, num_nodes, obs_size))

  def make_policy_network() -> FeedForwardNetwork:
    """Creates a policy network."""

    def apply(processor_params, 
              policy_params, 
              obs, 
              obs_mask, 
              action_mask, 
              non_actuator_nodes, 
              dropout_rng=None):
      obs = preprocess_observations_fn(obs, processor_params)
      if dropout_rng is not None:
        apply = policy_module.apply(
          policy_params, 
          obs, 
          obs_mask, 
          action_mask,
          non_actuator_nodes,
          rngs={'dropout': dropout_rng})
      else:
        apply = policy_module.apply(
          policy_params, 
          obs, 
          obs_mask,
          action_mask,
          non_actuator_nodes)
      return apply
    
    return FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), apply=apply)

  def make_value_network() -> FeedForwardNetwork:
    """Creates a value network."""

    def apply(processor_params, 
              policy_params, 
              obs, 
              obs_mask, 
              action_mask, 
              non_actuator_nodes, 
              dropout_rng=None):
      obs = preprocess_observations_fn(obs, processor_params)
      if dropout_rng is not None:
        apply = value_module.apply(
          policy_params, 
          obs, 
          obs_mask, 
          action_mask,
          non_actuator_nodes,
          rngs={'dropout': dropout_rng})
      else:
        apply = value_module.apply(
          policy_params, 
          obs, 
          obs_mask,
          action_mask,
          non_actuator_nodes)
      return apply

    return FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_obs), apply=apply)

  return make_policy_network(), make_value_network()
  

  





