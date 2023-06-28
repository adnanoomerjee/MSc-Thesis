import functools
from typing import Any, Optional, Protocol, Sequence, Tuple

from hct.training.models import MLP, SNMLP, Transformer
from hct.training.types import PolicyValueNetworks

from brax.training import distribution
from brax.training import types
from brax.training.networks import ActivationFn, FeedForwardNetwork

from flax import linen, struct

import jax
import jax.numpy as jp


def make_mlp_policy_value(
    obs_size: int,
    policy_params_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: ActivationFn = linen.swish,
    obs_mask: Optional[jp.ndarray] = None
  ) -> PolicyValueNetworks:
  """Creates MLP policy and value modules"""
  policy_module = MLP(
      layer_sizes=list(policy_hidden_layer_sizes) + [policy_params_size],
      activation=activation,
      kernel_init=jax.nn.initializers.lecun_uniform())
  
  value_module = MLP(
      layer_sizes=list(value_hidden_layer_sizes) + [1],
      activation=activation,
      kernel_init=jax.nn.initializers.lecun_uniform())
  
  dummy_obs = jp.zeros((1, obs_size))

  def make_policy_network() -> FeedForwardNetwork:
    """Creates a policy network."""

    def apply(processor_params, policy_params, obs, obs_mask):
      obs = preprocess_observations_fn(obs, processor_params)
      return policy_module.apply(policy_params, obs)
    
    return FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), apply=apply)

  def make_value_network() -> FeedForwardNetwork:
    """Creates a value network."""

    def apply(processor_params, policy_params, obs, obs_mask):
      obs = preprocess_observations_fn(obs, processor_params)
      return jp.squeeze(value_module.apply(policy_params, obs), axis=-1)

    return FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_obs), apply=apply)

  return make_policy_network(), make_value_network()


def make_transformer_policy_value(
  obs_size: int,
  policy_params_size: int,
  preprocess_observations_fn: types.PreprocessObservationFn,
  max_num_nodes: int,
  obs_mask: jp.ndarray = None,
  num_layers: int = 3,
  d_model: int = 256,
  num_heads: int = 2,
  dim_feedforward: int = 512,
  dropout_rate: float = 0.1,
  transformer_norm: bool = True,
  condition_decoder: bool = True
) -> PolicyValueNetworks: 
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
    policy_params_size=policy_params_size,
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dim_feedforward=dim_feedforward,
    dropout_rate=dropout_rate,
    transformer_norm=transformer_norm,
    condition_decoder=condition_decoder
  )
  
  policy_module = transformer(
    network_type = 'policy')

  value_module = transformer(
    network_type = 'value')
  
  dummy_obs = jp.zeros((1, max_num_nodes, obs_size))

  def make_policy_network() -> FeedForwardNetwork:
    """Creates a policy network."""

    def apply(processor_params, policy_params, obs, obs_mask):
      obs = preprocess_observations_fn(obs, processor_params)
      return policy_module.apply(policy_params, obs, obs_mask)
    
    return FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), apply=apply)

  def make_value_network() -> FeedForwardNetwork:
    """Creates a value network."""

    def apply(processor_params, policy_params, obs, obs_mask):
      obs = preprocess_observations_fn(obs, processor_params)
      return value_module.apply(policy_params, obs, obs_mask)

    return FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_obs), apply=apply)

  return make_policy_network(), make_value_network()