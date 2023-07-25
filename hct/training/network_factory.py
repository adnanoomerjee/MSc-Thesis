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

"""PPO networks."""

import functools
from typing import Literal, Optional, Tuple

from hct.training.models import make_mlp_model, make_transformer_model
from hct.training import distribution

from brax.envs import Env
from brax.training.types import (
  Action, 
  identity_observation_preprocessor, 
  Extra, 
  Observation, 
  Policy, 
  PolicyParams, 
  PreprocessObservationFn,
  PRNGKey)
from brax.training.networks import FeedForwardNetwork


from flax import linen, struct

import jax
import jax.numpy as jp


@struct.dataclass
class PPONetworks:
  policy_network: FeedForwardNetwork
  value_network: FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(ppo_networks: PPONetworks):
  """Creates params and inference function for the PPO agent."""

  def inference_fn(
        params: PolicyParams,
        train: bool = False,
        obs_mask: jp.ndarray = None,
        action_mask: jp.ndarray = None,
        non_actuator_nodes: jp.ndarray = None,
        deterministic: bool = False
        ) -> Policy:
    
    policy_network = ppo_networks.policy_network
    parametric_action_distribution = ppo_networks.parametric_action_distribution

    def policy(observations: Observation,
               rng: PRNGKey) -> Tuple[Action, Extra]:
      dropout_rng, sample_rng = jax.random.split(rng)
      if train == False:
        dropout_rng = None
      logits, attn_weights = policy_network.apply(
        *params, 
        obs=observations, 
        obs_mask=obs_mask,
        action_mask=action_mask, 
        non_actuator_nodes=non_actuator_nodes, 
        dropout_rng=dropout_rng)
      if deterministic:
        return ppo_networks.parametric_action_distribution.mode(logits), {'attn_weights': attn_weights}
      raw_actions = parametric_action_distribution.sample_no_postprocessing(
          logits, sample_rng)
      log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
      postprocessed_actions = parametric_action_distribution.postprocess(
          raw_actions)
      return postprocessed_actions, {
          'log_prob': log_prob,
          'raw_action': raw_actions,
          'attn_weights': attn_weights
      }

    return policy

  return inference_fn


# Creates a PPONetworks object
def make_ppo_networks(
      env: Env,
      observation_size: int,
      preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor
      ) -> PPONetworks:
  """Make PPO networks with preprocessor."""

  network_architecture = env.network_architecture
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=env.max_actions_per_node, 
      network_architecture=network_architecture.name,
      min_std=0)

  if network_architecture.name == 'MLP':
    policy_network, value_network = make_mlp_model(
        obs_size = observation_size,
        policy_params_size = parametric_action_distribution.param_size,
        preprocess_observations_fn=preprocess_observations_fn,
        **network_architecture.configs)

  elif network_architecture.name == 'Transformer':
    policy_network, value_network = make_transformer_model(
        obs_size = observation_size,
        policy_params_size = parametric_action_distribution.param_size,
        preprocess_observations_fn=preprocess_observations_fn,
        num_nodes=env.num_nodes,
        **network_architecture.configs)
  
  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
  )
      
'''
# Creates a PPONetworks object
def make_ppo_networks(
      observation_size: int,
      action_size: int,
      policy_value_factory: PolicyValueFactory,
      architecture: Literal['MLP', 'Transformer'] = 'Transformer',
      obs_mask: jp.ndarray = None,
      action_mask: jp.ndarray = None,
      non_actuator_nodes: jp.ndarray = None, 
      max_num_nodes: Optional[jp.ndarray] = None,
      preprocess_observations_fn: types.PreprocessObservationFn = types
      .identity_observation_preprocessor,
      **network_configs,
      ) -> PPONetworks:
  """Make PPO networks with preprocessor."""

  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size)
  
  policy_network, value_network = policy_value_factory(
      obs_size = observation_size,
      policy_param_size = parametric_action_distribution.param_size,
      preprocess_observations_fn=preprocess_observations_fn,
      obs_mask=obs_mask,
      max_num_nodes=max_num_nodes
      **network_configs)
  
  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
      action_mask=action_mask,
      non_actuator_nodes=non_actuator_nodes)


"""
    policy_module = TransformerPolicyNetwork(
          policy_params_size=policy_params_size,
          num_layers=num_layers,
          d_model=d_model,
          num_heads=num_heads,
          dim_feedforward=dim_feedforward,
          dropout_rate=dropout_rate,
          transformer_norm=transformer_norm,
          condition_decoder=condition_decoder
        )

    value_module = TransformerValueNetwork(
          num_layers=num_layers,
          d_model=d_model,
          num_heads=num_heads,
          dim_feedforward=dim_feedforward,
          dropout_rate=dropout_rate,
          transformer_norm=transformer_norm,
          condition_decoder=condition_decoder
        )
    
      policy_params_size: int,
      max_num_limb: int,
      obs_size: int,
      preprocess_observations_fn: types.PreprocessObservationFn = types
      .identity_observation_preprocessor,
      num_layers: int = 3,
      d_model: int = 256,
      num_heads: int = 2,
      dim_feedforward: int = 512,
      dropout_rate: float = 0.1,
      transformer_norm: bool = True,
      condition_decoder: bool = True

def make_MLP_modules(
    obs_size: int,
    policy_params_size: int,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: ActivationFn = linen.swish
    ):
  
  policy_module = MLP(
    layer_sizes=list(policy_hidden_layer_sizes) + [policy_params_size],
    activation=activation,
    kernel_init=jax.nn.initializers.lecun_uniform())
  
  value_module = MLP(
      layer_sizes=list(value_hidden_layer_sizes) + [1],
      activation=activation,
      kernel_init=jax.nn.initializers.lecun_uniform())
  
  dummy_obs = jp.zeros((1, obs_size))

  return policy_module, value_module, dummy_obs
def make_inference_fn(ppo_networks: PPONetworks):
  """Creates params and inference function for the PPO agent."""

  def inference_fn(
        params: PolicyParams,
        train: bool = False,
        obs_mask: jp.ndarray = None,
        deterministic: bool = False
        ) -> Policy:
    
    env = ppo_networks.env
    policy_network = ppo_networks.policy_network
    parametric_action_distribution = ppo_networks.parametric_action_distribution
    non_actuator_nodes = env.non_actuator_nodes if hasattr(env, 'non_actuator_nodes') else None # (num_actuators, action_size)
    action_mask = env.action_mask if hasattr(env, 'action_mask') else None # actuator node idx (num_idx,)

    def policy(observations: Observation,
               rng: PRNGKey) -> Tuple[Action, Extra]:
      dropout_rng, sample_rng = jax.random.split(rng)
      if train == False:
        dropout_rng = None
      logits, attn_weights = policy_network.apply(
        *params, 
        obs=observations, 
        obs_mask=obs_mask,
        dropout_rng=dropout_rng)
      if non_actuator_nodes is not None:
        logits = jp.delete(logits, non_actuator_nodes, axis=-2)
      if deterministic:
        return ppo_networks.parametric_action_distribution.mode(logits), {'attn_weights': attn_weights}
      print(logits.shape)
      raw_actions = parametric_action_distribution.sample_no_postprocessing(
          logits, sample_rng)
      print(raw_actions.shape)
      log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
      print(log_prob.shape)
      print(parametric_action_distribution.postprocess(
          raw_actions).shape)
      postprocessed_actions = jp.squeeze(parametric_action_distribution.postprocess(
          raw_actions), axis=-1)
      if action_mask is not None:
        postprocessed_actions = postprocessed_actions * action_mask
      return postprocessed_actions, {
          'log_prob': log_prob,
          'raw_action': raw_actions,
          'attn_weights': attn_weights
      }

    return policy

  return inference_fn
'''