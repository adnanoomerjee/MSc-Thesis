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

from typing import Callable, Literal, Sequence, Tuple

from hct.training.transformer.modules import TransformerEncoder
from hct.training.network_factory import PPONetworks, make_inference_fn

from brax.training.types import Action ,Extra, Observation, PRNGKey
from brax.training.spectral_norm import SNDense
from brax.training.networks import ActivationFn, Initializer

from flax import linen
import jax
import jax.numpy as jp



class MLP(linen.Module):
  """MLP module."""
  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False
  bias: bool = True

  @linen.compact
  def __call__(self, data: jp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = linen.Dense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias)(
              hidden)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        hidden = self.activation(hidden)
    return hidden, None


class Transformer(linen.Module):
  """Low-Level Transformer Network"""
  network_type: Literal['policy', 'value']
  policy_params_size: int
  num_layers: int = 3
  d_model: int = 256
  num_heads: int = 2
  dim_feedforward: int = 512
  dropout_rate: float = 0.1
  transformer_norm: bool = True
  condition_decoder: bool = True

  @linen.compact
  def __call__(self, data: jp.ndarray, src_mask: jp.ndarray = None):
    # (B, L, O) O: observation size
    input_size = data.shape[-1]
    # encoder
    output = linen.Dense(
      self.d_model,
      kernel_init=jax.nn.initializers.uniform(scale=0.1),
      bias_init=linen.initializers.zeros)(
        data) * jp.sqrt(input_size)
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
      #output = output.reshape((data.shape[0], -1))
    else:
      output = linen.DenseGeneral( # (B, 1) 
        1,
        axis = (-2, -1), 
        kernel_init=jax.nn.initializers.uniform(scale=0.1),
        bias_init=linen.initializers.zeros)(output) 
      output = output.reshape((data.shape[0], -1))
    return output, attn_weights
  

  





