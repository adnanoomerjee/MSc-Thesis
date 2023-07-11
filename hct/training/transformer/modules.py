"""Transformer Encoder
Reference: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
"""
from typing import Any, Callable, Optional, Sequence, Tuple
from flax import linen
from flax.linen.initializers import lecun_normal, zeros
import jax
import jax.numpy as jp

from brax.training import types
from brax.training.networks import ActivationFn, FeedForwardNetwork
from hct.training.transformer.attention import SelfAttention


class PositionalEncoding(linen.Module):
  """PositionalEncoding module.
  Learnable positional encoding
  """
  d_model: int
  seq_len: int
  dropout_rate: float = 0.1
  kernel_init: Callable[..., Any] = jax.nn.initializers.normal(stddev=1.0)
  deterministic: bool = False if dropout_rate > 0.0 else True

  @linen.compact
  def __call__(self, data: jp.ndarray):
    # (B, L, O)
    pe = self.param(
        'pe',
        self.kernel_init,
        (data.shape[1], self.d_model),
        jp.float32)
    output = data + pe
    output = linen.Dropout(
        rate=self.dropout_rate,
        deterministic=self.deterministic)(output)
    return output

class TransformerEncoderLayer(linen.Module):
  """TransformerEncoderLayer module."""
  d_model: int
  num_heads: int
  dim_feedforward: int
  dropout_rate: float = 0.1
  dtype: Any = jp.float32
  qkv_features: Optional[int] = None
  activation: Callable[[jp.ndarray], jp.ndarray] = linen.relu
  kernel_init: Callable[..., Any] = lecun_normal()
  bias_init: Callable[..., Any] = zeros
  deterministic: bool = False if dropout_rate > 0.0 else True

  @linen.compact
  def __call__(
      self,
      src: jp.ndarray,
      src_mask: Optional[jp.ndarray] = None) -> jp.ndarray:
    src2, attn_weights = SelfAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_features,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=True,
        broadcast_dropout=False,
        dropout_rate=self.dropout_rate)(src, mask=src_mask)
    src = src + linen.Dropout(
        rate=self.dropout_rate,
        deterministic=self.deterministic)(src2)
    src = linen.LayerNorm(dtype=self.dtype)(src)
    src2 = linen.Dense(
        self.dim_feedforward,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(src)
    src2 = self.activation(src2)
    src2 = linen.Dropout(
        rate=self.dropout_rate,
        deterministic=self.deterministic)(src2)
    src2 = linen.Dense(
        self.d_model,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(src2)
    src = src + linen.Dropout(
        rate=self.dropout_rate,
        deterministic=self.deterministic)(src2)
    src = linen.LayerNorm(dtype=self.dtype)(src)
    return src, attn_weights


class TransformerEncoder(linen.Module):
  """TransformerEncoder module."""
  num_layers: int
  d_model: int
  num_heads: int
  dim_feedforward: int
  norm: Optional[Callable[..., Any]] = None
  dropout_rate: float = 0.1
  dtype: Any = jp.float32
  qkv_features: Optional[int] = None
  activation: Callable[[jp.ndarray], jp.ndarray] = linen.relu
  kernel_init: Callable[..., Any] = lecun_normal()
  bias_init: Callable[..., Any] = zeros

  @linen.compact
  def __call__(
      self,
      src: jp.ndarray,
      src_mask: Optional[jp.ndarray] = None) -> jp.ndarray:
      # NOTE: Shape of attn_weights: Batch Size x MAX_JOINTS x MAX_JOINTS
      attn_weights = []
      output = src
      for _ in range(self.num_layers):
          output, attn_weight = TransformerEncoderLayer(
              d_model=self.d_model,
              num_heads=self.num_heads,
              dim_feedforward=self.dim_feedforward,
              dropout_rate=self.dropout_rate,
              dtype=self.dtype,
              qkv_features=self.qkv_features,
              activation=self.activation,
              kernel_init=self.kernel_init,
              bias_init=self.bias_init)(output, src_mask)
          attn_weights.append(attn_weight)
      if self.norm is not None:
          output = self.norm(dtype=self.dtype)(output)
      return output, attn_weights




