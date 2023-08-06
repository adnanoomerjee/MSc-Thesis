
from typing import Any, Optional, Protocol, Tuple, TypeVar

from brax.training.types import *
from brax.training.networks import FeedForwardNetwork

import jax
import jax.numpy as jp


PolicyValueNetworks = TypeVar('PolicyValueNetworks')

class Transition(NamedTuple):
  """Container for a transition."""
  observation: NestedArray
  action: NestedArray
  reward: NestedArray
  discount: NestedArray
  next_observation: NestedArray
  nstate: NestedArray = None
  extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray

class PolicyValueFactory(Protocol[PolicyValueNetworks]):
  def __call__(
      self,
      obs_size: int,
      policy_params_size: int,
      preprocess_observations_fn: PreprocessObservationFn,
      obs_mask: Optional[jp.ndarray],
      max_num_nodes: Optional[int] = None,
      **network_configs: Any
  ) -> PolicyValueNetworks:
    pass
