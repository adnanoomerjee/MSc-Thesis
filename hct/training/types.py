
from typing import Any, Protocol, Tuple, TypeVar

from brax.training import types
from brax.training.networks import FeedForwardNetwork

import jax
import jax.numpy as jp

PolicyValueNetworks = TypeVar('PolicyValueNetworks')

class PolicyValueFactory(Protocol[PolicyValueNetworks]):
  def __call__(
      self,
      obs_size: int,
      policy_params_size: int,
      preprocess_observations_fn: types.PreprocessObservationFn,
      **kwargs: Any
  ) -> PolicyValueNetworks:
    pass
