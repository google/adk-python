# Copyright 2025 Google LLC
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

from __future__ import annotations

import functools
from typing import Callable
from typing import cast
from typing import TypeVar
from typing import Union

from ._feature_registry import _get_feature_config
from ._feature_registry import _register_feature
from ._feature_registry import FeatureConfig
from ._feature_registry import FeatureName
from ._feature_registry import FeatureStage
from ._feature_registry import is_feature_enabled

T = TypeVar("T", bound=Union[Callable, type])


def _make_feature_decorator(
    *,
    feature_name: FeatureName,
    feature_stage: FeatureStage,
    default_on: bool = False,
) -> Callable[[T], T]:
  """Decorator for experimental features.

  Args:
    feature_name: The name of the feature to decorate.
    feature_stage: The stage of the feature.
    default_on: Whether the feature is enabled by default.

  Returns:
    A decorator that checks if the feature is enabled and raises an error if
    not.
  """
  config = _get_feature_config(feature_name)
  if config is None:
    config = FeatureConfig(feature_stage, default_on=default_on)
    _register_feature(feature_name, config)

  if config.stage != feature_stage:
    raise ValueError(
        f"Feature '{feature_name}' is being defined with stage"
        f" '{feature_stage}', but it was previously registered with stage"
        f" '{config.stage}'. Please ensure the feature is consistently defined."
    )

  def decorator(obj: T) -> T:
    def check_feature_enabled():
      if not is_feature_enabled(feature_name):
        raise RuntimeError(f"Feature {feature_name} is not enabled.")

    if isinstance(obj, type):  # decorating a class
      original_init = obj.__init__

      @functools.wraps(original_init)
      def new_init(*args, **kwargs):
        check_feature_enabled()
        return original_init(*args, **kwargs)

      obj.__init__ = new_init
      return cast(T, obj)
    elif isinstance(obj, Callable):  # decorating a function

      @functools.wraps(obj)
      def wrapper(*args, **kwargs):
        check_feature_enabled()
        return obj(*args, **kwargs)

      return cast(T, wrapper)

    else:
      raise TypeError(
          "@experimental can only be applied to classes or callable objects"
      )

  return decorator


def working_in_progress(feature_name: FeatureName) -> Callable[[T], T]:
  """Decorator for working in progress features."""
  return _make_feature_decorator(
      feature_name=feature_name,
      feature_stage=FeatureStage.WIP,
      default_on=False,
  )


def experimental(feature_name: FeatureName) -> Callable[[T], T]:
  """Decorator for experimental features."""
  return _make_feature_decorator(
      feature_name=feature_name,
      feature_stage=FeatureStage.EXPERIMENTAL,
      default_on=False,
  )


def stable(feature_name: FeatureName) -> Callable[[T], T]:
  """Decorator for stable features."""
  return _make_feature_decorator(
      feature_name=feature_name,
      feature_stage=FeatureStage.STABLE,
      default_on=True,
  )
