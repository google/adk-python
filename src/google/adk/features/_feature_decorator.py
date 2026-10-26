# Copyright 2026 Google LLC
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

from ._feature_registry import _feature_display_name
from ._feature_registry import _get_feature_config
from ._feature_registry import _register_feature
from ._feature_registry import FeatureConfig
from ._feature_registry import FeatureName
from ._feature_registry import FeatureStage
from ._feature_registry import is_feature_enabled

T = TypeVar("T", bound=Callable[..., object])

# The public decorator for each stage, as callers spell it.
_DECORATOR_NAMES: dict[FeatureStage, str] = {
    FeatureStage.WIP: "working_in_progress",
    FeatureStage.EXPERIMENTAL: "experimental",
    FeatureStage.STABLE: "stable",
}


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

  Raises:
    TypeError: If `feature_name` is not a string. This catches the decorator
      being applied bare (`@experimental` instead of
      `@experimental(FeatureName.X)`), which would otherwise register the
      decorated class or function itself as a feature name and silently
      replace it with the inner decorator.
  """
  if not isinstance(feature_name, str):
    decorator_name = _DECORATOR_NAMES[feature_stage]
    raise TypeError(
        f"@{decorator_name} from google.adk.features takes a FeatureName, e.g."
        f" @{decorator_name}(FeatureName.MY_FEATURE); got"
        f" {type(feature_name).__name__} {feature_name!r}. Did you apply it"
        " without arguments?"
    )
  config = _get_feature_config(feature_name)
  if config is None:
    config = FeatureConfig(feature_stage, default_on=default_on)
    _register_feature(feature_name, config)

  display_name = _feature_display_name(feature_name)
  if config.stage != feature_stage:
    raise ValueError(
        f"Feature '{display_name}' is being defined with stage"
        f" '{feature_stage}', but it was previously registered with stage"
        f" '{config.stage}'. Please ensure the feature is consistently defined."
    )

  def decorator(obj: T) -> T:
    def check_feature_enabled() -> None:
      if not is_feature_enabled(feature_name):
        raise RuntimeError(f"Feature {display_name} is not enabled.")

    if isinstance(obj, type):  # decorating a class
      original_init = cast(Callable[..., None], getattr(obj, "__init__"))

      @functools.wraps(original_init)
      def new_init(*args: object, **kwargs: object) -> None:
        check_feature_enabled()
        original_init(*args, **kwargs)

      setattr(obj, "__init__", new_init)
      return obj
    if callable(obj):  # decorating a function

      @functools.wraps(obj)
      def wrapper(*args: object, **kwargs: object) -> object:
        check_feature_enabled()
        return obj(*args, **kwargs)

      return cast(T, wrapper)

    raise TypeError(
        f"@{_DECORATOR_NAMES[feature_stage]} can only be applied to classes or"
        " callable objects"
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
