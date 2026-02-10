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

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Generator
import warnings

from ..utils.env_utils import is_env_enabled


class FeatureName(str, Enum):
  """Feature names."""

  AUTHENTICATED_FUNCTION_TOOL = "AUTHENTICATED_FUNCTION_TOOL"
  BASE_AUTHENTICATED_TOOL = "BASE_AUTHENTICATED_TOOL"
  BIG_QUERY_TOOLSET = "BIG_QUERY_TOOLSET"
  BIG_QUERY_TOOL_CONFIG = "BIG_QUERY_TOOL_CONFIG"
  BIGTABLE_TOOL_SETTINGS = "BIGTABLE_TOOL_SETTINGS"
  BIGTABLE_TOOLSET = "BIGTABLE_TOOLSET"
  COMPUTER_USE = "COMPUTER_USE"
  DATA_AGENT_TOOL_CONFIG = "DATA_AGENT_TOOL_CONFIG"
  DATA_AGENT_TOOLSET = "DATA_AGENT_TOOLSET"
  GOOGLE_CREDENTIALS_CONFIG = "GOOGLE_CREDENTIALS_CONFIG"
  GOOGLE_TOOL = "GOOGLE_TOOL"
  JSON_SCHEMA_FOR_FUNC_DECL = "JSON_SCHEMA_FOR_FUNC_DECL"
  PROGRESSIVE_SSE_STREAMING = "PROGRESSIVE_SSE_STREAMING"
  PUBSUB_TOOL_CONFIG = "PUBSUB_TOOL_CONFIG"
  PUBSUB_TOOLSET = "PUBSUB_TOOLSET"
  SKILL_TOOLSET = "SKILL_TOOLSET"
  SPANNER_TOOLSET = "SPANNER_TOOLSET"
  SPANNER_TOOL_SETTINGS = "SPANNER_TOOL_SETTINGS"
  SPANNER_VECTOR_STORE = "SPANNER_VECTOR_STORE"
  TOOL_CONFIG = "TOOL_CONFIG"
  TOOL_CONFIRMATION = "TOOL_CONFIRMATION"


class FeatureStage(Enum):
  """Feature lifecycle stages.

  Attributes:
    WIP: Work in progress, not functioning completely. ADK internal development
      only.
    EXPERIMENTAL: Feature works but API may change.
    STABLE: Production-ready, no breaking changes without MAJOR version bump.
  """

  WIP = "wip"
  EXPERIMENTAL = "experimental"
  STABLE = "stable"


@dataclass
class FeatureConfig:
  """Feature configuration.

  Attributes:
    stage: The feature stage.
    default_on: Whether the feature is enabled by default.
  """

  stage: FeatureStage
  default_on: bool = False


# Central registry: FeatureName -> FeatureConfig
_FEATURE_REGISTRY: dict[FeatureName, FeatureConfig] = {
    FeatureName.AUTHENTICATED_FUNCTION_TOOL: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.BASE_AUTHENTICATED_TOOL: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.BIG_QUERY_TOOLSET: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.BIG_QUERY_TOOL_CONFIG: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.BIGTABLE_TOOL_SETTINGS: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.BIGTABLE_TOOLSET: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.COMPUTER_USE: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.DATA_AGENT_TOOL_CONFIG: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.DATA_AGENT_TOOLSET: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.GOOGLE_CREDENTIALS_CONFIG: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.GOOGLE_TOOL: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.JSON_SCHEMA_FOR_FUNC_DECL: FeatureConfig(
        FeatureStage.WIP, default_on=False
    ),
    FeatureName.PROGRESSIVE_SSE_STREAMING: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.PUBSUB_TOOL_CONFIG: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.PUBSUB_TOOLSET: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.SKILL_TOOLSET: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.SPANNER_TOOLSET: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.SPANNER_TOOL_SETTINGS: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.SPANNER_VECTOR_STORE: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.TOOL_CONFIG: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
    FeatureName.TOOL_CONFIRMATION: FeatureConfig(
        FeatureStage.EXPERIMENTAL, default_on=True
    ),
}

# Track which experimental features have already warned (warn only once)
_WARNED_FEATURES: set[FeatureName] = set()

# Programmatic overrides (highest priority, checked before env vars)
_FEATURE_OVERRIDES: dict[FeatureName, bool] = {}


def _get_feature_config(
    feature_name: FeatureName,
) -> FeatureConfig | None:
  """Get the stage of a feature from the registry.

  Args:
    feature_name: The feature name.

  Returns:
    The feature config from the registry, or None if not found.
  """
  return _FEATURE_REGISTRY.get(feature_name, None)


def _register_feature(
    feature_name: FeatureName,
    config: FeatureConfig,
) -> None:
  """Register a feature with a specific config.

  Args:
    feature_name: The feature name.
    config: The feature config to register.
  """
  _FEATURE_REGISTRY[feature_name] = config


def override_feature_enabled(
    feature_name: FeatureName,
    enabled: bool,
) -> None:
  """Programmatically override a feature's enabled state.

  This override takes highest priority, superseding environment variables
  and registry defaults. Use this when environment variables are not
  available or practical in your deployment environment.

  Args:
    feature_name: The feature name to override.
    enabled: Whether the feature should be enabled.

  Example:
    ```python
    from google.adk.features import FeatureName, override_feature_enabled

    # Enable a feature programmatically
    override_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, True)
    ```
  """
  config = _get_feature_config(feature_name)
  if config is None:
    raise ValueError(f"Feature {feature_name} is not registered.")
  _FEATURE_OVERRIDES[feature_name] = enabled


def is_feature_enabled(feature_name: FeatureName) -> bool:
  """Check if a feature is enabled at runtime.

  This function is used for runtime behavior gating within stable features.
  It allows you to conditionally enable new behavior based on feature flags.

  Priority order (highest to lowest):
    1. Programmatic overrides (via override_feature_enabled)
    2. Environment variables (ADK_ENABLE_* / ADK_DISABLE_*)
    3. Registry defaults

  Args:
    feature_name: The feature name (e.g., FeatureName.RESUMABILITY).

  Returns:
    True if the feature is enabled, False otherwise.

  Example:
    ```python
    def _execute_agent_loop():
      if is_feature_enabled(FeatureName.RESUMABILITY):
        # New behavior: save checkpoints for resuming
        return _execute_with_checkpoints()
      else:
        # Old behavior: run without checkpointing
        return _execute_standard()
    ```
  """
  config = _get_feature_config(feature_name)
  if config is None:
    raise ValueError(f"Feature {feature_name} is not registered.")

  # Check programmatic overrides first (highest priority)
  if feature_name in _FEATURE_OVERRIDES:
    enabled = _FEATURE_OVERRIDES[feature_name]
    if enabled and config.stage != FeatureStage.STABLE:
      _emit_non_stable_warning_once(feature_name, config.stage)
    return enabled

  # Check environment variables second
  feature_name_str = (
      feature_name.value
      if isinstance(feature_name, FeatureName)
      else feature_name
  )
  enable_var = f"ADK_ENABLE_{feature_name_str}"
  disable_var = f"ADK_DISABLE_{feature_name_str}"
  if is_env_enabled(enable_var):
    if config.stage != FeatureStage.STABLE:
      _emit_non_stable_warning_once(feature_name, config.stage)
    return True
  if is_env_enabled(disable_var):
    return False

  # Fall back to registry config
  if config.stage != FeatureStage.STABLE and config.default_on:
    _emit_non_stable_warning_once(feature_name, config.stage)
  return config.default_on


def _emit_non_stable_warning_once(
    feature_name: FeatureName,
    feature_stage: FeatureStage,
) -> None:
  """Emit a warning for a non-stable feature, but only once per feature.

  Args:
    feature_name: The feature name.
    feature_stage: The feature stage.
  """
  if feature_name not in _WARNED_FEATURES:
    _WARNED_FEATURES.add(feature_name)
    full_message = (
        f"[{feature_stage.name.upper()}] feature {feature_name} is enabled."
    )
    warnings.warn(full_message, category=UserWarning, stacklevel=4)


@contextmanager
def temporary_feature_override(
    feature_name: FeatureName,
    enabled: bool,
) -> Generator[None, None, None]:
  """Temporarily override a feature's enabled state within a context.

  This context manager is useful for testing or temporarily enabling/disabling
  a feature within a specific scope. The original state is restored when the
  context exits.

  Args:
    feature_name: The feature name to override.
    enabled: Whether the feature should be enabled.

  Yields:
    None

  Example:
    ```python
    from google.adk.features import FeatureName, temporary_feature_override

    # Temporarily enable a feature for testing
    with temporary_feature_override(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL, True):
        # Feature is enabled here
        result = some_function_that_checks_feature()
    # Feature is restored to original state here
    ```
  """
  config = _get_feature_config(feature_name)
  if config is None:
    raise ValueError(f"Feature {feature_name} is not registered.")

  # Save the original override state
  had_override = feature_name in _FEATURE_OVERRIDES
  original_value = _FEATURE_OVERRIDES.get(feature_name)

  # Apply the temporary override
  _FEATURE_OVERRIDES[feature_name] = enabled
  try:
    yield
  finally:
    # Restore the original state
    if had_override:
      _FEATURE_OVERRIDES[feature_name] = original_value
    else:
      _FEATURE_OVERRIDES.pop(feature_name, None)
