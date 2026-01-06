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

import os
import warnings

from google.adk.features._feature_registry import _FEATURE_OVERRIDES
from google.adk.features._feature_registry import _FEATURE_REGISTRY
from google.adk.features._feature_registry import _get_feature_config
from google.adk.features._feature_registry import _register_feature
from google.adk.features._feature_registry import _WARNED_FEATURES
from google.adk.features._feature_registry import FeatureConfig
from google.adk.features._feature_registry import FeatureStage
from google.adk.features._feature_registry import is_feature_enabled
from google.adk.features._feature_registry import override_feature_enabled
import pytest

FEATURE_CONFIG_WIP = FeatureConfig(FeatureStage.WIP, default_on=False)
FEATURE_CONFIG_EXPERIMENTAL_DISABLED = FeatureConfig(
    FeatureStage.EXPERIMENTAL, default_on=False
)
FEATURE_CONFIG_EXPERIMENTAL_ENABLED = FeatureConfig(
    FeatureStage.EXPERIMENTAL, default_on=True
)
FEATURE_CONFIG_STABLE = FeatureConfig(FeatureStage.STABLE, default_on=True)


@pytest.fixture(autouse=True)
def reset_env_and_registry(monkeypatch):
  """Reset environment variables, registry and overrides before each test."""
  # Clean up environment variables
  for key in list(os.environ.keys()):
    if key.startswith("ADK_ENABLE_") or key.startswith("ADK_DISABLE_"):
      monkeypatch.delenv(key, raising=False)

  # Reset warned features set
  _WARNED_FEATURES.clear()

  # Reset feature overrides
  _FEATURE_OVERRIDES.clear()

  yield

  # Reset warned features set
  _WARNED_FEATURES.clear()

  # Reset feature overrides
  _FEATURE_OVERRIDES.clear()


class TestGetFeatureConfig:
  """Tests for get_feature_config() function."""

  def test_feature_in_registry(self):
    """Returns correct config for features in registry."""
    _register_feature("MY_FEATURE", FEATURE_CONFIG_EXPERIMENTAL_ENABLED)
    assert (
        _get_feature_config("MY_FEATURE") == FEATURE_CONFIG_EXPERIMENTAL_ENABLED
    )

  def test_feature_not_in_registry(self):
    """Returns EXPERIMENTAL_DISABLED for features not in registry."""
    assert _get_feature_config("UNKNOWN_FEATURE") is None


class TestIsFeatureEnabled:
  """Tests for is_feature_enabled() runtime check function."""

  def test_not_in_registry_raises_value_error(self):
    """Features not in registry raise ValueError when checked."""
    with pytest.raises(ValueError):
      is_feature_enabled("NEW_FEATURE")

  def test_wip_feature_disabled(self):
    """WIP features are disabled by default."""
    _register_feature("WIP_FEATURE", FEATURE_CONFIG_WIP)
    with warnings.catch_warnings(record=True) as w:
      assert not is_feature_enabled("WIP_FEATURE")
      assert not w

  def test_wip_feature_enabled(self):
    """WIP features are disabled by default."""
    _register_feature(
        "WIP_FEATURE", FeatureConfig(FeatureStage.WIP, default_on=True)
    )
    with warnings.catch_warnings(record=True) as w:
      assert is_feature_enabled("WIP_FEATURE")
      assert len(w) == 1
      assert "[WIP] feature WIP_FEATURE is enabled." in str(w[0].message)

  def test_experimental_disabled_feature(self):
    """Experimental disabled features are disabled."""
    _register_feature("EXP_DISABLED", FEATURE_CONFIG_EXPERIMENTAL_DISABLED)
    with warnings.catch_warnings(record=True) as w:
      assert not is_feature_enabled("EXP_DISABLED")
      assert not w

  def test_experimental_enabled_feature(self):
    """Experimental enabled features are enabled."""
    _register_feature("EXP_ENABLED", FEATURE_CONFIG_EXPERIMENTAL_ENABLED)
    with warnings.catch_warnings(record=True) as w:
      assert is_feature_enabled("EXP_ENABLED")
      assert len(w) == 1
      assert "[EXPERIMENTAL] feature EXP_ENABLED is enabled." in str(
          w[0].message
      )

  def test_stable_feature_enabled(self):
    """Stable features are enabled."""
    _register_feature("STABLE_FEATURE", FEATURE_CONFIG_STABLE)
    with warnings.catch_warnings(record=True) as w:
      assert is_feature_enabled("STABLE_FEATURE")
      assert not w

  def test_enable_env_var_takes_precedence(self, monkeypatch):
    """ADK_ENABLE_<FEATURE> takes precedence over registry."""
    # Feature disabled in registry
    _register_feature("DISABLED_FEATURE", FEATURE_CONFIG_EXPERIMENTAL_DISABLED)

    # But enabled via env var
    monkeypatch.setenv("ADK_ENABLE_DISABLED_FEATURE", "true")

    with warnings.catch_warnings(record=True) as w:
      assert is_feature_enabled("DISABLED_FEATURE")
      assert len(w) == 1
      assert "[EXPERIMENTAL] feature DISABLED_FEATURE is enabled." in str(
          w[0].message
      )

  def test_disable_env_var_takes_precedence(self, monkeypatch):
    """ADK_DISABLE_<FEATURE> takes precedence over registry."""
    # Feature enabled in registry
    _register_feature("ENABLED_FEATURE", FEATURE_CONFIG_STABLE)

    # But disabled via env var
    monkeypatch.setenv("ADK_DISABLE_ENABLED_FEATURE", "true")

    with warnings.catch_warnings(record=True) as w:
      assert not is_feature_enabled("ENABLED_FEATURE")
      assert not w

  def test_warn_once_per_feature(self, monkeypatch):
    """Warn once per feature, even if being used multiple times."""
    # Feature disabled in registry
    _register_feature("DISABLED_FEATURE", FEATURE_CONFIG_EXPERIMENTAL_DISABLED)

    # But enabled via env var
    monkeypatch.setenv("ADK_ENABLE_DISABLED_FEATURE", "true")

    with warnings.catch_warnings(record=True) as w:
      is_feature_enabled("DISABLED_FEATURE")
      is_feature_enabled("DISABLED_FEATURE")
      assert len(w) == 1
      assert "[EXPERIMENTAL] feature DISABLED_FEATURE is enabled." in str(
          w[0].message
      )


class TestOverrideFeatureEnabled:
  """Tests for override_feature_enabled() function."""

  def test_override_not_in_registry_raises_value_error(self):
    """Overriding features not in registry raises ValueError."""
    with pytest.raises(ValueError):
      override_feature_enabled("UNKNOWN_FEATURE", True)

  def test_override_enables_disabled_feature(self):
    """Programmatic override can enable a disabled feature."""
    _register_feature("OVERRIDE_TEST", FEATURE_CONFIG_EXPERIMENTAL_DISABLED)
    assert not is_feature_enabled("OVERRIDE_TEST")

    override_feature_enabled("OVERRIDE_TEST", True)
    with warnings.catch_warnings(record=True) as w:
      assert is_feature_enabled("OVERRIDE_TEST")
      assert len(w) == 1
      assert "[EXPERIMENTAL] feature OVERRIDE_TEST is enabled." in str(
          w[0].message
      )

  def test_override_disables_enabled_feature(self):
    """Programmatic override can disable an enabled feature."""
    _register_feature("OVERRIDE_TEST", FEATURE_CONFIG_EXPERIMENTAL_ENABLED)

    override_feature_enabled("OVERRIDE_TEST", False)
    with warnings.catch_warnings(record=True) as w:
      assert not is_feature_enabled("OVERRIDE_TEST")
      assert not w

  def test_override_takes_precedence_over_env_enable(self, monkeypatch):
    """Programmatic override takes precedence over ADK_ENABLE_* env var."""
    _register_feature("PRIORITY_TEST", FEATURE_CONFIG_EXPERIMENTAL_DISABLED)

    # Set env var to enable
    monkeypatch.setenv("ADK_ENABLE_PRIORITY_TEST", "true")
    assert is_feature_enabled("PRIORITY_TEST")

    # But override to disable
    override_feature_enabled("PRIORITY_TEST", False)

    with warnings.catch_warnings(record=True) as w:
      assert not is_feature_enabled("PRIORITY_TEST")
      assert not w

  def test_override_takes_precedence_over_env_disable(self, monkeypatch):
    """Programmatic override takes precedence over ADK_DISABLE_* env var."""
    _register_feature("PRIORITY_TEST", FEATURE_CONFIG_EXPERIMENTAL_ENABLED)

    # Set env var to disable
    monkeypatch.setenv("ADK_DISABLE_PRIORITY_TEST", "true")
    assert not is_feature_enabled("PRIORITY_TEST")

    # But override to enable
    override_feature_enabled("PRIORITY_TEST", True)

    with warnings.catch_warnings(record=True) as w:
      assert is_feature_enabled("PRIORITY_TEST")
      assert len(w) == 1
      assert "[EXPERIMENTAL] feature PRIORITY_TEST is enabled." in str(
          w[0].message
      )

  def test_override_stable_feature_no_warning(self):
    """Overriding stable features does not emit warnings."""
    _register_feature("STABLE_OVERRIDE", FEATURE_CONFIG_STABLE)

    override_feature_enabled("STABLE_OVERRIDE", True)
    with warnings.catch_warnings(record=True) as w:
      assert is_feature_enabled("STABLE_OVERRIDE")
      assert not w
