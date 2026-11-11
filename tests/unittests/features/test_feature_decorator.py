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

import os
import warnings

from google.adk.features._feature_decorator import experimental
from google.adk.features._feature_decorator import stable
from google.adk.features._feature_decorator import working_in_progress
from google.adk.features._feature_registry import _FEATURE_REGISTRY
from google.adk.features._feature_registry import _get_feature_config
from google.adk.features._feature_registry import _register_feature
from google.adk.features._feature_registry import _WARNED_FEATURES
from google.adk.features._feature_registry import FeatureConfig
from google.adk.features._feature_registry import FeatureName
from google.adk.features._feature_registry import FeatureStage
from google.adk.features._feature_registry import temporary_feature_override
import pytest


@working_in_progress("WIP_CLASS")
class IncompleteFeature:

  def run(self):
    return "running"


@working_in_progress("WIP_FUNCTION")
def wip_function():
  return "executing"


@experimental("EXPERIMENTAL_CLASS")
class ExperimentalClass:

  def run(self):
    return "running"


@experimental("EXPERIMENTAL_FUNCTION")
def experimental_function():
  return "executing"


@stable("STABLE_CLASS")
class StableClass:

  def run(self):
    return "running"


@stable("STABLE_FUNCTION")
def stable_function():
  return "executing"


@pytest.fixture(autouse=True)
def reset_env_and_registry(monkeypatch):
  """Reset environment variables and registry before each test."""
  # Clean up environment variables
  for key in list(os.environ.keys()):
    if (
        key.startswith("ADK_ENABLE_")
        or key.startswith("ADK_DISABLE_")
        or key == "ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS"
    ):
      monkeypatch.delenv(key, raising=False)

  # Add an existing feature to the registry
  _register_feature(
      "ENABLED_EXPERIMENTAL_FEATURE",
      FeatureConfig(FeatureStage.EXPERIMENTAL, default_on=True),
  )

  _register_feature(
      "EXPERIMENTAL_FUNCTION",
      FeatureConfig(FeatureStage.EXPERIMENTAL, default_on=True),
  )


def test_working_in_progress_stage_mismatch():
  """Test that working_in_progress is used with a non-WIP stage."""
  try:

    @working_in_progress("ENABLED_EXPERIMENTAL_FEATURE")
    def unused_function():  # pylint: disable=unused-variable
      return "unused"

    assert False, "Expected ValueError to be raised."
  except ValueError as e:
    assert (
        "Feature 'ENABLED_EXPERIMENTAL_FEATURE' is being defined with stage"
        " 'FeatureStage.WIP', but it was previously registered with stage"
        " 'FeatureStage.EXPERIMENTAL'."
        in str(e)
    )


def test_working_in_progress_class_raises_error():
  """Test that WIP class raises RuntimeError by default."""

  try:
    IncompleteFeature()
    assert False, "Expected RuntimeError to be raised."
  except RuntimeError as e:
    assert "Feature WIP_CLASS is not enabled." in str(e)


def test_working_in_progress_class_bypass_with_env_var(monkeypatch):
  """Test that WIP class can be bypassed with env var."""

  monkeypatch.setenv("ADK_ENABLE_WIP_CLASS", "true")

  with warnings.catch_warnings(record=True) as w:
    feature = IncompleteFeature()
    feature.run()
    assert len(w) == 1
    assert "[WIP] feature WIP_CLASS is enabled." in str(w[0].message)


def test_working_in_progress_function_raises_error():
  """Test that WIP function raises RuntimeError by default."""

  try:
    wip_function()
    assert False, "Expected RuntimeError to be raised."
  except RuntimeError as e:
    assert "Feature WIP_FUNCTION is not enabled." in str(e)


def test_working_in_progress_function_bypass_with_env_var(monkeypatch):
  """Test that WIP function can be bypassed with env var."""

  monkeypatch.setenv("ADK_ENABLE_WIP_FUNCTION", "true")

  with warnings.catch_warnings(record=True) as w:
    wip_function()
    assert len(w) == 1
    assert "[WIP] feature WIP_FUNCTION is enabled." in str(w[0].message)


def test_disabled_experimental_class_raises_error():
  """Test that disabled experimental class raises RuntimeError by default."""

  try:
    ExperimentalClass()
    assert False, "Expected RuntimeError to be raised."
  except RuntimeError as e:
    assert "Feature EXPERIMENTAL_CLASS is not enabled." in str(e)


def test_disabled_experimental_class_bypass_with_env_var(monkeypatch):
  """Test that disabled experimental class can be bypassed with env var."""

  monkeypatch.setenv("ADK_ENABLE_EXPERIMENTAL_CLASS", "true")

  with warnings.catch_warnings(record=True) as w:
    feature = ExperimentalClass()
    feature.run()
    assert len(w) == 1
    assert "[EXPERIMENTAL] feature EXPERIMENTAL_CLASS is enabled." in str(
        w[0].message
    )


def test_enabled_experimental_function_does_not_raise_error():
  """Test that enabled experimental function does not raise error."""

  with warnings.catch_warnings(record=True) as w:
    experimental_function()
    assert len(w) == 1
    assert "[EXPERIMENTAL] feature EXPERIMENTAL_FUNCTION is enabled." in str(
        w[0].message
    )


def test_enabled_experimental_function_disabled_by_env_var(monkeypatch):
  """Test that enabled experimental function can be disabled by env var."""

  monkeypatch.setenv("ADK_DISABLE_EXPERIMENTAL_FUNCTION", "true")

  try:
    experimental_function()
    assert False, "Expected RuntimeError to be raised."
  except RuntimeError as e:
    assert "Feature EXPERIMENTAL_FUNCTION is not enabled." in str(e)


def test_stable_class_does_not_raise_error_or_warn():
  """Test that stable class does not raise error or warn."""

  with warnings.catch_warnings(record=True) as w:
    StableClass().run()
    assert not w


def test_stable_function_does_not_raise_error_or_warn():
  """Test that stable function does not raise error or warn."""

  with warnings.catch_warnings(record=True) as w:
    stable_function()
    assert not w


@pytest.mark.parametrize(
    "decorator", [experimental, working_in_progress, stable]
)
def test_bare_decorator_on_class_raises_type_error(decorator):
  """Applying the decorator without a FeatureName must fail at decoration."""
  registry_before = dict(_FEATURE_REGISTRY)

  with pytest.raises(TypeError, match="Did you apply it without arguments"):

    @decorator
    class Unused:  # pylint: disable=unused-variable
      pass

  assert _FEATURE_REGISTRY == registry_before


@pytest.mark.parametrize(
    "decorator", [experimental, working_in_progress, stable]
)
def test_bare_decorator_on_function_raises_type_error(decorator):
  """Applying the decorator without a FeatureName must fail at decoration."""
  with pytest.raises(TypeError, match="takes a FeatureName"):

    @decorator
    def unused():  # pylint: disable=unused-variable
      pass


def test_bare_decorator_error_names_the_decorator_that_was_misused():
  with pytest.raises(
      TypeError, match="@working_in_progress from google.adk.features takes a"
  ):

    @working_in_progress
    def unused():  # pylint: disable=unused-variable
      pass


@pytest.fixture
def real_experimental_feature():
  """A public FeatureName currently at the experimental stage.

  Picked at test time so that graduating any one feature does not break this
  file, and its warned-once state is restored afterwards.
  """
  feature = next(
      (
          name
          for name in FeatureName
          if _get_feature_config(name).stage == FeatureStage.EXPERIMENTAL
      ),
      None,
  )
  if feature is None:
    pytest.skip("No FeatureName is at the experimental stage.")
  already_warned = feature in _WARNED_FEATURES
  _WARNED_FEATURES.discard(feature)
  yield feature
  if already_warned:
    _WARNED_FEATURES.add(feature)
  else:
    _WARNED_FEATURES.discard(feature)


def test_disabled_real_feature_error_names_the_bare_flag(
    real_experimental_feature,
):
  """Messages name a FeatureName as in ADK_ENABLE_<NAME>, not FeatureName.X."""

  @experimental(real_experimental_feature)
  def gated():
    return "executing"

  with temporary_feature_override(real_experimental_feature, False):
    with pytest.raises(RuntimeError) as excinfo:
      gated()

  assert (
      str(excinfo.value)
      == f"Feature {real_experimental_feature.value} is not enabled."
  )


def test_enabled_real_feature_warning_names_the_bare_flag(
    real_experimental_feature,
):
  """Messages name a FeatureName as in ADK_ENABLE_<NAME>, not FeatureName.X."""

  @experimental(real_experimental_feature)
  def gated():
    return "executing"

  with temporary_feature_override(real_experimental_feature, True):
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      gated()

  assert [str(warning.message) for warning in w] == [
      f"[EXPERIMENTAL] feature {real_experimental_feature.value} is enabled."
  ]
