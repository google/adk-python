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

import click
from click.testing import CliRunner
from google.adk.cli.cli_tools_click import _apply_feature_overrides
from google.adk.cli.cli_tools_click import feature_options
from google.adk.features._feature_registry import _FEATURE_OVERRIDES
from google.adk.features._feature_registry import _WARNED_FEATURES
from google.adk.features._feature_registry import FeatureName
from google.adk.features._feature_registry import is_feature_enabled
import pytest


@pytest.fixture(autouse=True)
def reset_feature_overrides():
  """Reset feature overrides and warnings before/after each test."""
  _FEATURE_OVERRIDES.clear()
  _WARNED_FEATURES.clear()
  yield
  _FEATURE_OVERRIDES.clear()
  _WARNED_FEATURES.clear()


class TestApplyFeatureOverrides:
  """Tests for _apply_feature_overrides helper function."""

  def test_single_feature(self):
    """Single feature name is applied correctly."""
    _apply_feature_overrides(enable_features=("JSON_SCHEMA_FOR_FUNC_DECL",))
    assert is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)

  def test_comma_separated_features(self):
    """Comma-separated feature names are applied correctly."""
    _apply_feature_overrides(
        enable_features=("JSON_SCHEMA_FOR_FUNC_DECL,PROGRESSIVE_SSE_STREAMING",)
    )
    assert is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)
    assert is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING)

  def test_multiple_flag_values(self):
    """Multiple --enable_features flags are applied correctly."""
    _apply_feature_overrides(
        enable_features=(
            "JSON_SCHEMA_FOR_FUNC_DECL",
            "PROGRESSIVE_SSE_STREAMING",
        )
    )
    assert is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)
    assert is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING)

  def test_whitespace_handling(self):
    """Whitespace around feature names is stripped."""
    _apply_feature_overrides(
        enable_features=(" JSON_SCHEMA_FOR_FUNC_DECL , COMPUTER_USE ",)
    )
    assert is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)
    assert is_feature_enabled(FeatureName.COMPUTER_USE)

  def test_empty_string_ignored(self):
    """Empty strings in the list are ignored."""
    _apply_feature_overrides(enable_features=("",))
    # No error should be raised

  def test_unknown_feature_warns(self, capsys):
    """Unknown feature names emit a warning."""
    _apply_feature_overrides(enable_features=("UNKNOWN_FEATURE_XYZ",))
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "UNKNOWN_FEATURE_XYZ" in captured.err
    assert "Valid names are:" in captured.err

  def test_single_disable_feature(self):
    """Single feature name is disabled correctly."""
    # First enable a feature
    _apply_feature_overrides(enable_features=("JSON_SCHEMA_FOR_FUNC_DECL",))
    assert is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)

    # Then disable it
    _apply_feature_overrides(disable_features=("JSON_SCHEMA_FOR_FUNC_DECL",))
    assert not is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)

  def test_comma_separated_disable_features(self):
    """Comma-separated feature names are disabled correctly."""
    # First enable features
    _apply_feature_overrides(
        enable_features=("JSON_SCHEMA_FOR_FUNC_DECL,PROGRESSIVE_SSE_STREAMING",)
    )

    # Then disable them
    _apply_feature_overrides(
        disable_features=(
            "JSON_SCHEMA_FOR_FUNC_DECL,PROGRESSIVE_SSE_STREAMING",
        )
    )
    assert not is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)
    assert not is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING)

  def test_disable_overrides_enable(self):
    """Disable is applied after enable, so disable wins for same feature."""
    _apply_feature_overrides(
        enable_features=("JSON_SCHEMA_FOR_FUNC_DECL",),
        disable_features=("JSON_SCHEMA_FOR_FUNC_DECL",),
    )
    # disable_features is processed after enable_features
    assert not is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)

  def test_enable_and_disable_different_features(self):
    """Enable and disable can be used together for different features."""
    # First enable a feature that we'll disable
    _apply_feature_overrides(enable_features=("PROGRESSIVE_SSE_STREAMING",))

    _apply_feature_overrides(
        enable_features=("JSON_SCHEMA_FOR_FUNC_DECL",),
        disable_features=("PROGRESSIVE_SSE_STREAMING",),
    )
    assert is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)
    assert not is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING)


class TestFeatureOptionsDecorator:
  """Tests for feature_options decorator."""

  def test_decorator_adds_enable_features_option(self):
    """Decorator adds --enable_features option to command."""

    @click.command()
    @feature_options()
    def test_cmd():
      pass

    runner = CliRunner()
    result = runner.invoke(test_cmd, ["--help"])
    assert "--enable_features" in result.output

  def test_enable_features_applied_before_command(self):
    """Features are enabled before the command function runs."""
    feature_was_enabled = []

    @click.command()
    @feature_options()
    def test_cmd():
      feature_was_enabled.append(
          is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)
      )

    runner = CliRunner()
    runner.invoke(
        test_cmd,
        ["--enable_features=JSON_SCHEMA_FOR_FUNC_DECL"],
        catch_exceptions=False,
    )
    assert feature_was_enabled == [True]

  def test_multiple_enable_features_flags(self):
    """Multiple --enable_features flags work correctly."""
    enabled_features = []

    @click.command()
    @feature_options()
    def test_cmd():
      enabled_features.append(
          is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)
      )
      enabled_features.append(
          is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING)
      )

    runner = CliRunner()
    runner.invoke(
        test_cmd,
        [
            "--enable_features=JSON_SCHEMA_FOR_FUNC_DECL",
            "--enable_features=PROGRESSIVE_SSE_STREAMING",
        ],
        catch_exceptions=False,
    )
    assert enabled_features == [True, True]

  def test_comma_separated_enable_features(self):
    """Comma-separated feature names work correctly."""
    enabled_features = []

    @click.command()
    @feature_options()
    def test_cmd():
      enabled_features.append(
          is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)
      )
      enabled_features.append(
          is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING)
      )

    runner = CliRunner()
    runner.invoke(
        test_cmd,
        [
            "--enable_features=JSON_SCHEMA_FOR_FUNC_DECL,PROGRESSIVE_SSE_STREAMING"
        ],
        catch_exceptions=False,
    )
    assert enabled_features == [True, True]

  def test_no_enable_features_flag(self):
    """Command works without --enable_features flag."""
    enabled_features = []

    @click.command()
    @feature_options()
    def test_cmd():
      enabled_features.append(
          is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)
      )

    runner = CliRunner()
    result = runner.invoke(test_cmd, [], catch_exceptions=False)
    assert result.exit_code == 0
    assert enabled_features == [False]

  def test_preserves_function_metadata(self):
    """Decorator preserves the wrapped function's metadata."""

    @click.command()
    @feature_options()
    def my_test_command():
      """My docstring."""
      pass

    # The callback should have preserved metadata
    assert (
        "my_test_command" in my_test_command.name
        or my_test_command.callback.__name__ == "my_test_command"
    )

  def test_decorator_adds_disable_features_option(self):
    """Decorator adds --disable_features option to command."""

    @click.command()
    @feature_options()
    def test_cmd():
      pass

    runner = CliRunner()
    result = runner.invoke(test_cmd, ["--help"])
    assert "--disable_features" in result.output

  def test_disable_features_applied_before_command(self):
    """Features are disabled before the command function runs."""
    # First enable the feature via override
    _apply_feature_overrides(enable_features=("JSON_SCHEMA_FOR_FUNC_DECL",))

    feature_was_disabled = []

    @click.command()
    @feature_options()
    def test_cmd():
      feature_was_disabled.append(
          not is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)
      )

    runner = CliRunner()
    runner.invoke(
        test_cmd,
        ["--disable_features=JSON_SCHEMA_FOR_FUNC_DECL"],
        catch_exceptions=False,
    )
    assert feature_was_disabled == [True]

  def test_enable_and_disable_together(self):
    """Both --enable_features and --disable_features work together."""
    feature_states = []

    @click.command()
    @feature_options()
    def test_cmd():
      feature_states.append(
          is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL)
      )
      feature_states.append(
          is_feature_enabled(FeatureName.PROGRESSIVE_SSE_STREAMING)
      )

    runner = CliRunner()
    runner.invoke(
        test_cmd,
        [
            "--enable_features=JSON_SCHEMA_FOR_FUNC_DECL",
            "--disable_features=PROGRESSIVE_SSE_STREAMING",
        ],
        catch_exceptions=False,
    )
    # JSON_SCHEMA_FOR_FUNC_DECL should be enabled
    # PROGRESSIVE_SSE_STREAMING should be disabled
    assert feature_states == [True, False]
