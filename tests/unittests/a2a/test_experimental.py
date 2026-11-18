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
"""Tests for a2a_experimental decorator and warning suppression."""

import warnings

from google.adk.a2a.experimental import a2a_experimental
import pytest


@pytest.fixture(autouse=True)
def clear_suppression_env_vars(monkeypatch):
  monkeypatch.delenv(
      "ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS", raising=False
  )
  monkeypatch.delenv(
      "ADK_SUPPRESS_A2A_EXPERIMENTAL_FEATURE_WARNINGS", raising=False
  )


def test_a2a_experimental_class_warns_by_default():
  """Classes decorated with @a2a_experimental warn when instantiated."""

  @a2a_experimental
  class A2aExperimentalClass:

    def run(self):
      return "running"

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    feature = A2aExperimentalClass()
    assert feature.run() == "running"
    assert len(w) == 1
    assert "[EXPERIMENTAL] A2aExperimentalClass:" in str(w[0].message)


def test_a2a_experimental_function_warns_by_default():
  """Functions decorated with @a2a_experimental warn when called."""

  @a2a_experimental
  def a2a_experimental_function():
    return "running fn"

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    assert a2a_experimental_function() == "running fn"
    assert len(w) == 1
    assert "[EXPERIMENTAL] a2a_experimental_function:" in str(w[0].message)


def test_a2a_experimental_warning_suppressed_by_general_env_var(monkeypatch):
  """General suppression env var silences @a2a_experimental warnings."""

  @a2a_experimental
  class GeneralSuppressedA2aClass:

    def run(self):
      return "running"

  @a2a_experimental
  def general_suppressed_a2a_function():
    return "running fn"

  monkeypatch.setenv("ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS", "yes")

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    feature = GeneralSuppressedA2aClass()
    assert feature.run() == "running"
    assert general_suppressed_a2a_function() == "running fn"
    assert not w


def test_a2a_experimental_warning_suppressed_by_a2a_env_var(monkeypatch):
  """A2A-specific suppression env var silences @a2a_experimental warnings."""

  @a2a_experimental
  class A2aSuppressedClass:

    def run(self):
      return "running"

  @a2a_experimental
  def a2a_suppressed_function():
    return "running fn"

  monkeypatch.setenv("ADK_SUPPRESS_A2A_EXPERIMENTAL_FEATURE_WARNINGS", "on")

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    feature = A2aSuppressedClass()
    assert feature.run() == "running"
    assert a2a_suppressed_function() == "running fn"
    assert not w
