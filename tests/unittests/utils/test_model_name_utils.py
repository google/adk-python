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

"""Tests for model name utility functions."""

from google.adk.utils.model_name_utils import extract_model_name
from google.adk.utils.model_name_utils import is_gemini_1_model
from google.adk.utils.model_name_utils import is_gemini_2_model
from google.adk.utils.model_name_utils import is_gemini_model


class TestExtractModelName:
  """Test the extract_model_name function."""

  def test_extract_model_name_simple_model(self):
    """Test extraction of simple model names."""
    assert extract_model_name('gemini-2.5-pro') == 'gemini-2.5-pro'
    assert extract_model_name('gemini-1.5-flash') == 'gemini-1.5-flash'
    assert extract_model_name('gemini-1.0-pro') == 'gemini-1.0-pro'
    assert extract_model_name('claude-3-sonnet') == 'claude-3-sonnet'
    assert extract_model_name('gpt-4') == 'gpt-4'

  def test_extract_model_name_path_based_model(self):
    """Test extraction of path-based model names."""
    path_model = 'projects/265104255505/locations/us-central1/publishers/google/models/gemini-2.0-flash-001'
    assert extract_model_name(path_model) == 'gemini-2.0-flash-001'

    path_model_2 = 'projects/12345/locations/us-east1/publishers/google/models/gemini-1.5-pro-preview'
    assert extract_model_name(path_model_2) == 'gemini-1.5-pro-preview'

    path_model_3 = 'projects/test-project/locations/europe-west1/publishers/google/models/claude-3-sonnet'
    assert extract_model_name(path_model_3) == 'claude-3-sonnet'

  def test_extract_model_name_invalid_path(self):
    """Test that invalid path formats return the original string."""
    invalid_paths = [
        'projects/invalid/path/format',
        'invalid/path/format',
        'projects/123/locations/us-central1/models/gemini-2.0-flash',  # missing publishers
        'projects/123/publishers/google/models/gemini-2.0-flash',  # missing locations
        'projects/123/locations/us-central1/publishers/google/gemini-2.0-flash',  # missing models
    ]

    for invalid_path in invalid_paths:
      assert extract_model_name(invalid_path) == invalid_path

  def test_extract_model_name_empty_string(self):
    """Test extraction from empty string."""
    assert extract_model_name('') == ''

  def test_extract_model_name_edge_cases(self):
    """Test edge cases for model name extraction."""
    # Test with unusual but valid path patterns
    path_with_numbers = 'projects/123456789/locations/us-central1/publishers/google/models/gemini-2.0-flash-001'
    assert extract_model_name(path_with_numbers) == 'gemini-2.0-flash-001'

    # Test with hyphens in project/location names
    path_with_hyphens = 'projects/my-test-project/locations/us-central1/publishers/google/models/gemini-1.5-pro'
    assert extract_model_name(path_with_hyphens) == 'gemini-1.5-pro'


class TestIsGeminiModel:
  """Test the is_gemini_model function."""

  def test_is_gemini_model_simple_names(self):
    """Test Gemini model detection with simple model names."""
    assert is_gemini_model('gemini-2.5-pro') is True
    assert is_gemini_model('gemini-1.5-flash') is True
    assert is_gemini_model('gemini-1.0-pro') is True
    assert is_gemini_model('gemini-2.0-flash-001') is True
    assert is_gemini_model('claude-3-sonnet') is False
    assert is_gemini_model('gpt-4') is False
    assert is_gemini_model('llama-2') is False

  def test_is_gemini_model_path_based_names(self):
    """Test Gemini model detection with path-based model names."""
    gemini_path = 'projects/265104255505/locations/us-central1/publishers/google/models/gemini-2.0-flash-001'
    assert is_gemini_model(gemini_path) is True

    gemini_path_2 = 'projects/12345/locations/us-east1/publishers/google/models/gemini-1.5-pro-preview'
    assert is_gemini_model(gemini_path_2) is True

    non_gemini_path = 'projects/265104255505/locations/us-central1/publishers/google/models/claude-3-sonnet'
    assert is_gemini_model(non_gemini_path) is False

  def test_is_gemini_model_edge_cases(self):
    """Test edge cases for Gemini model detection."""
    # Test with None
    assert is_gemini_model(None) is False

    # Test with empty string
    assert is_gemini_model('') is False

    # Test with model names containing gemini but not starting with it
    assert is_gemini_model('my-gemini-model') is False
    assert is_gemini_model('claude-gemini-hybrid') is False

    # Test with model names that have gemini in the middle of the path
    tricky_path = 'projects/265104255505/locations/us-central1/publishers/gemini/models/claude-3-sonnet'
    assert is_gemini_model(tricky_path) is False

    # Test with just "gemini" without dash
    assert is_gemini_model('gemini') is False
    assert is_gemini_model('gemini_1_5_flash') is False

  def test_is_gemini_model_case_sensitivity(self):
    """Test that model detection is case sensitive."""
    assert is_gemini_model('Gemini-2.5-pro') is False
    assert is_gemini_model('GEMINI-2.5-pro') is False
    assert is_gemini_model('gemini-2.5-PRO') is True  # Only the start matters


class TestIsGemini1Model:
  """Test the is_gemini_1_model function."""

  def test_is_gemini_1_model_simple_names(self):
    """Test Gemini 1.x model detection with simple model names."""
    assert is_gemini_1_model('gemini-1.5-flash') is True
    assert is_gemini_1_model('gemini-1.0-pro') is True
    assert is_gemini_1_model('gemini-1.5-pro-preview') is True
    assert is_gemini_1_model('gemini-1.9-experimental') is True
    assert is_gemini_1_model('gemini-2.0-flash') is False
    assert is_gemini_1_model('gemini-2.5-pro') is False
    assert is_gemini_1_model('gemini-10.0-pro') is False  # Only 1.x versions
    assert is_gemini_1_model('claude-3-sonnet') is False

  def test_is_gemini_1_model_path_based_names(self):
    """Test Gemini 1.x model detection with path-based model names."""
    gemini_1_path = 'projects/265104255505/locations/us-central1/publishers/google/models/gemini-1.5-flash-001'
    assert is_gemini_1_model(gemini_1_path) is True

    gemini_1_path_2 = 'projects/12345/locations/us-east1/publishers/google/models/gemini-1.0-pro-preview'
    assert is_gemini_1_model(gemini_1_path_2) is True

    gemini_2_path = 'projects/265104255505/locations/us-central1/publishers/google/models/gemini-2.0-flash-001'
    assert is_gemini_1_model(gemini_2_path) is False

  def test_is_gemini_1_model_edge_cases(self):
    """Test edge cases for Gemini 1.x model detection."""
    # Test with None
    assert is_gemini_1_model(None) is False

    # Test with empty string
    assert is_gemini_1_model('') is False

    # Test with model names containing gemini-1 but not starting with it
    assert is_gemini_1_model('my-gemini-1.5-model') is False
    assert is_gemini_1_model('custom-gemini-1.5-flash') is False

    # Test with invalid versions
    assert is_gemini_1_model('gemini-1') is False  # Missing dot
    assert is_gemini_1_model('gemini-1-pro') is False  # Missing dot
    assert is_gemini_1_model('gemini-1.') is False  # Missing version number


class TestIsGemini2Model:
  """Test the is_gemini_2_model function."""

  def test_is_gemini_2_model_simple_names(self):
    """Test Gemini 2.x model detection with simple model names."""
    assert is_gemini_2_model('gemini-2.0-flash') is True
    assert is_gemini_2_model('gemini-2.5-pro') is True
    assert is_gemini_2_model('gemini-2.0-flash-001') is True
    assert is_gemini_2_model('gemini-2.9-experimental') is True
    assert is_gemini_2_model('gemini-1.5-flash') is False
    assert is_gemini_2_model('gemini-1.0-pro') is False
    assert is_gemini_2_model('gemini-3.0-pro') is False  # Only 2.x versions
    assert is_gemini_2_model('claude-3-sonnet') is False

  def test_is_gemini_2_model_path_based_names(self):
    """Test Gemini 2.x model detection with path-based model names."""
    gemini_2_path = 'projects/265104255505/locations/us-central1/publishers/google/models/gemini-2.0-flash-001'
    assert is_gemini_2_model(gemini_2_path) is True

    gemini_2_path_2 = 'projects/12345/locations/us-east1/publishers/google/models/gemini-2.5-pro-preview'
    assert is_gemini_2_model(gemini_2_path_2) is True

    gemini_1_path = 'projects/265104255505/locations/us-central1/publishers/google/models/gemini-1.5-flash-001'
    assert is_gemini_2_model(gemini_1_path) is False

  def test_is_gemini_2_model_edge_cases(self):
    """Test edge cases for Gemini 2.x model detection."""
    # Test with None
    assert is_gemini_2_model(None) is False

    # Test with empty string
    assert is_gemini_2_model('') is False

    # Test with model names containing gemini-2 but not starting with it
    assert is_gemini_2_model('my-gemini-2.5-model') is False
    assert is_gemini_2_model('custom-gemini-2.0-flash') is False

    # Test with invalid versions
    assert is_gemini_2_model('gemini-2') is False  # Missing dot
    assert is_gemini_2_model('gemini-2-pro') is False  # Missing dot
    assert is_gemini_2_model('gemini-2.') is False  # Missing version number


class TestModelNameUtilsIntegration:
  """Integration tests for model name utilities."""

  def test_model_classification_consistency(self):
    """Test that model classification functions are consistent."""
    test_models = [
        'gemini-1.5-flash',
        'gemini-2.0-flash',
        'gemini-2.5-pro',
        'projects/123/locations/us-central1/publishers/google/models/gemini-1.5-pro',
        'projects/123/locations/us-central1/publishers/google/models/gemini-2.0-flash',
        'claude-3-sonnet',
        'gpt-4',
    ]

    for model in test_models:
      # A model can only be either Gemini 1.x or Gemini 2.x, not both
      if is_gemini_1_model(model):
        assert not is_gemini_2_model(
            model
        ), f'Model {model} classified as both Gemini 1.x and 2.x'
        assert is_gemini_model(
            model
        ), f'Model {model} is Gemini 1.x but not classified as Gemini'

      if is_gemini_2_model(model):
        assert not is_gemini_1_model(
            model
        ), f'Model {model} classified as both Gemini 1.x and 2.x'
        assert is_gemini_model(
            model
        ), f'Model {model} is Gemini 2.x but not classified as Gemini'

      # If it's neither Gemini 1.x nor 2.x, it should not be classified as Gemini
      if not is_gemini_1_model(model) and not is_gemini_2_model(model):
        if model and 'gemini-' not in extract_model_name(model):
          assert not is_gemini_model(
              model
          ), f'Non-Gemini model {model} classified as Gemini'

  def test_path_vs_simple_model_consistency(self):
    """Test that path-based and simple model names are classified consistently."""
    model_pairs = [
        (
            'gemini-1.5-flash',
            'projects/123/locations/us-central1/publishers/google/models/gemini-1.5-flash',
        ),
        (
            'gemini-2.0-flash',
            'projects/123/locations/us-central1/publishers/google/models/gemini-2.0-flash',
        ),
        (
            'gemini-2.5-pro',
            'projects/123/locations/us-central1/publishers/google/models/gemini-2.5-pro',
        ),
        (
            'claude-3-sonnet',
            'projects/123/locations/us-central1/publishers/google/models/claude-3-sonnet',
        ),
    ]

    for simple_model, path_model in model_pairs:
      # Both forms should be classified identically
      assert is_gemini_model(simple_model) == is_gemini_model(path_model), (
          f'Inconsistent Gemini classification for {simple_model} vs'
          f' {path_model}'
      )
      assert is_gemini_1_model(simple_model) == is_gemini_1_model(path_model), (
          f'Inconsistent Gemini 1.x classification for {simple_model} vs'
          f' {path_model}'
      )
      assert is_gemini_2_model(simple_model) == is_gemini_2_model(path_model), (
          f'Inconsistent Gemini 2.x classification for {simple_model} vs'
          f' {path_model}'
      )
