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

"""Utilities for model name validation and parsing."""

from __future__ import annotations

import re
from typing import Optional

from packaging.version import InvalidVersion
from packaging.version import Version


def extract_model_name(model_string: str) -> str:
  """Extract the actual model name from either simple or path-based format.

  Args:
    model_string: Either a simple model name like "gemini-2.5-pro" or a
      path-based model name like "projects/.../models/gemini-2.0-flash-001"

  Returns:
    The extracted model name (e.g., "gemini-2.5-pro")
  """
  # Pattern for path-based model names
  path_pattern = (
      r'^projects/[^/]+/locations/[^/]+/publishers/[^/]+/models/(.+)$'
  )
  match = re.match(path_pattern, model_string)
  if match:
    return match.group(1)

  # Handle 'models/' prefixed names like "models/gemini-2.5-pro"
  if model_string.startswith('models/'):
    return model_string[len('models/') :]

  # If it's not a path-based model, return as-is (simple model name)
  return model_string


def is_gemini_model(model_string: Optional[str]) -> bool:
  """Check if the model is a Gemini model using regex patterns.

  Args:
    model_string: Either a simple model name or path-based model name

  Returns:
    True if it's a Gemini model, False otherwise
  """
  if not model_string:
    return False

  model_name = extract_model_name(model_string)
  return re.match(r'^gemini-', model_name) is not None


def is_gemini_1_model(model_string: Optional[str]) -> bool:
  """Check if the model is a Gemini 1.x model using regex patterns.

  Args:
    model_string: Either a simple model name or path-based model name

  Returns:
    True if it's a Gemini 1.x model, False otherwise
  """
  if not model_string:
    return False

  model_name = extract_model_name(model_string)
  return re.match(r'^gemini-1\.\d+', model_name) is not None


def is_gemini_2_or_above(model_string: Optional[str]) -> bool:
  """Check if the model is a Gemini 2.0 or newer model using semantic versions.

  Args:
    model_string: Either a simple model name or path-based model name

  Returns:
    True if it's a Gemini 2.0+ model, False otherwise
  """
  if not model_string:
    return False

  model_name = extract_model_name(model_string)
  if not model_name.startswith('gemini-'):
    return False

  version_string = model_name[len('gemini-') :].split('-', 1)[0]
  if not version_string:
    return False

  try:
    parsed_version = Version(version_string)
  except InvalidVersion:
    return False

  return parsed_version.major >= 2
