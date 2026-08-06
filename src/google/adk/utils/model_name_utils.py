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

"""Utilities for model name validation and parsing."""

from __future__ import annotations

import re
from typing import Optional
from typing import TYPE_CHECKING

from packaging.version import InvalidVersion
from packaging.version import Version

from .env_utils import is_env_enabled

if TYPE_CHECKING:
  from ..models.llm_request import LlmRequest

_DISABLE_GEMINI_MODEL_ID_CHECK_ENV_VAR = 'ADK_DISABLE_GEMINI_MODEL_ID_CHECK'


def is_gemini_model_id_check_disabled() -> bool:
  """Returns True when Gemini model-id validation should be bypassed.

  This opt-in environment variable is intended for internal usage where model
  ids may not follow the public ``gemini-*`` naming convention.
  """
  return is_env_enabled(_DISABLE_GEMINI_MODEL_ID_CHECK_ENV_VAR)


def _is_managed_agent(llm_request: LlmRequest) -> bool:
  """Whether the request was built by a ManagedAgent."""
  return llm_request._is_managed_agent


def extract_model_name(model_string: str) -> str:
  """Extract the actual model name from either simple or path-based format.

  Args:
    model_string: Either a simple model name like "gemini-2.5-pro" or a
      path-based model name like "projects/.../models/gemini-2.5-flash",
      or a provider-prefixed model name like "gemini/gemini-2.5-flash".

  Returns:
    The extracted model name (e.g., "gemini-2.5-pro")
  """
  # Pattern for path-based model names
  # Need to support both Vertex/Gemini and Apigee model paths.
  path_patterns = (
      r'^projects/[^/]+/locations/[^/]+/publishers/[^/]+/models/(.+)$',
      r'^apigee/(?:[^/]+/)?(?:[^/]+/)?(.+)$',
  )
  # Check against all path-based patterns
  for pattern in path_patterns:
    match = re.match(pattern, model_string)
    if match:
      # Return the captured group (the model name)
      return match.group(1)

  # Handle 'models/' prefixed names like "models/gemini-2.5-pro"
  if model_string.startswith('models/'):
    return model_string[len('models/') :]

  # Malformed 'projects/' path (didn't match the Vertex pattern above); return
  # as-is so the provider-prefix block below doesn't misread it as a Gemini id.
  if model_string.startswith('projects/'):
    return model_string

  # Handle provider-prefixed LiteLLM-compatible names like
  # "gemini/gemini-2.5-flash" or "openrouter/google/gemini-2.5-pro:online".
  # Only Gemini names are extracted; other providers fall through unchanged.
  if '/' in model_string:
    model_name = model_string.rsplit('/', 1)[1]
    if model_name.startswith('gemini-'):
      return model_name

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


def is_gemini_eap_or_2_or_above(model_string: Optional[str]) -> bool:
  """Check if the model is a Gemini EAP or a Gemini 2.0+ model.

  EAP (Early Access Program) Gemini models follow a different naming
  convention (see ``_is_gemini_eap_model``) and do not encode a numeric
  version, so they are checked first. Otherwise the model name is parsed
  as a semantic version and is considered a match when the major version
  is ``>= 2``.

  Args:
    model_string: Either a simple model name or path-based model name

  Returns:
    True if it's a Gemini EAP model or a Gemini 2.0+ model, False otherwise
  """
  if not model_string:
    return False

  if _is_gemini_eap_model(model_string):
    return True

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


def _is_gemini_eap_model(model_string: Optional[str]) -> bool:
  """Check if the model is an Early Access Program (EAP) Gemini model.

  Matches names of the form ``gemini-<variant>-early-exp`` optionally
  followed by a numeric suffix, e.g. ``gemini-flash-early-exp`` or
  ``gemini-flash-early-exp3``. ``<variant>`` is one or more
  alphanumeric/underscore segments separated by ``-`` (e.g. ``flash``,
  ``pro``, ``flash-lite``).

  Args:
    model_string: Either a simple model name or path-based model name.

  Returns:
    True if it matches the EAP naming convention, False otherwise.
  """
  if not model_string:
    return False

  model_name = extract_model_name(model_string)
  return (
      re.match(r'^gemini-[a-z0-9_]+(?:-[a-z0-9_]+)*-early-exp\d*$', model_name)
      is not None
  )


def _is_gemini_3_x_live(model_string: Optional[str]) -> bool:
  """Check if the model is a Gemini 3.x Live model.

  Args:
    model_string: The model name

  Returns:
    True if it's a Gemini 3.x Live model, False otherwise
  """
  if not model_string:
    return False
  model_name = extract_model_name(model_string)
  return (
      model_name.startswith('gemini-3.')
      and '-live' in model_name
      and not is_gemini_3_5_live_translate(model_string)
  )


def is_gemini_3_5_live_translate(model_string: Optional[str]) -> bool:
  """Check if the model is a Gemini 3.5 Live Translate model.

  Args:
    model_string: The model name

  Returns:
    True if it's a Gemini 3.5 Live Translate model, False otherwise
  """
  if not model_string:
    return False
  model_name = extract_model_name(model_string)
  return model_name.startswith('gemini-3.5-live-translate')
