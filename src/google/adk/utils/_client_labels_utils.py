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

from contextlib import contextmanager
import contextvars
import os
import sys
from typing import List

from .. import version

_ADK_LABEL = "google-adk"
_LANGUAGE_LABEL = "gl-python"
_AGENT_ENGINE_TELEMETRY_TAG = "remote_reasoning_engine"
_AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME = "GOOGLE_CLOUD_AGENT_ENGINE_ID"


EVAL_CLIENT_LABEL = f"google-adk-eval/{version.__version__}"
"""Label used to denote calls emerging to external system as a part of Evals."""

# The ContextVar holds client label collected for the current request.
_LABEL_CONTEXT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_LABEL_CONTEXT", default=None
)


def _get_default_labels() -> List[str]:
  """Returns a list of labels that are always added."""
  framework_label = f"{_ADK_LABEL}/{version.__version__}"

  if os.environ.get(_AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME):
    framework_label = f"{framework_label}+{_AGENT_ENGINE_TELEMETRY_TAG}"

  language_label = f"{_LANGUAGE_LABEL}/" + sys.version.split()[0]
  return [framework_label, language_label]


@contextmanager
def client_label_context(client_label: str):
  """Runs the operation within the context of the given client label."""
  current_client_label = _LABEL_CONTEXT.get()

  if current_client_label is not None:
    raise ValueError(
        "Client label already exists. You can only add one client label."
    )

  token = _LABEL_CONTEXT.set(client_label)

  try:
    yield
  finally:
    # Restore the previous state of the context variable
    _LABEL_CONTEXT.reset(token)


def get_client_labels() -> List[str]:
  """Returns the current list of client labels that can be added to HTTP Headers."""
  labels = _get_default_labels()
  current_client_label = _LABEL_CONTEXT.get()

  if current_client_label:
    labels.append(current_client_label)

  return labels
