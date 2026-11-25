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

from collections.abc import Iterator
from contextlib import contextmanager
import contextvars
import os
import sys

from .. import version

_ADK_LABEL = "google-adk"
_LANGUAGE_LABEL = "gl-python"
_AGENT_ENGINE_TELEMETRY_TAG = "remote_reasoning_engine"
_AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME = "GOOGLE_CLOUD_AGENT_ENGINE_ID"
_CLOUD_RUN_TELEMETRY_TAG = "cloud_run"
# K_SERVICE is set by the Cloud Run runtime for services; CLOUD_RUN_JOB is set
# for Cloud Run jobs. K_SERVICE is also set by Cloud Functions (2nd gen) and by
# Knative on any cluster (including self-hosted Knative outside Google Cloud),
# so those report cloud_run too.
_CLOUD_RUN_TELEMETRY_ENV_VARIABLE_NAMES = ("K_SERVICE", "CLOUD_RUN_JOB")
_K8S_TELEMETRY_TAG = "k8s"
# Set by Kubernetes in every pod. Not GKE-specific (any Kubernetes cluster sets
# it), and there is no GKE-only env var available without a metadata server
# call, so the tag is k8s rather than gke.
_K8S_TELEMETRY_ENV_VARIABLE_NAME = "KUBERNETES_SERVICE_HOST"
# Every env var that selects a runtime tag. Tests clear these.
_RUNTIME_ENV_VARIABLE_NAMES = (
    _AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME,
    *_CLOUD_RUN_TELEMETRY_ENV_VARIABLE_NAMES,
    _K8S_TELEMETRY_ENV_VARIABLE_NAME,
)


EVAL_CLIENT_LABEL = f"google-adk-eval/{version.__version__}"
"""Label used to denote calls emerging to external system as a part of Evals."""

MODEL_CONSULT_CLIENT_LABEL = f"google-adk-model-consult/{version.__version__}"
"""Label used to denote advisor LLM calls made by ModelConsultTool."""

CASCADE_LIVE_CLIENT_LABEL = f"google-adk-cascade-live/{version.__version__}"
"""Label used to denote reasoning LLM calls made by CascadeLive."""


def classifier_client_label(kind: str) -> str:
  """Returns the label for calls made by a classifier, such as "llm"."""
  return f"google-adk-classifier/{version.__version__}+{kind}"


# The ContextVar holds client label collected for the current request.
_LABEL_CONTEXT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_LABEL_CONTEXT", default=None
)


def _detect_runtime_tag() -> str | None:
  """Returns the tag for the hosting runtime, or None if none is detected.

  Only checks whether env vars the platform sets have a non-empty value; the
  values are never sent. Not cached, so in-process env changes are picked up.
  """
  # Most specific first: Agent Engine wins in case its runtime also sets K_*
  # variables, and Knative on a cluster also sets KUBERNETES_SERVICE_HOST.
  if os.environ.get(_AGENT_ENGINE_TELEMETRY_ENV_VARIABLE_NAME):
    return _AGENT_ENGINE_TELEMETRY_TAG
  if any(
      os.environ.get(name) for name in _CLOUD_RUN_TELEMETRY_ENV_VARIABLE_NAMES
  ):
    return _CLOUD_RUN_TELEMETRY_TAG
  if os.environ.get(_K8S_TELEMETRY_ENV_VARIABLE_NAME):
    return _K8S_TELEMETRY_TAG
  return None


def _get_default_labels(
    framework_label: str | None = None, *, exclude_runtime_tag: bool = False
) -> list[str]:
  """Returns a list of labels that are always added.

  Args:
    framework_label: Optional SemVer build-metadata suffix appended to the
      google-adk framework token (e.g. "managed_agent" ->
      "google-adk/<version>+managed_agent"). When provided, it takes precedence
      over the hosting runtime suffix.
    exclude_runtime_tag: Whether to leave out the hosting runtime suffix (Agent
      Engine, Cloud Run or Kubernetes). Pass True for requests that do not go
      to a Google API endpoint.
  """
  framework_token = f"{_ADK_LABEL}/{version.__version__}"

  # TODO: framework_label takes the only suffix slot, so the runtime is lost
  # when it is set (e.g. managed_agent on Cloud Run).
  if framework_label:
    framework_token = f"{framework_token}+{framework_label}"
  elif not exclude_runtime_tag and (runtime_tag := _detect_runtime_tag()):
    framework_token = f"{framework_token}+{runtime_tag}"

  language_label = f"{_LANGUAGE_LABEL}/" + sys.version.split()[0]
  return [framework_token, language_label]


@contextmanager
def client_label_context(client_label: str) -> Iterator[None]:
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


def get_client_labels(
    framework_label: str | None = None, *, exclude_runtime_tag: bool = False
) -> list[str]:
  """Returns the current list of client labels that can be added to HTTP Headers.

  Args:
    framework_label: Optional SemVer build-metadata suffix for the google-adk
      framework token (see _get_default_labels).
    exclude_runtime_tag: Whether to leave out the hosting runtime suffix (see
      _get_default_labels).
  """
  labels = _get_default_labels(
      framework_label=framework_label, exclude_runtime_tag=exclude_runtime_tag
  )
  current_client_label = _LABEL_CONTEXT.get()

  if current_client_label:
    labels.append(current_client_label)

  return labels
