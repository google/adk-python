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

"""Sample agent that uses Secret Manager without exposing secret values."""

from __future__ import annotations

import os
import textwrap

from google.adk import Agent
from google.adk.integrations.secret_manager import SecretManagerClient
import google.auth

# Set to a region (e.g. "europe-north1") to use regional secrets, which live
# under `projects/*/locations/*/secrets/*`. Unset means global secrets.
LOCATION = os.getenv("SECRET_MANAGER_LOCATION") or None

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or google.auth.default()[1]
if not PROJECT_ID:
  raise ValueError(
      "Set GOOGLE_CLOUD_PROJECT, or use credentials that carry a project."
  )

# Only secrets in this allowlist may be accessed by the agent's tools. Pair
# this with secret-level IAM bindings (roles/secretmanager.secretAccessor on
# each secret rather than on the project).
ALLOWED_SECRET_IDS: frozenset[str] = frozenset(
    s.strip()
    for s in (
        os.getenv("ADK_ALLOWED_SECRET_IDS") or "my-api-key,partner-token"
    ).split(",")
    if s.strip()
)

# Server-side registry mapping a logical service name to (secret_id, fixed_url).
# Binding the secret and the destination URL in code prevents a prompt injection
# from sending a secret to a model-supplied URL.
_SERVICES: dict[str, tuple[str, str]] = {
    "partner-health": ("partner-token", "https://api.example.com/health"),
}

_secret_client = SecretManagerClient(location=LOCATION)


def _resource_name(secret_id: str) -> str:
  if LOCATION:
    return (
        f"projects/{PROJECT_ID}/locations/{LOCATION}"
        f"/secrets/{secret_id}/versions/latest"
    )
  return f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"


def _require_allowed_secret(secret_id: str) -> str | None:
  if secret_id not in ALLOWED_SECRET_IDS:
    return (
        f"Secret {secret_id!r} is not in ALLOWED_SECRET_IDS"
        f" ({sorted(ALLOWED_SECRET_IDS)})."
    )
  return None


def check_secret(secret_id: str) -> dict[str, object]:
  """Checks that an allowlisted secret exists and is readable.

  Args:
    secret_id: The short secret ID (for example `my-api-key`).

  Returns:
    A dict with the verification status and resource name.
  """
  if err := _require_allowed_secret(secret_id):
    return {"status": "error", "error": err}

  resource_name = _resource_name(secret_id)
  try:
    _secret_client.get_secret(resource_name)
  except Exception as e:  # pylint: disable=broad-exception-caught
    return {
        "status": "error",
        "resource_name": resource_name,
        "error": f"{type(e).__name__}: {e}",
    }
  return {
      "status": "ok",
      "resource_name": resource_name,
  }


def call_service_with_secret(service_name: str) -> dict[str, object]:
  """Calls a registered external service using its configured secret.

  Args:
    service_name: Name of the registered service (for example `partner-health`).

  Returns:
    A dict with the call status, target URL, and secret resource name.
  """
  if service_name not in _SERVICES:
    return {
        "status": "error",
        "error": (
            f"Unknown service {service_name!r}. Available: {sorted(_SERVICES)}."
        ),
    }

  secret_id, service_url = _SERVICES[service_name]
  if err := _require_allowed_secret(secret_id):
    return {"status": "error", "error": err}

  resource_name = _resource_name(secret_id)
  try:
    api_key = _secret_client.get_secret(resource_name)
  except Exception as e:  # pylint: disable=broad-exception-caught
    return {
        "status": "error",
        "service": service_name,
        "secret_resource": resource_name,
        "error": f"{type(e).__name__}: {e}",
    }

  # In a real tool, call `service_url` with `api_key`, e.g.:
  #   httpx.get(service_url, headers={"Authorization": f"Bearer {api_key}"})
  del api_key
  return {
      "status": "dry_run",
      "service": service_name,
      "url": service_url,
      "secret_resource": resource_name,
      "note": (
          "Placeholder only: credential was resolved in Python; no outbound"
          " HTTP request was made."
      ),
  }


root_agent = Agent(
    name="secret_manager_agent",
    description=(
        "Agent that verifies and uses Google Cloud Secret Manager secrets"
        " without exposing their values."
    ),
    instruction=textwrap.dedent("""\
        You help operators verify that allowlisted secrets stored in Google
        Cloud Secret Manager are readable and invoke registered services that
        use those secrets.

        Never ask for, guess, or repeat secret values. Tools only return
        readability status and service dry-run statuses; relay those
        accurately.
    """),
    tools=[check_secret, call_service_with_secret],
)
