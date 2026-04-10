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

"""Shared pytest fixtures for remote trigger integration tests.

Exposes GCP resource references by reading current Terraform state.

Environment variables:
    GCP_PROJECT_ID    : GCP project.
    GCP_REGION        : GCP region (default: ``us-central1``).
    ADK_TERRAFORM_BIN : Path to terraform binary (default: ``terraform``).
    ADK_TERRAFORM_CWD : Directory to run terraform from (default:
    ``tests/remote/terraform``).
    ADK_TERRAFORM_ARGS: Extra arguments for terraform commands.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time

import pytest
import requests

TERRAFORM_DIR = os.path.join(os.path.dirname(__file__), "terraform")


def _get_project_id() -> str | None:
  """Return GCP project ID from env or gcloud config."""
  project = os.environ.get("GCP_PROJECT_ID")
  if project:
    return project
  try:
    result = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None
  except FileNotFoundError:
    return None


def _get_identity_token(audience: str) -> str | None:
  """Fetch an identity token for the given audience via gcloud."""
  try:
    result = subprocess.run(
        [
            "gcloud",
            "auth",
            "print-identity-token",
            f"--audiences={audience}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
  except (FileNotFoundError, subprocess.CalledProcessError):
    return None


# ---------------------------------------------------------------------------
# Infrastructure Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def terraform_outputs():
  """Read Terraform outputs from the current state."""
  project = _get_project_id()
  if not project:
    pytest.skip(
        "GCP_PROJECT_ID not set and no gcloud default project configured"
    )

  tf_bin = os.environ.get("ADK_TERRAFORM_BIN", "terraform")
  tf_cwd = os.environ.get("ADK_TERRAFORM_CWD", TERRAFORM_DIR)
  tf_args = shlex.split(os.environ.get("ADK_TERRAFORM_ARGS", ""))

  try:
    # Build the command using provided overrides
    cmd = [tf_bin] + tf_args + ["output", "-json"]
    result = subprocess.run(
        cmd,
        cwd=tf_cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    raw = json.loads(result.stdout)
    return {k: v["value"] for k, v in raw.items()}
  except (
      subprocess.CalledProcessError,
      FileNotFoundError,
      json.JSONDecodeError,
  ) as e:
    pytest.fail(
        "Failed to read Terraform outputs. Ensure 'terraform apply' has been"
        f" run successfully.\nCommand: {' '.join(cmd)}\nCWD:"
        f" {tf_cwd}\nError: {e}"
    )


@pytest.fixture(scope="session")
def cloud_run_url(terraform_outputs) -> str:
  """Base URL of the deployed Cloud Run service."""
  return terraform_outputs["cloud_run_url"]


@pytest.fixture(scope="session")
def pubsub_topic(terraform_outputs) -> str:
  """Fully qualified Pub/Sub topic name."""
  return terraform_outputs["pubsub_topic"]


@pytest.fixture(scope="session")
def pubsub_topic_short(terraform_outputs) -> str:
  """Short Pub/Sub topic name."""
  return terraform_outputs["pubsub_topic_short"]


@pytest.fixture(scope="session")
def eventarc_topic(terraform_outputs) -> str:
  """Fully qualified Eventarc source topic name."""
  return terraform_outputs["eventarc_topic"]


@pytest.fixture(scope="session")
def eventarc_topic_short(terraform_outputs) -> str:
  """Short Eventarc source topic name."""
  return terraform_outputs["eventarc_topic_short"]


@pytest.fixture(scope="session")
def project_id(terraform_outputs) -> str:
  """GCP project ID."""
  return terraform_outputs["project_id"]


@pytest.fixture(scope="session")
def auth_headers(cloud_run_url) -> dict[str, str]:
  """Authorization headers for direct HTTP calls to Cloud Run."""
  try:
    resp = requests.get(cloud_run_url, timeout=10)
    if resp.status_code != 403:
      return {}
  except requests.RequestException:
    pass

  token = _get_identity_token(cloud_run_url)
  if token:
    return {"Authorization": f"Bearer {token}"}
  return {}
