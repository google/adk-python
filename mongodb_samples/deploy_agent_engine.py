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

"""Deploys the mongodb_samples agent (Atlas auto-embedding) to Agent Engine.

Agent Engine packages the app object with cloudpickle; the MongoDB toolset
drops its live clients at pickle time and rebuilds them on the runtime from
`MONGODB_URI` and the ambient Google Cloud environment, so the deployment
needs no code changes — only the environment variables below.

Configuration is read from environment variables:
  MONGODB_URI / MONGODB_DATABASE / MONGODB_COLLECTION: see README.md.
  GOOGLE_CLOUD_PROJECT:    GCP project to deploy into.
  GOOGLE_CLOUD_LOCATION:   Agent Engine region (default "us-central1").
  GOOGLE_CLOUD_STAGING_BUCKET: gs:// bucket Agent Engine stages artifacts in.

Prerequisites:
  pip install "google-cloud-aiplatform[adk,agent_engines]"

Usage (from the repository root):
  set -a && source .env && set +a
  uv run python mongodb_samples/deploy_agent_engine.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# The repository root, so `mongodb_samples` imports as a package and ships as
# one through `extra_packages`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from vertexai import agent_engines  # noqa: E402
import vertexai  # noqa: E402
from google.cloud.aiplatform_v1.types import SecretRef  # noqa: E402

from mongodb_samples.agent import root_agent  # noqa: E402

# google-adk[mongodb] pulls in pymongo; dnspython resolves mongodb+srv URIs.
# The toolset's pickling and connection_string_env_var support are not in a
# released google-adk yet, so the runtime installs the integration branch
# from the fork (see the "Deploying to Agent Engine" section of
# src/google/adk/integrations/mongodb/README.md).
_REQUIREMENTS = [
    "google-adk[mongodb] @ git+https://github.com/theshanbhag/adk-python.git@mongodb-hardening",
    "dnspython",
    "google-cloud-aiplatform[adk,agent_engines]",
]

# The runtime needs the MongoDB configuration; Atlas embeds the queries, so
# no Google embedding credentials are required. GOOGLE_CLOUD_PROJECT and
# GOOGLE_CLOUD_LOCATION are set by Agent Engine itself. MONGODB_URI carries
# the database password, so it is injected from Secret Manager (SecretRef)
# instead of landing in the deployment spec in plaintext — the Reasoning
# Engine service agent needs roles/secretmanager.secretAccessor on the
# secret, and the toolset resolves the value on the runtime from
# `connection_string_env_var` (the pickled package holds only the name).
_ENV_VARS = [
    "MONGODB_DATABASE",
    "MONGODB_COLLECTION",
]
_MONGODB_URI_SECRET = "mongodb-uri"  # Secret Manager secret in the project


def _require_env(name: str) -> str:
  value = os.environ.get(name)
  if not value:
    sys.exit(f"Environment variable {name} is required. See .env.")
  return value


def main() -> None:
  project = _require_env("GOOGLE_CLOUD_PROJECT")
  location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
  staging_bucket = _require_env("GOOGLE_CLOUD_STAGING_BUCKET")

  vertexai.init(
      project=project, location=location, staging_bucket=staging_bucket
  )

  env_vars = {
      name: os.environ[name] for name in _ENV_VARS if name in os.environ
  }
  # The connection string (password inside) is injected from Secret Manager
  # at runtime; the deployment spec and pickled package carry no plaintext.
  env_vars["MONGODB_URI"] = SecretRef(
      secret=_MONGODB_URI_SECRET, version="latest"
  )

  app = agent_engines.AdkApp(agent=root_agent, enable_tracing=True)
  remote_app = agent_engines.create(
      app,
      display_name="mongodb-samples",
      requirements=_REQUIREMENTS,
      extra_packages=[str(_REPO_ROOT / "mongodb_samples")],
      env_vars=env_vars,
  )
  print(f"Deployed to Agent Engine: {remote_app.resource_name}")


if __name__ == "__main__":
  main()
