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

"""Utilities for mTLS regional endpoint resolution."""

from __future__ import annotations

import enum
import os

from google.auth.transport import mtls


class MtlsEndpoint(enum.Enum):
  """Enum for the mTLS endpoint setting."""

  AUTO = "auto"
  ALWAYS = "always"
  NEVER = "never"


def use_client_cert_effective() -> bool:
  """Returns whether client certificate should be used for mTLS."""
  try:
    return mtls.should_use_client_cert()
  except (ImportError, AttributeError):
    return (
        os.getenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false").lower()
        == "true"
    )


def get_api_endpoint(
    location: str, default_template: str, mtls_template: str
) -> str:
  """Returns API endpoint based on mTLS configuration and cert availability.

  Args:
      location: The region location.
      default_template: Template for default regional endpoint (e.g.
        "secretmanager.{location}.rep.googleapis.com").
      mtls_template: Template for mTLS regional endpoint (e.g.
        "secretmanager.{location}.rep.mtls.googleapis.com").
  """
  use_mtls_endpoint_str = os.getenv(
      "GOOGLE_API_USE_MTLS_ENDPOINT", MtlsEndpoint.AUTO.value
  ).lower()
  try:
    use_mtls_endpoint = MtlsEndpoint(use_mtls_endpoint_str)
  except ValueError:
    use_mtls_endpoint = MtlsEndpoint.AUTO

  if (use_mtls_endpoint == MtlsEndpoint.ALWAYS) or (
      use_mtls_endpoint == MtlsEndpoint.AUTO and use_client_cert_effective()
  ):
    return mtls_template.format(location=location)
  return default_template.format(location=location)
