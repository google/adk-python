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

from typing import TYPE_CHECKING

from google.auth.credentials import Credentials

from ... import version
from .._google_sdk import create_client_info as _create_client_info

if TYPE_CHECKING:
  from google.api_core.client_info import ClientInfo
  from google.cloud import storage

USER_AGENT = f"adk-gcs-tool google-adk/{version.__version__}"

_STORAGE_REQUIRED = (
    "GCS tools require google-cloud-storage. "
    "Install it with: pip install 'google-adk[gcp]'."
)


def _get_client_info() -> ClientInfo:
  """Get client info."""
  return _create_client_info(user_agent=USER_AGENT)


def get_gcs_client(
    *, credentials: Credentials, project: str | None = None
) -> storage.Client:
  """Get a GCS client.

  Raises:
    ImportError: If `google-cloud-storage` is not installed.
  """
  try:
    from google.cloud import storage  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    raise ImportError(_STORAGE_REQUIRED) from e

  kwargs = {
      "credentials": credentials,
      "client_info": _get_client_info(),
  }
  if project is not None:
    kwargs["project"] = project

  return storage.Client(**kwargs)
