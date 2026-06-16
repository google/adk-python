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

import google.api_core.client_info
from google.auth.credentials import Credentials
from google.cloud import storage

from ... import version

USER_AGENT = f"adk-gcs-tool google-adk/{version.__version__}"


def _get_client_info() -> google.api_core.client_info.ClientInfo:
  """Get client info."""
  return google.api_core.client_info.ClientInfo(user_agent=USER_AGENT)


_client_cache: dict[tuple[int, str | None], storage.Client] = {}


def get_gcs_client(
    *, credentials: Credentials, project: str | None = None
) -> storage.Client:
  """Get a GCS client."""
  cache_key = (id(credentials), project)

  if cache_key not in _client_cache:
    kwargs = {
        "credentials": credentials,
        "client_info": _get_client_info(),
    }
    if project is not None:
      kwargs["project"] = project

    _client_cache[cache_key] = storage.Client(**kwargs)

  return _client_cache[cache_key]
