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

import google.api_core.client_info
from google.auth.credentials import Credentials
from google.cloud import bigtable
from google.cloud.bigtable import data

from ... import version

USER_AGENT = f"adk-bigtable-tool google-adk/{version.__version__}"


def _get_client_info() -> google.api_core.client_info.ClientInfo:
  """Get client info."""
  return google.api_core.client_info.ClientInfo(user_agent=USER_AGENT)


def get_bigtable_data_client(
    *, project: str, credentials: Credentials
) -> bigtable.BigtableDataClient:
  """Get a Bigtable client."""

  bigtable_data_client = data.BigtableDataClient(
      project=project, credentials=credentials, client_info=_get_client_info()
  )

  return bigtable_data_client


def get_bigtable_admin_client(
    *, project: str, credentials: Credentials
) -> bigtable.Client:
  """Get a Bigtable client."""

  bigtable_admin_client = bigtable.Client(
      project=project,
      admin=True,
      credentials=credentials,
      client_info=_get_client_info(),
  )

  return bigtable_admin_client
