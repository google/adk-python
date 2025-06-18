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
from google.cloud import bigquery
from google.oauth2.credentials import Credentials

from ... import version

USER_AGENT = f"adk-bigquery-tool google-adk/{version.__version__}"


def get_bigquery_client(
    *, project: str, credentials: Credentials
) -> bigquery.Client:
  """Get a BigQuery client."""

  client_info = google.api_core.client_info.ClientInfo(user_agent=USER_AGENT)

  bigquery_client = bigquery.Client(
      project=project, credentials=credentials, client_info=client_info
  )

  return bigquery_client
