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

from typing import List
from typing import Optional
from typing import Union

import google.api_core.client_info
from google.api_core.gapic_v1 import client_info as gapic_client_info
from google.auth.credentials import Credentials
from google.cloud import bigquery
from google.cloud import dataplex_v1

from ... import version

USER_AGENT_BASE = f"google-adk/{version.__version__}"
BQ_USER_AGENT = f"adk-bigquery-tool {USER_AGENT_BASE}"
DP_USER_AGENT = f"adk-dataplex-tool {USER_AGENT_BASE}"
USER_AGENT = BQ_USER_AGENT


def get_bigquery_client(
    *,
    project: Optional[str],
    credentials: Credentials,
    location: Optional[str] = None,
    user_agent: Optional[Union[str, List[str]]] = None,
) -> bigquery.Client:
  """Get a BigQuery client.

  Args:
    project: The GCP project ID.
    credentials: The credentials to use for the request.
    location: The location of the BigQuery client.
    user_agent: The user agent to use for the request.

  Returns:
    A BigQuery client.
  """

  user_agents = [BQ_USER_AGENT]
  if user_agent:
    if isinstance(user_agent, str):
      user_agents.append(user_agent)
    else:
      user_agents.extend([ua for ua in user_agent if ua])

  client_info = google.api_core.client_info.ClientInfo(
      user_agent=" ".join(user_agents)
  )

  bigquery_client = bigquery.Client(
      project=project,
      credentials=credentials,
      location=location,
      client_info=client_info,
  )

  return bigquery_client


def get_dataplex_catalog_client(
    *,
    credentials: Credentials,
    user_agent: Optional[Union[str, List[str]]] = None,
) -> dataplex_v1.CatalogServiceClient:
  """Get a Dataplex CatalogServiceClient with minimal necessary arguments.

  Args:
    credentials: The credentials to use for the request.
    user_agent: Additional user agent string(s) to append.

  Returns:
    A Dataplex Client.
  """

  user_agents = [DP_USER_AGENT]
  if user_agent:
    if isinstance(user_agent, str):
      user_agents.append(user_agent)
    else:
      user_agents.extend([ua for ua in user_agent if ua])

  client_info = gapic_client_info.ClientInfo(user_agent=" ".join(user_agents))

  return dataplex_v1.CatalogServiceClient(
      credentials=credentials,
      client_info=client_info,
  )
