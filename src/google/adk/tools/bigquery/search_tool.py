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

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from google.api_core import exceptions as api_exceptions
from google.auth.credentials import Credentials
from google.cloud import dataplex_v1

from . import client
from .config import BigQueryToolConfig


def _construct_search_query_helper(
    predicate: str, operator: str, items: List[str]
) -> str:
  if not items:
    return ""
  if len(items) == 1:
    return f'{predicate}{operator}"{items[0]}"'

  clauses = [f'{predicate}{operator}"{item}"' for item in items]
  return "(" + " OR ".join(clauses) + ")"


def search_catalog(
    prompt: str,
    project_id: str,
    credentials: Credentials,
    settings: BigQueryToolConfig,
    location: str,
    page_size: int = 10,
    project_ids_filter: Optional[List[str]] = None,
    dataset_ids_filter: Optional[List[str]] = None,
    types_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
  """Search for BigQuery assets within Dataplex.

  Args:
      prompt (str): The base search query (natural language or keywords).
      project_id (str): The Google Cloud project ID to scope the search.
      credentials (Credentials): Credentials for the request.
      settings (BigQueryToolConfig): BigQuery tool settings.
      location (str): The Dataplex location to use.
      page_size (int): Maximum number of results.
      project_ids_filter (Optional[List[str]]): Specific project IDs to include in the search results.
                                              If None, defaults to the scoping project_id.
      dataset_ids_filter (Optional[List[str]]): BigQuery dataset IDs to filter by.
      types_filter (Optional[List[str]]): Entry types to filter by (e.g., "TABLE", "DATASET").

  Returns:
      dict: Search results or error.
  """
  try:
    if not project_id:
      return {
          "status": "ERROR",
          "error_details": "project_id must be provided.",
      }

    dataplex_client = client.get_dataplex_catalog_client(
        credentials=credentials,
        user_agent=[settings.application_name, "search_catalog"],
    )

    query_parts = []
    if prompt:
      query_parts.append(f"({prompt})")

    # Filter by project IDs
    projects_to_filter = (
        project_ids_filter if project_ids_filter else [project_id]
    )
    if projects_to_filter:
      query_parts.append(
          _construct_search_query_helper("projectid", "=", projects_to_filter)
      )

    # Filter by dataset IDs
    if dataset_ids_filter:
      dataset_resource_filters = [
          f'linked_resource:"//bigquery.googleapis.com/projects/{pid}/datasets/{did}/*"'
          for pid in projects_to_filter
          for did in dataset_ids_filter
      ]
      if dataset_resource_filters:
        query_parts.append(f"({' OR '.join(dataset_resource_filters)})")
    # Filter by entry types
    if types_filter:
      query_parts.append(
          _construct_search_query_helper("type", "=", types_filter)
      )

    # Always scope to BigQuery system
    query_parts.append("system=BIGQUERY")

    full_query = " AND ".join(filter(None, query_parts))

    search_scope = f"projects/{project_id}/locations/{location}"

    request = dataplex_v1.SearchEntriesRequest(
        name=search_scope,
        query=full_query,
        page_size=page_size,
        semantic_search=True,
    )

    response = dataplex_client.search_entries(request=request)

    results = []
    for result in response.results:
      entry = result.dataplex_entry
      source = entry.entry_source
      results.append({
          "name": entry.name,
          "display_name": source.display_name or "",
          "entry_type": entry.entry_type,
          "update_time": str(entry.update_time),
          "linked_resource": source.resource or "",
          "description": source.description or "",
          "location": source.location or "",
      })
    return {"status": "SUCCESS", "results": results}

  except api_exceptions.GoogleAPICallError as e:
    logging.exception("search_catalog tool: API call failed")
    return {"status": "ERROR", "error_details": f"Dataplex API Error: {str(e)}"}
  except Exception as ex:
    logging.exception("search_catalog tool: Unexpected error")
    return {"status": "ERROR", "error_details": str(ex)}
