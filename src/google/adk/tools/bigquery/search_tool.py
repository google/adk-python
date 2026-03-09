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

from google.api_core import exceptions as api_exceptions
from google.auth.credentials import Credentials
from google.cloud import dataplex_v1

from . import client
from .config import BigQueryToolConfig


def _construct_search_query_helper(
    predicate: str, operator: str, items: list[str]
) -> str:
  """Constructs a search query part for a specific predicate and items."""
  if not items:
    return ""

  clauses = [f'{predicate}{operator}"{item}"' for item in items]
  return "(" + " OR ".join(clauses) + ")" if len(items) > 1 else clauses[0]


def search_catalog(
    prompt: str,
    project_id: str,
    *,
    credentials: Credentials,
    settings: BigQueryToolConfig,
    location: str | None = None,
    page_size: int = 10,
    project_ids_filter: list[str] | None = None,
    dataset_ids_filter: list[str] | None = None,
    types_filter: list[str] | None = None,
) -> dict[str, Any]:
  """Searches for BigQuery assets within Dataplex.

  Args:
      prompt: The base search query (natural language or keywords).
      project_id: The Google Cloud project ID to scope the search.
      credentials: Credentials for the request.
      settings: BigQuery tool settings.
      location: The Dataplex location to use.
      page_size: Maximum number of results.
      project_ids_filter: Specific project IDs to include in the search results.
        If None, defaults to the scoping project_id.
      dataset_ids_filter: BigQuery dataset IDs to filter by.
      types_filter: Entry types to filter by (e.g., BigQueryEntryType.TABLE,
        BigQueryEntryType.DATASET).

  Returns:
      Search results or error. The "results" list contains items with:
          - name: The Dataplex Entry name (e.g.,
            "projects/p/locations/l/entryGroups/g/entries/e").
          - linked_resource: The underlying BigQuery resource name (e.g.,
            "//bigquery.googleapis.com/projects/p/datasets/d/tables/t").
          - display_name, entry_type, description, location, update_time.

  Examples:
      Search for tables related to customer data:

          >>> search_catalog(
          ...     prompt="Search for tables related to customer data",
          ...     project_id="my-project",
          ...     credentials=creds,
          ...     settings=settings
          ... )
          {
            "status": "SUCCESS",
            "results": [
              {
                "name":
                "projects/my-project/locations/us/entryGroups/@bigquery/entries/entry-id",
                "display_name": "customer_table",
                "entry_type":
                "projects/p/locations/l/entryTypes/bigquery-table",
                "linked_resource":
                "//bigquery.googleapis.com/projects/my-project/datasets/d/tables/customer_table",
                "description": "Table containing customer details.",
                "location": "us",
                "update_time": "2024-01-01 12:00:00+00:00"
              }
            ]
          }
  """

  try:
    if not project_id:
      return {
          "status": "ERROR",
          "error_details": "project_id must be provided.",
      }

    with client.get_dataplex_catalog_client(
        credentials=credentials,
        user_agent=[settings.application_name, "search_catalog"],
    ) as dataplex_client:
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
        dataset_resource_filters = []
        for pid in projects_to_filter:
          for did in dataset_ids_filter:
            dataset_resource_filters.append(
                f'linked_resource:"//bigquery.googleapis.com/projects/{pid}/datasets/{did}/*"'
            )
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

      search_location = location or settings.location or "global"
      search_scope = f"projects/{project_id}/locations/{search_location}"

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
    return {"status": "ERROR", "error_details": f"Dataplex API Error: {e}"}
  except Exception as e:
    logging.exception("search_catalog tool: Unexpected error")
    return {"status": "ERROR", "error_details": repr(e)}
