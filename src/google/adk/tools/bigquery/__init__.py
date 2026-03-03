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

"""Alias — canonical location is google.adk.integration.bigquery.

BigQuery Tools under this module are hand crafted and customized while the tools
under google.adk.tools.google_api_tool are auto generated based on API
definition. The rationales to have customized tool are:

1. BigQuery APIs have functions overlaps and LLM can't tell what tool to use
2. BigQuery APIs have a lot of parameters with some rarely used, which are not
   LLM-friendly
3. We want to provide more high-level tools like forecasting, RAG, segmentation,
   etc.
4. We want to provide extra access guardrails in those tools. For example,
   execute_sql can't arbitrarily mutate existing data.
"""

import sys

from google.adk.integration.bigquery import bigquery_credentials
from google.adk.integration.bigquery import bigquery_skill
from google.adk.integration.bigquery import bigquery_toolset
from google.adk.integration.bigquery import client
from google.adk.integration.bigquery import config
from google.adk.integration.bigquery import data_insights_tool
from google.adk.integration.bigquery import metadata_tool
from google.adk.integration.bigquery import query_tool

# Register canonical modules under the alias path so that imports like
# ``from google.adk.tools.bigquery.client import get_bigquery_client``
# resolve to the *same* module object where the real code lives.  This
# ensures mock.patch.object works correctly in tests.
_CANONICAL_MODULES = {
    "bigquery_credentials": bigquery_credentials,
    "bigquery_skill": bigquery_skill,
    "bigquery_toolset": bigquery_toolset,
    "client": client,
    "config": config,
    "data_insights_tool": data_insights_tool,
    "metadata_tool": metadata_tool,
    "query_tool": query_tool,
}

for _name, _mod in _CANONICAL_MODULES.items():
  sys.modules[f"{__name__}.{_name}"] = _mod

# Re-export top-level names for convenience.
from google.adk.integration.bigquery import BigQueryCredentialsConfig
from google.adk.integration.bigquery import BigQueryToolset
from google.adk.integration.bigquery import get_bigquery_skill

__all__ = [
    "BigQueryCredentialsConfig",
    "BigQueryToolset",
    "get_bigquery_skill",
]
