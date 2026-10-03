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

"""Snowflake Cortex Agents integration.

Runs an existing, named Snowflake Cortex Agent as an ADK ``BaseAgent`` node.
The agent calls the Cortex Agents Run REST API directly with the ``httpx``
client that ADK already depends on, so no extra package is required.

Experimental: like everything under ``google.adk.labs``, this API may change
or be removed without notice.

Example:
  ```python
  from google.adk.labs.snowflake import SnowflakeCortexAgent

  def bearer_headers(ctx):
    return {'Authorization': f'Bearer {load_snowflake_token()}'}

  root_agent = SnowflakeCortexAgent(
      name='sales_analyst',
      account_url='https://<account>.snowflakecomputing.com',
      database='SALES_DB',
      schema_name='ANALYTICS',
      cortex_agent_name='SALES_AGENT',
      header_provider=bearer_headers,
  )
  ```
"""

from ._snowflake_cortex_agent import SnowflakeCortexAgent

__all__ = [
    'SnowflakeCortexAgent',
]
