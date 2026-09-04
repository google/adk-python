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

"""Analytics assistant that runs a Snowflake Cortex Agent as an ADK root agent.

Wraps an existing Cortex Agent object with `SnowflakeCortexAgent`. See the
guide at docs/guides/labs/snowflake/snowflake_cortex_agent/index.md for setup
and details.
"""

import os

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.labs.snowflake import SnowflakeCortexAgent


def _required(name: str) -> str:
  value = os.environ.get(name)
  if not value:
    raise RuntimeError(
        f"Set {name} in the environment or in a .env file next to agent.py."
    )
  return value


def snowflake_headers(ctx: ReadonlyContext) -> dict[str, str]:
  """Reads the token on every request so a rotated token is picked up."""
  del ctx  # One service token for every user; see the guide for per-user auth.
  return {
      "Authorization": f"Bearer {_required('SNOWFLAKE_TOKEN')}",
      "X-Snowflake-Authorization-Token-Type": os.environ.get(
          "SNOWFLAKE_TOKEN_TYPE", "PROGRAMMATIC_ACCESS_TOKEN"
      ),
  }


# 1. Point at the Cortex Agent object that already exists in Snowflake. The
#    ADK name is separate from the Snowflake object name.
# 2. Credentials come from `header_provider`, never from a field, so they stay
#    out of `repr`, the adk web agent graph and the session store.
root_agent = SnowflakeCortexAgent(
    name="snowflake_cortex_analyst",
    description="Answers data questions by running a Snowflake Cortex Agent.",
    account_url=_required("SNOWFLAKE_ACCOUNT_URL"),
    database=_required("SNOWFLAKE_DATABASE"),
    schema_name=_required("SNOWFLAKE_SCHEMA"),
    cortex_agent_name=_required("SNOWFLAKE_CORTEX_AGENT"),
    header_provider=snowflake_headers,
)
