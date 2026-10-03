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

_REQUIRED_ENV = (
    "SNOWFLAKE_ACCOUNT_URL",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
    "SNOWFLAKE_CORTEX_AGENT",
    "SNOWFLAKE_TOKEN",
)


def _env(name: str, default: str = "") -> str:
  return os.environ.get(name, default).strip()


def _check_configured() -> None:
  # Checked when the first request is made, not at import: `adk web` and the
  # sample tests import every sample, with or without Snowflake credentials.
  missing = [name for name in _REQUIRED_ENV if not _env(name)]
  if missing:
    raise RuntimeError(
        "Snowflake settings are missing:"
        f" {', '.join(missing)}. Set them in the environment or in a .env"
        " file next to agent.py."
    )


def snowflake_headers(ctx: ReadonlyContext) -> dict[str, str]:
  """Reads the token on every request so a rotated token is picked up."""
  del ctx  # One service token for every user; see the guide for per-user auth.
  _check_configured()
  return {
      "Authorization": f"Bearer {_env('SNOWFLAKE_TOKEN')}",
      "X-Snowflake-Authorization-Token-Type": _env(
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
    account_url=_env("SNOWFLAKE_ACCOUNT_URL"),
    database=_env("SNOWFLAKE_DATABASE"),
    schema_name=_env("SNOWFLAKE_SCHEMA"),
    cortex_agent_name=_env("SNOWFLAKE_CORTEX_AGENT"),
    header_provider=snowflake_headers,
)
