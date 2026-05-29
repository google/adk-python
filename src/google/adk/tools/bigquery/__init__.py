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

"""BigQuery Tools.

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

from __future__ import annotations

import importlib
import typing
import warnings

warnings.warn(
    "google.adk.tools.bigquery is deprecated, use"
    " google.adk.integrations.bigquery instead",
    DeprecationWarning,
    stacklevel=2,
)

if typing.TYPE_CHECKING:
  from google.adk.integrations.bigquery import BigQueryCredentialsConfig
  from google.adk.integrations.bigquery import BigQueryToolset
  from google.adk.integrations.bigquery import get_bigquery_skill

# Forward public names to integrations/bigquery for backward compatibility.
# Uses __getattr__ instead of sys.modules replacement so that submodules
# (e.g. bigquery_skill) under this package remain importable.
_TARGET = "google.adk.integrations.bigquery"

_FORWARDED_NAMES = {
    "BigQueryToolset",
    "BigQueryCredentialsConfig",
    "get_bigquery_skill",
}


def __getattr__(name: str) -> typing.Any:
  if name in _FORWARDED_NAMES:
    mod = importlib.import_module(_TARGET)
    return getattr(mod, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  return list(_FORWARDED_NAMES)
