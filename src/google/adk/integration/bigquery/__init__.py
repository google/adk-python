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

"""BigQuery integration — tools, credentials, and skills.

Imports are lazy so that ``get_bigquery_skill()`` can be used without
pulling in the full BigQuery dependency stack (dataplex, etc.).
"""

from __future__ import annotations

__all__ = [
    "BigQueryCredentialsConfig",
    "BigQueryToolset",
    "get_bigquery_skill",
]


def __getattr__(name: str):
  if name == "BigQueryCredentialsConfig":
    from .bigquery_credentials import BigQueryCredentialsConfig

    return BigQueryCredentialsConfig
  if name == "BigQueryToolset":
    from .bigquery_toolset import BigQueryToolset

    return BigQueryToolset
  if name == "get_bigquery_skill":
    from .bigquery_skill import get_bigquery_skill

    return get_bigquery_skill
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
