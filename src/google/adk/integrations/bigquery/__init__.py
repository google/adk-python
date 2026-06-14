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

"""BigQuery Integration.

This module provides tools and skills for interacting with BigQuery.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
  from .bigquery_credentials import BigQueryCredentialsConfig
  from .bigquery_skill import get_bigquery_skill
  from .bigquery_toolset import BigQueryToolset

# Map attribute names to relative module paths
_lazy_imports = {
    "BigQueryCredentialsConfig": ".bigquery_credentials",
    "BigQueryToolset": ".bigquery_toolset",
    "get_bigquery_skill": ".bigquery_skill",
}


def __getattr__(name: str) -> any:
  if name in _lazy_imports:
    import importlib

    module_path = _lazy_imports[name]
    # __name__ is 'google.adk.integrations.bigquery'
    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
  return list(_lazy_imports.keys())
