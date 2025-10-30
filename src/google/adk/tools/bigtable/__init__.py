# Copyright 2025 Google LLC
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

"""Bigtable Tools (Experimental).

Bigtable tools under this module are hand crafted and customized while the tools
under google.adk.tools.google_api_tool are auto generated based on API
definition. The rationales to have customized tool are:

1. A dedicated Bigtable toolset to provide an easier, integrated way to interact
with Bigtable for building AI Agent applications quickly.
2. We want to provide extra access guardrails and controls in those tools.
3. Use Bigtable Toolset for more customization and control to interact with
Bigtable tables.
"""

from .bigtable_credentials import BigtableCredentialsConfig
from .bigtable_toolset import BigtableToolset

__all__ = [
    "BigtableToolset",
    "BigtableCredentialsConfig",
]
