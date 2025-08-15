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

"""Spanner Tools (Experimental).

Spanner Tools under this module are hand crafted and customized while the tools
under google.adk.tools.google_api_tool are auto generated based on API
definition. The rationales to have customized tool are:

1. A dedicated Spanner toolset to provide an easier, integrated way to interact
with Spanner database and tables for building AI Agent applications quickly.
2. We want to provide more high-level tools like Search, ML.Predict, and Graph
etc.
3. We want to provide extra access guardrails and controls in those tools.
For example, execute_sql can't arbitrarily mutate existing data.
4. We want to provide Spanner best practices and knowledge assistants for ad-hoc
analytics queries.
5. Use Spanner Toolset for more customization and control to interact with
Spanner database and tables.
"""

from . import spanner_credentials
from .spanner_toolset import SpannerToolset

SpannerCredentialsConfig = spanner_credentials.SpannerCredentialsConfig
__all__ = [
    "SpannerToolset",
    "SpannerCredentialsConfig",
]
