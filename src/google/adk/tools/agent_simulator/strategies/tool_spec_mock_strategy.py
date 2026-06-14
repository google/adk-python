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

import warnings

from google.adk.tools.environment_simulation.strategies.tool_spec_mock_strategy import ToolSpecMockStrategy

warnings.warn(
    "google.adk.tools.agent_simulator.strategies.tool_spec_mock_strategy is"
    " moved to"
    " google.adk.tools.environment_simulation.strategies.tool_spec_mock_strategy",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ToolSpecMockStrategy"]
