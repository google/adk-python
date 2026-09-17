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

"""The MCP SDK's module name, as ADK's open-source build resolves it.

Importing `_mcp.py` beside this file loads the SDK, which is several hundred
modules. Code that only has to know whether the SDK is already loaded asks
`sys.modules` for this name instead, and pays nothing.

Like `_mcp.py`, this flavor must never name the internal copy.
"""

from __future__ import annotations

SDK_MODULE_NAME = "mcp"
