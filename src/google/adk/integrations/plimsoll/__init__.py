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

"""Plimsoll integration for self-hosted, persistent agent workspaces.

Plimsoll is a self-hosted sandbox service. Each call states the isolation tier
it ran behind and carries a run record that the client checks.

Requires the ``plimsoll`` extra: ``pip install google-adk[plimsoll]``.

Example:
  ```python
  from google.adk.integrations.plimsoll import PlimsollEnvironment

  env = PlimsollEnvironment(url="http://127.0.0.1:8746")
  await env.initialize()
  result = await env.execute("python3 -c 'print(6 * 7)'")
  await env.close()
  ```
"""

from ._plimsoll_environment import PlimsollEnvironment

__all__ = [
    "PlimsollEnvironment",
]
