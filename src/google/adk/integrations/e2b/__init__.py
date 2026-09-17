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

"""E2B sandbox integration.

This module provides a BaseEnvironment implementation backed by an E2B
remote sandbox, offering a persistent remote workspace for file CRUD,
shell execution, and on-demand software installs.

Requires the ``e2b`` extra: ``pip install google-adk[e2b]``.

Example:
  ```python
  from google.adk.integrations.e2b import E2BEnvironment

  env = E2BEnvironment(image="base", timeout=300)
  await env.initialize()
  result = await env.execute("pip install requests")
  await env.close()
  ```
"""

from ._e2b_environment import E2BEnvironment

__all__ = [
    'E2BEnvironment',
]
