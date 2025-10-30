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

"""A2A specific experimental decorator with custom warning message."""

from __future__ import annotations

from google.adk.utils.feature_decorator import _make_feature_decorator

a2a_experimental = _make_feature_decorator(
    label="EXPERIMENTAL",
    default_message=(
        "ADK Implementation for A2A support (A2aAgentExecutor, RemoteA2aAgent "
        "and corresponding supporting components etc.) is in experimental mode "
        "and is subjected to breaking changes. A2A protocol and SDK are"
        "themselves not experimental. Once it's stable enough the experimental "
        "mode will be removed. Your feedback is welcome."
    ),
)
"""Mark a class or function as experimental A2A feature.

This decorator shows a specific warning message for A2A functionality,
indicating that the API is experimental and subject to breaking changes.

Sample usage:

```
# Use with default A2A experimental message
@a2a_experimental
class A2AExperimentalClass:
  pass

# Use with custom message (overrides default A2A message)
@a2a_experimental("Custom A2A experimental message.")
def a2a_experimental_function():
  pass

# Use with empty parentheses (same as default A2A message)
@a2a_experimental()
class AnotherA2AClass:
  pass
```
"""
