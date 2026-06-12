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

try:
  import google.antigravity  # noqa: F401
except ImportError as e:
  raise ImportError(
      "The 'google-antigravity' package is required to use the ADK"
      ' Antigravity integration. Install it with: pip install'
      ' "google-adk[antigravity]"'
  ) from e

from ._antigravity_agent import AntigravityAgent

__all__ = [
    'AntigravityAgent',
]
