"""Static Instruction Test Agent Package.

This package contains a sample agent for testing static instruction functionality
and context caching optimization features.

The agent demonstrates:
- Static instructions that remain constant for caching
- Dynamic instructions that change based on session state
- Various instruction provider patterns
- Performance benefits of context caching
"""

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

from . import agent

__all__ = ['agent']
