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


import logging

from .google_api_tool_set import GoogleApiToolSet

logger = logging.getLogger(__name__)

_tool_sets = {}

def load_tool_set(tool_name):
    if tool_name not in _tool_sets:
        _tool_sets[tool_name] = GoogleApiToolSet.load_tool_set(
            api_name=tool_name,
            api_version="v1",
        )
    return _tool_sets[tool_name]
