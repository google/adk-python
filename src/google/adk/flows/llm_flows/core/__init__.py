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

"""Core runtime utilities and turn lifecycle orchestration for LLM flows.

Submodules (_finalizer, _resume, _utils) are intentionally not eagerly imported
here so that low-level leaf helpers (such as core._utils) can be imported by
extensions without triggering circular initialization through tools.
"""
