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

from __future__ import annotations

from typing_extensions import override

from ..agents import BaseAgent


class IdentityAgentCreator:
  """An implementation of the AgentCreator interface that always returns a copy of the root agent."""

  def __init__(self, root_agent: BaseAgent):
    self._root_agent = root_agent

  @override
  def get_agent(
      self,
  ) -> BaseAgent:
    """Returns a deep copy of the root agent."""
    # TODO: Use Agent.clone() when the PR is merged.
    # return self._root_agent.model_copy(deep=True)
    return self._root_agent.clone()
