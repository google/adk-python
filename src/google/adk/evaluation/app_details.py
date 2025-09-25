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

from google.genai import types as genai_types
from pydantic import Field

from .common import EvalBaseModel


class AgentDetails(EvalBaseModel):
  """Details about the individual agent in the App.

  This could be a root agent or the sub-agents in the Agent Tree.
  """

  name: str
  """The name of the Agent that uniquely identifies it in the App."""

  instructions: str = Field(default="")
  """The instructions set on the Agent."""

  tool_declarations: genai_types.ToolListUnion = Field(default_factory=list)
  """A list of tools available to the Agent."""


class AppDetails(EvalBaseModel):
  """Contains details about the App (the agentic system).

  This structure is only a projection of the acutal app. Only details
  that are relevant to the Eval System are captured here.
  """

  agent_details: dict[str, AgentDetails] = Field(
      default_factory=dict,
  )
  """A mapping from the agent name to the details of that agent."""
