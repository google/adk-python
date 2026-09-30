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

from __future__ import annotations

from typing import Any
from typing import cast
from typing import TypeAlias

from google.genai import types as genai_types

from ..agents.llm_agent import Agent
from ..models.base_llm import BaseLlm
from ..models.llm_request import LlmRequest
from ..utils.context_utils import Aclosing

# This is GEPA's public LanguageModel input contract.
GEPAPrompt: TypeAlias = str | list[dict[str, Any]]


def require_static_instruction(agent: Agent) -> str:
  """Returns the instruction that can seed an offline GEPA optimization."""
  instruction = agent.instruction
  if not isinstance(instruction, str):
    raise ValueError(
        "GEPA optimization requires initial_agent.instruction to be a static"
        " string; request-scoped instruction providers cannot be resolved"
        " without an invocation context."
    )
  return instruction


async def generate_reflection_response(
    *,
    llm: BaseLlm,
    model: str,
    config: genai_types.GenerateContentConfig,
    prompt: GEPAPrompt,
) -> str:
  """Runs one GEPA reflection request and returns all non-thought text."""
  request = LlmRequest(
      model=model,
      config=config,
      contents=[
          genai_types.Content(
              parts=[genai_types.Part(text=cast(str, prompt))], role="user"
          )
      ],
  )
  async with Aclosing(llm.generate_content_async(request)) as responses:
    # only one yield expected so no need to loop
    response = await responses.__anext__()
    content = response.content
    if not content or not content.parts:
      return ""
    return "".join(
        part.text for part in content.parts if part.text and not part.thought
    )
