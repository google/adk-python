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

"""Finalizes the instructions tools contributed while they were resolved."""

from __future__ import annotations

from typing import AsyncGenerator

from google.genai import types
from typing_extensions import override

from ....agents.invocation_context import InvocationContext
from ....events.event import Event
from ....models.llm_request import LlmRequest
from .._base_llm_processor import BaseLlmRequestProcessor
from ..core._utils import as_llm_agent


def _routes_into_contents(invocation_context: InvocationContext) -> bool:
  """Returns whether a tool's instruction should ride in the contents.

  Routing keeps `system_instruction` stable for provider-side prompt caching.
  Live and Interactions API requests don't gain from that, and both read a
  trailing user-role content as user input: live replays it as a finished
  user turn, so the model answers before the user speaks, and a chained
  interaction stores it as user input on every turn. Those keep the
  instruction in `system_instruction`.
  """
  if invocation_context.live_request_queue is not None:
    return False

  from ....models.google_llm import Gemini

  model = getattr(as_llm_agent(invocation_context), 'canonical_model', None)
  return not (isinstance(model, Gemini) and model.use_interactions_api)


async def finalize_dynamic_instructions(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
) -> None:
  """Finalizes and resolves dynamic instructions from LlmRequest."""
  if not llm_request._dynamic_instructions:
    return

  combined_text = '\n\n'.join(llm_request._dynamic_instructions)

  from ....features import FeatureName
  from ....features import is_feature_enabled

  # TODO: Deprecate system_instruction fallback and make user content routing standard.
  if is_feature_enabled(
      FeatureName.DYNAMIC_INSTRUCTION_ROUTING
  ) and _routes_into_contents(invocation_context):
    from ..context._contents import _add_instructions_to_user_content
    from ._instructions import _label_dynamic_instruction

    # Same user-role carrier as the agent's instruction, so same label.
    instruction_content = types.Content(
        role='user',
        parts=[
            types.Part.from_text(text=_label_dynamic_instruction(combined_text))
        ],
    )
    await _add_instructions_to_user_content(
        invocation_context,
        llm_request,
        [instruction_content],
    )
  else:
    llm_request.append_instructions([combined_text])

  # Clear dynamic instructions to prevent double finalization.
  llm_request._dynamic_instructions.clear()


class _DynamicInstructionsLlmRequestProcessor(BaseLlmRequestProcessor):
  """Folds tool-contributed instructions into the request.

  Tools queue instructions while they are being resolved, so this has to run
  after the processor that resolves them.
  """

  name = 'dynamic_instructions'

  @override
  async def run_async(
      self, invocation_context: InvocationContext, llm_request: LlmRequest
  ) -> AsyncGenerator[Event, None]:
    await finalize_dynamic_instructions(invocation_context, llm_request)

    # Maintain async generator behavior
    if False:  # Ensures it behaves as a generator
      yield  # This is a no-op but maintains generator structure


request_processor = _DynamicInstructionsLlmRequestProcessor()
