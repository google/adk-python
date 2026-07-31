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

from abc import abstractmethod
from typing import AsyncGenerator
from typing import TYPE_CHECKING
import warnings

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict

from ._capabilities import gemini_output_schema_and_tools
from ._capabilities import LlmCapabilities
from .base_llm_connection import BaseLlmConnection

if TYPE_CHECKING:
  from .llm_request import LlmRequest
  from .llm_response import LlmResponse


class BaseLlm(BaseModel):
  """The BaseLLM class."""

  model_config = ConfigDict(
      # This allows us to use arbitrary types in the model. E.g. PIL.Image.
      arbitrary_types_allowed=True,
  )
  """The pydantic model config."""

  model: str
  """The name of the LLM, e.g. gemini-2.5-flash or gemini-2.5-pro."""

  @property
  def capabilities(self) -> LlmCapabilities:
    """The capabilities of this model instance.

    Subclasses override this to declare what they support, so that callers can
    ask the model instead of deriving support from its name, backend variant,
    or type. A model that does not override it falls back to name-based
    detection, which reproduces the behavior that predates this property.

    Users who need different capabilities for a specific model can subclass it
    and override this property. Build on what the parent reports, so that
    capabilities the override does not name keep the value the parent gave
    them::

        class MyGemini(Gemini):

          @property
          def capabilities(self) -> LlmCapabilities:
            return LlmCapabilities(
                **super().capabilities.model_dump()
                | {'output_schema_and_tools': True}
            )

    Avoid ``model_copy(update=...)`` here: it skips validation, so a misspelled
    capability name silently has no effect instead of raising.

    Declare them outright when subclassing ``BaseLlm`` directly, rather than
    building on ``super().capabilities``, which would route through the
    deprecated name-based fallback below::

        class MyModel(BaseLlm):

          @property
          def capabilities(self) -> LlmCapabilities:
            return LlmCapabilities(output_schema_and_tools=True)

    Overrides should stay a plain property rather than a cached one: a
    capability may depend on state that changes after construction, such as an
    environment variable or a reassigned ``model``.
    """
    return LlmCapabilities(
        output_schema_and_tools=self._legacy_output_schema_and_tools(),
    )

  def _legacy_output_schema_and_tools(self) -> bool:
    """Name-based fallback for models that do not report the capability.

    Deprecated. It exists so that a model defined outside ADK keeps resolving
    the way it did before :attr:`capabilities`, when support was inferred from
    the model name rather than declared by the model. It is removed once such
    models declare the capability explicitly, at which point a model that has
    not been updated stops pairing an output schema with tools.

    The warning fires only when the fallback grants the capability, i.e. only
    for the models whose behavior would change if it were removed. ``Gemini``
    and ``LiteLlm`` override :attr:`capabilities` outright and never reach it.

    It is a ``FutureWarning`` rather than a ``DeprecationWarning`` because ADK,
    not the model's author, is the one that reads :attr:`capabilities`. Python
    ignores ``DeprecationWarning`` unless it is attributed to ``__main__``, and
    no ``stacklevel`` reaches the author's code from here — they only declare
    the subclass, they never call this. The warning has to be visible by
    default to be worth anything.

    Returns:
      True if the model can use an output schema together with tools.
    """
    if not gemini_output_schema_and_tools(self.model):
      return False
    warnings.warn(
        f'{type(self).__name__} relies on name-based detection of'
        ' output_schema_and_tools. Override BaseLlm.capabilities to declare'
        ' it explicitly; this fallback will be removed in a future release.',
        FutureWarning,
        stacklevel=3,
    )
    return True

  @classmethod
  def supported_models(cls) -> list[str]:
    """Returns a list of supported models in regex for LlmRegistry."""
    return []

  @abstractmethod
  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Generates content for a single model turn.

    This method handles Server-Sent Events (SSE) streaming for unidirectional
    content generation. For bidirectional streaming (e.g., Gemini Live API),
    use the `connect()` method instead.

    Args:
      llm_request: LlmRequest, the request to send to the LLM.
      stream: bool = False, whether to enable SSE streaming mode.

    Yields:
      LlmResponse objects representing the model's response for one turn.

      **Non-streaming mode (stream=False):**

        Yields exactly one LlmResponse containing the complete model output
        (text, function calls, bytes, etc.). This response has `partial=False`.

      **Streaming mode (stream=True):**

        Yields multiple LlmResponse objects as chunks arrive:

        - Intermediate chunks: `partial=True` (progressive updates)
        - Final chunk: `partial=False` (aggregated content from entire turn,
          identical to stream=False output)
        - Text consolidation: Consecutive text parts of the same type
          (thought/non-thought) SHOULD merge without separator, but client
          code must not rely on this - unconsolidated parts are unusual but also
          valid

      **Common content in partial chunks:**

        All intermediate chunks have `partial=True` regardless of content type.
        Common examples include:

        - Text: Streams incrementally as tokens arrive
        - Function calls: May arrive in separate chunks
        - Bytes (e.g., images): Typically arrive as single chunk, interleaved
          with text
        - Thoughts: Stream incrementally when thinking_config is enabled

      **Examples:**

      1. Simple text streaming::

           LlmResponse(partial=True,  parts=["The weather"])
           LlmResponse(partial=True,  parts=[" in Tokyo is"])
           LlmResponse(partial=True,  parts=[" sunny."])
           LlmResponse(partial=False, parts=["The weather in Tokyo is sunny."])

      2. Text + function call::

           LlmResponse(partial=True,  parts=[Text("Let me check...")])
           LlmResponse(partial=True,  parts=[FunctionCall("get_weather", ...)])
           LlmResponse(partial=False, parts=[Text("Let me check..."),
                                             FunctionCall("get_weather", ...)])

      3. Parallel function calls across chunks::

           LlmResponse(partial=True,  parts=[Text("Checking both cities...")])
           LlmResponse(partial=True,  parts=[FunctionCall("get_weather", Tokyo)])
           LlmResponse(partial=True,  parts=[FunctionCall("get_weather", NYC)])
           LlmResponse(partial=False, parts=[Text("Checking both cities..."),
                                             FunctionCall("get_weather", Tokyo),
                                             FunctionCall("get_weather", NYC)])

      4. Text + bytes (image generation with gemini-2.5-flash-image)::

           LlmResponse(partial=True,  parts=[Text("Here's an image of a dog.")])
           LlmResponse(partial=True,  parts=[Text("\n")])
           LlmResponse(partial=True,  parts=[Blob(image/png, 1.6MB)])
           LlmResponse(partial=True,  parts=[Text("It carries a bone")])
           LlmResponse(partial=True,  parts=[Text(" and running around.")])
           LlmResponse(partial=False, parts=[Text("Here's an image of a dog.\n"),
                                             Blob(image/png, 1.6MB),
                                             Text("It carries a bone and running around.")])

         Note: Consecutive text parts before and after blob merge separately.

      5. Text with thinking (gemini-2.5-flash with thinking_config)::

           LlmResponse(partial=True,  parts=[Thought("Let me analyze...")])
           LlmResponse(partial=True,  parts=[Thought("The user wants...")])
           LlmResponse(partial=True,  parts=[Text("Based on my analysis,")])
           LlmResponse(partial=True,  parts=[Text(" the answer is 42.")])
           LlmResponse(partial=False, parts=[Thought("Let me analyze...The user wants..."),
                                             Text("Based on my analysis, the answer is 42.")])

         Note: Consecutive parts of same type merge (thoughts→thought, text→text).

      **Important:** All yielded responses represent one logical model turn.
      The final response with `partial=False` should be identical to the
      response that would be received with `stream=False`.
    """
    raise NotImplementedError(
        f'Async generation is not supported for {self.model}.'
    )
    yield  # AsyncGenerator requires a yield statement in function body.

  def _maybe_append_user_content(self, llm_request: LlmRequest) -> None:
    """Appends a user content, so that model can continue to output.

    Args:
      llm_request: LlmRequest, the request to send to the Gemini model.
    """
    # If no content is provided, append a user content to hint model response
    # using system instruction.
    if not llm_request.contents:
      llm_request.contents.append(
          types.Content(
              role='user',
              parts=[
                  types.Part(
                      text=(
                          'Handle the requests as specified in the System'
                          ' Instruction.'
                      )
                  )
              ],
          )
      )
      return

    # Insert a user content to preserve user intent and to avoid empty
    # model response.
    if llm_request.contents[-1].role != 'user':
      llm_request.contents.append(
          types.Content(
              role='user',
              parts=[
                  types.Part(
                      text=(
                          'Continue processing previous requests as instructed.'
                          ' Exit or provide a summary if no more outputs are'
                          ' needed.'
                      )
                  )
              ],
          )
      )

  def connect(self, llm_request: LlmRequest) -> BaseLlmConnection:
    """Creates a live connection to the LLM.

    Args:
      llm_request: LlmRequest, the request to send to the LLM.

    Returns:
      BaseLlmConnection, the connection to the LLM.
    """
    raise NotImplementedError(
        f'Live connection is not supported for {self.model}.'
    )
