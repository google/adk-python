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

"""Capabilities reported by a model."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict

from ..utils.model_name_utils import is_gemini_eap_or_2_or_above
from ..utils.variant_utils import get_google_llm_variant
from ..utils.variant_utils import GoogleLLMVariant


def gemini_output_schema_and_tools(model_name: str) -> bool:
  """Whether a Gemini model id can pair an output schema with tools.

  Shared by two callers that must not drift apart:

  * :attr:`Gemini.capabilities`, where it is the model's own permanent
    self-report.
  * The deprecated name-based fallback on :attr:`BaseLlm.capabilities`, which
    keeps models that do not self-report resolving as they did before the
    capability system existed. Only that fallback goes away once out-of-tree
    models declare their capabilities explicitly.

  Args:
    model_name: The name of the model.

  Returns:
    True if the model can use an output schema together with tools.
  """
  return (
      get_google_llm_variant() == GoogleLLMVariant.VERTEX_AI
      and is_gemini_eap_or_2_or_above(model_name)
  )


class LlmCapabilities(BaseModel):
  """Resolved capabilities for an LLM instance.

  Each field holds the *computed result* for one capability, not an override,
  so there is no "defer to auto-detection" placeholder. Most capabilities are
  a simple support flag, but a capability may also carry data.

  Models self-report by overriding :attr:`BaseLlm.capabilities`. Callers read
  the field directly instead of re-deriving support from the model name,
  backend variant, or type::

      if model.capabilities.output_schema_and_tools:
        ...

  This object is immutable: :attr:`BaseLlm.capabilities` recomputes a fresh
  snapshot on every access, so mutating one in place would have no effect on
  the model. Override a capability by subclassing the model instead.
  """

  model_config = ConfigDict(
      extra="forbid",
      frozen=True,  # A resolved snapshot; override by subclassing the model.
      use_attribute_docstrings=True,
  )

  output_schema_and_tools: bool = False
  """Whether the model can use an output schema together with tools."""
