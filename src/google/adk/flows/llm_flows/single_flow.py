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

"""Implementation of single flow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from . import basic
from .base_llm_flow import BaseLlmFlow
from .context import _cache
from .context import _compaction
from .context import _contents
from .context import _interactions
from .extensions import _code_execution
from .extensions import _planning
from .prompt import _identity
from .prompt import _instructions
from .prompt import _schema as _output_schema_processor
from .tools import _confirmation as request_confirmation

if TYPE_CHECKING:
  from ._base_llm_processor import BaseLlmRequestProcessor
  from ._base_llm_processor import BaseLlmResponseProcessor

logger = logging.getLogger('google_adk.' + __name__)


def _create_request_processors() -> list[BaseLlmRequestProcessor]:
  """Create the standard request processor list for a single-agent flow.

  The list order below is authoritative: processors run in insertion order,
  and the comments spell out the constraints that order has to respect. Each
  processor declares a stable `name` so callers can locate one without relying
  on its index, and insert relative to it instead of appending blindly.
  """
  from ...auth import auth_preprocessor

  return [
      basic.request_processor,
      auth_preprocessor.request_processor,
      request_confirmation.request_processor,
      _instructions.request_processor,
      _identity.request_processor,
      # Compaction should run before contents so compacted events are reflected
      # in the model request context.
      _compaction.request_processor,
      # Extract the Interactions chain id before contents. Chained requests
      # only need the current turn because the service retains prior state.
      _interactions.request_processor,
      _contents.request_processor,
      # Context cache processor sets up cache config and finds
      # existing cache metadata.
      _cache.request_processor,
      # Some implementations of NL Planning mark planning contents
      # as thoughts in the post processor.  Since these need to be
      # unmarked, NL Planning should be after contents.
      _planning.request_processor,
      # Code execution should be after the contents as it mutates
      # the contents to optimize data files.
      _code_execution.request_processor,
      # Output schema processor adds system instruction and
      # set_model_response when both output_schema and tools are
      # present.
      _output_schema_processor.request_processor,
  ]


def _create_response_processors() -> list[BaseLlmResponseProcessor]:
  """Create the standard response processor list for a single-agent flow.

  As with the request processors, the list order is authoritative and each
  processor carries a stable `name` for callers to locate it by.
  """
  return [
      _planning.response_processor,
      _code_execution.response_processor,
  ]


class SingleFlow(BaseLlmFlow):
  """SingleFlow is the LLM flows that handles tools calls.

  A single flow only consider an agent itself and tools.
  No sub-agents are allowed for single flow.
  """

  def __init__(self) -> None:
    super().__init__()
    self.request_processors += _create_request_processors()
    self.response_processors += _create_response_processors()
