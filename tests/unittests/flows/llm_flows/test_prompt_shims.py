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

"""Tests ensuring backward compatibility shims for prompt modules function correctly."""

from __future__ import annotations

from google.adk.flows.llm_flows import identity
from google.adk.flows.llm_flows import instructions
from google.adk.flows.llm_flows import prompt
from google.adk.flows.llm_flows.prompt import _identity as prompt_identity
from google.adk.flows.llm_flows.prompt import _instructions as prompt_instructions


def test_instructions_shim_reexports():
  assert instructions.request_processor is prompt_instructions.request_processor
  assert (
      instructions._InstructionsLlmRequestProcessor
      is prompt_instructions._InstructionsLlmRequestProcessor
  )
  assert (
      instructions._label_dynamic_instruction
      is prompt_instructions._label_dynamic_instruction
  )
  assert (
      instructions._INSTRUCTION_BEGIN is prompt_instructions._INSTRUCTION_BEGIN
  )
  assert instructions._INSTRUCTION_END is prompt_instructions._INSTRUCTION_END
  assert (
      instructions._process_agent_instruction
      is prompt_instructions._process_agent_instruction
  )
  assert prompt._instructions is prompt_instructions


def test_identity_shim_reexports():
  assert identity.request_processor is prompt_identity.request_processor
  assert (
      identity._IdentityLlmRequestProcessor
      is prompt_identity._IdentityLlmRequestProcessor
  )
  assert prompt._identity is prompt_identity
