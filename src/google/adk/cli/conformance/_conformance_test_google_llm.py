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

import logging
import re
from typing import Any
from typing import AsyncGenerator
from typing import TYPE_CHECKING

from ...flows.llm_flows._fencing import OTHER_AGENT_CONTEXT_PREAMBLE
from ...flows.llm_flows._fencing import QUOTED_CONTENT_BEGIN
from ...flows.llm_flows._fencing import QUOTED_CONTENT_END
from ...models.google_llm import Gemini

if TYPE_CHECKING:
  from ...models.llm_request import LlmRequest
  from ...models.llm_response import LlmResponse

logger = logging.getLogger('google_adk.' + __name__)


class ReplayVerificationError(Exception):
  """Exception raised when replay verification fails."""


def _normalize_type(val: Any) -> Any:
  if hasattr(val, 'name') and hasattr(val, 'value'):
    return str(val.value).lower()
  if isinstance(val, str) and val.startswith('Type.'):
    return val.split('.')[-1].lower()
  if isinstance(val, str) and val in (
      'STRING',
      'NUMBER',
      'OBJECT',
      'ARRAY',
      'INTEGER',
      'BOOLEAN',
  ):
    return val.lower()
  return val


def _resolve_refs(data: Any, defs: dict[str, Any]) -> Any:
  if isinstance(data, dict):
    if '$ref' in data:
      ref_path = data['$ref']
      if ref_path.startswith('#/$defs/'):
        def_name = ref_path.split('/')[-1]
        if def_name in defs:
          return _resolve_refs(defs[def_name], defs)
    return {k: _resolve_refs(v, defs) for k, v in data.items()}
  elif isinstance(data, list):
    return [_resolve_refs(x, defs) for x in data]
  else:
    return data


def _normalize_schema_dict(data: Any) -> Any:
  if isinstance(data, dict):
    if '$defs' in data:
      defs = data['$defs']
      data = _resolve_refs(data, defs)
      data.pop('$defs', None)

    res = {}
    for k, v in data.items():
      if k in ('title', 'default', 'description'):
        continue
      if k == 'type':
        res[k] = _normalize_type(v)
      else:
        res[k] = _normalize_schema_dict(v)

    if 'anyOf' in res and isinstance(res['anyOf'], list):
      any_of = res['anyOf']
      null_schema = None
      non_null_schemas = []
      for s in any_of:
        if isinstance(s, dict) and s.get('type') == 'null':
          null_schema = s
        else:
          non_null_schemas.append(s)

      if null_schema is not None and len(non_null_schemas) == 1:
        target_schema = non_null_schemas[0]
        if isinstance(target_schema, dict):
          res.update(target_schema)
          res['nullable'] = True
          res.pop('anyOf', None)

    return res
  elif isinstance(data, list):
    return [_normalize_schema_dict(x) for x in data]
  else:
    return data


def _normalize_tool_config(data: Any) -> Any:
  """Normalize function declarations to ignore minor formatting changes."""
  if isinstance(data, dict):
    if 'name' in data and (
        'description' in data
        or 'parameters' in data
        or 'parameters_json_schema' in data
    ):
      if data.get('name') == 'transfer_to_agent':
        data['description'] = 'Transfer the question to another agent.'
      elif 'description' in data and isinstance(data['description'], str):
        data['description'] = data['description'].strip()

      params = data.pop('parameters', None)
      if params is not None:
        data['parameters_json_schema'] = params

      if 'parameters_json_schema' in data:
        data['parameters_json_schema'] = _normalize_schema_dict(
            data['parameters_json_schema']
        )

      data.pop('response', None)
      data.pop('response_json_schema', None)

    return {k: _normalize_tool_config(v) for k, v in data.items()}
  elif isinstance(data, list):
    return [_normalize_tool_config(x) for x in data]
  else:
    return data


_OTHER_AGENT_CONTEXT_PREFIX = 'For context:'

# The preamble as recordings cut before the fencing spell it, and as the runtime
# spells it now. Matched exactly rather than by prefix: a turn the real user
# typed that merely opens with these words is still a real turn and has to
# compare verbatim.
_OTHER_AGENT_PREAMBLES = (
    _OTHER_AGENT_CONTEXT_PREFIX,
    OTHER_AGENT_CONTEXT_PREAMBLE,
)

_QUOTED_CONTENT_PATTERN = re.compile(
    ':\n'
    + re.escape(QUOTED_CONTENT_BEGIN)
    + '\n(?P<payload>.*)\n'
    + re.escape(QUOTED_CONTENT_END)
    + r'\Z',
    re.DOTALL,
)


def _normalize_relayed_agent_text(text: str) -> str:
  """Reduces a relayed agent part to the payload it carries.

  When an agent hands off, its turn is replayed to the next agent behind a
  preamble and between quote markers, so that a payload it was talked into
  emitting cannot read as a fresh instruction. That framing is prose aimed at
  the model: tuning its wording changes every recording that covers a transfer
  without any runtime behavior having changed. Conformance cares that the same
  payload was relayed, not how it was framed, so both the framed and unframed
  shapes reduce to the payload here -- the same reason transfer_to_agent's
  description is pinned in `_normalize_tool_config`.

  The fence itself is asserted on directly in the `_present_other_agent_message`
  unit tests, which is where a regression in it should surface.
  """
  if text in _OTHER_AGENT_PREAMBLES:
    return _OTHER_AGENT_CONTEXT_PREFIX
  return _QUOTED_CONTENT_PATTERN.sub(': \\g<payload>', text)


def _normalize_relayed_agent_content(data: Any) -> Any:
  """Normalizes the user-role messages that carry another agent's turn.

  A relayed turn is a user-role message whose first part is exactly the context
  preamble, followed by at least one quoted part. Anything else -- above all a
  turn the real user typed -- is left to compare verbatim.
  """
  if isinstance(data, dict):
    parts = data.get('parts')
    if (
        data.get('role') == 'user'
        and isinstance(parts, list)
        and len(parts) >= 2
        and isinstance(parts[0], dict)
        and isinstance(parts[0].get('text'), str)
        and parts[0]['text'] in _OTHER_AGENT_PREAMBLES
    ):
      return {
          **data,
          'parts': [
              {**part, 'text': _normalize_relayed_agent_text(part['text'])}
              if isinstance(part, dict) and isinstance(part.get('text'), str)
              else part
              for part in parts
          ],
      }
    return {k: _normalize_relayed_agent_content(v) for k, v in data.items()}
  elif isinstance(data, list):
    return [_normalize_relayed_agent_content(x) for x in data]
  else:
    return data


class _ConformanceTestGemini(Gemini):
  """A mocked Gemini model for conformance test replay mode.

  This class is used to mock the Gemini model in conformance test replay mode.
  It is a subclass of Gemini and overrides the `generate_content_async` method
  to
  return a mocked response from the provided recordings.
  """

  def __init__(
      self,
      *,
      config: dict[str, Any],
      **kwargs: Any,
  ) -> None:
    super().__init__(**kwargs)
    recordings = config.get('_adk_replay_recordings')
    self._user_message_index = config.get('user_message_index')
    self._agent_name = config.get('agent_name')
    self._replay_index = config.get('current_replay_index')
    # Pre-filter LLM recordings for this agent and message index
    self._agent_llm_recordings = [
        recording.llm_recording
        for recording in recordings.recordings
        if recording.agent_name == self._agent_name
        and recording.user_message_index == self._user_message_index
        and recording.llm_recording
    ]

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    """Replay LLM response from recordings instead of making real call."""
    logger.debug(
        'Replaying LLM response for agent %s (index %d)',
        self._agent_name,
        self._replay_index,
    )

    if self._replay_index >= len(self._agent_llm_recordings):
      raise ReplayVerificationError(
          'Runtime sent more LLM requests than expected for agent'
          f" '{self._agent_name}' at user_message_index"
          f' {self._user_message_index}. Expected'
          f' {len(self._agent_llm_recordings)}, but got request at index'
          f' {self._replay_index}'
      )

    recording = self._agent_llm_recordings[self._replay_index]

    # Verify request matches
    self._verify_llm_request_match(
        recording.llm_request, llm_request, self._replay_index
    )

    for response in recording.llm_responses:
      yield response

  def _verify_llm_request_match(
      self,
      recorded_request: LlmRequest,
      current_request: LlmRequest,
      replay_index: int,
  ) -> None:
    """Verify that the current LLM request exactly matches the recorded one."""
    # Comprehensive exclude dict for all fields that can differ between runs
    excluded_fields = {
        'live_connect_config': True,
        'config': {  # some config fields can vary per run
            'http_options': True,
            'labels': True,
        },
    }

    # Compare using model dumps with nested exclude dict
    recorded_dict = recorded_request.model_dump(
        exclude_none=True, exclude=excluded_fields, exclude_defaults=True
    )
    current_dict = current_request.model_dump(
        exclude_none=True, exclude=excluded_fields, exclude_defaults=True
    )

    recorded_dict = _normalize_tool_config(recorded_dict)
    current_dict = _normalize_tool_config(current_dict)

    recorded_dict = _normalize_relayed_agent_content(recorded_dict)
    current_dict = _normalize_relayed_agent_content(current_dict)

    if recorded_dict != current_dict:
      raise ReplayVerificationError(
          f"""LLM request mismatch in turn {self._user_message_index} for agent '{self._agent_name}' (index {replay_index}):
recorded: {recorded_dict}
current: {current_dict}"""
      )
