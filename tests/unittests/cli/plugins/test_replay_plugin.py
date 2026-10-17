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

"""Tests for the replay plugin's load / replay / cleanup lifecycle."""

from typing import Any
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.cli.plugins.recordings_schema import Recording
from google.adk.cli.plugins.recordings_schema import Recordings
from google.adk.cli.plugins.recordings_schema import ToolRecording
from google.adk.cli.plugins.replay_plugin import ReplayConfigError
from google.adk.cli.plugins.replay_plugin import ReplayPlugin
from google.adk.cli.plugins.replay_plugin import ReplayVerificationError
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.utils.yaml_utils import dump_pydantic_to_yaml
from google.genai import types
import pytest

from ... import testing_utils

_NON_STREAMING_FILE = 'generated-recordings.yaml'
_STREAMING_FILE = 'generated-recordings-sse.yaml'


class _SpyTool(BaseTool):
  """Tool that records the args it was actually executed with."""

  def __init__(self, name: str = 'roll_die', live_result: Any = None):
    super().__init__(name=name, description='test tool')
    self.live_calls: list[dict[str, Any]] = []
    self._live_result = (
        {'result': 'live'} if live_result is None else live_result
    )

  async def run_async(self, *, args, tool_context):
    self.live_calls.append(args)
    return self._live_result


def _recording(
    *,
    agent_name: str = 'agent_a',
    user_message_index: int = 0,
    tool_name: str = 'roll_die',
    args: Optional[dict[str, Any]] = None,
    response: Optional[dict[str, Any]] = None,
    call_id: str = 'fc-1',
) -> Recording:
  return Recording(
      user_message_index=user_message_index,
      agent_name=agent_name,
      tool_recording=ToolRecording(
          tool_call=types.FunctionCall(
              id=call_id, name=tool_name, args=args or {'sides': 6}
          ),
          tool_response=types.FunctionResponse(
              id=call_id, name=tool_name, response=response or {'result': 4}
          ),
      ),
  )


def _write_recordings(case_dir, recordings, *, file_name=_NON_STREAMING_FILE):
  dump_pydantic_to_yaml(
      Recordings(recordings=recordings),
      case_dir / file_name,
      sort_keys=False,
  )


async def _make_invocation(
    *,
    case_dir=None,
    user_message_index: int = 0,
    streaming_mode: Optional[str] = 'none',
    agent_names: tuple[str, ...] = ('agent_a',),
    replay_tool_responses: Optional[dict[str, list[str]]] = None,
):
  """Builds one invocation plus a per-agent context sharing its session."""
  invocation_context = await testing_utils.create_invocation_context(
      testing_utils.create_test_agent(name=agent_names[0])
  )
  if case_dir is not None:
    config: dict[str, Any] = {
        'dir': str(case_dir),
        'user_message_index': user_message_index,
    }
    if streaming_mode is not None:
      config['streaming_mode'] = streaming_mode
    if replay_tool_responses is not None:
      config['replay_tool_responses'] = replay_tool_responses
    invocation_context.session.state['_adk_replay_config'] = config

  contexts = {agent_names[0]: CallbackContext(invocation_context)}
  for name in agent_names[1:]:
    contexts[name] = CallbackContext(
        invocation_context.model_copy(
            update={'agent': testing_utils.create_test_agent(name=name)}
        )
    )
  return invocation_context, contexts


async def test_before_run_without_replay_config_leaves_plugin_inert(tmp_path):
  """No replay config means the plugin must not intercept anything."""
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(case_dir=None)
  tool = _SpyTool()

  before_run_result = await plugin.before_run_callback(
      invocation_context=invocation_context
  )
  replayed = await plugin.before_tool_callback(
      tool=tool, tool_args={'sides': 6}, tool_context=contexts['agent_a']
  )

  # None tells the runtime to execute the tool itself; the plugin neither ran
  # the tool nor consumed a recording.
  assert before_run_result is None
  assert replayed is None
  assert tool.live_calls == []


async def test_before_run_with_partial_replay_config_leaves_plugin_inert(
    tmp_path,
):
  """A config missing user_message_index must not half-enable replay."""
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(case_dir=tmp_path)
  invocation_context.session.state['_adk_replay_config'] = {
      'dir': str(tmp_path),
      'streaming_mode': 'none',
  }
  tool = _SpyTool()

  await plugin.before_run_callback(invocation_context=invocation_context)
  replayed = await plugin.before_tool_callback(
      tool=tool, tool_args={'sides': 6}, tool_context=contexts['agent_a']
  )

  assert replayed is None
  assert tool.live_calls == []


async def test_before_tool_returns_recorded_response_not_live_result(tmp_path):
  """The recorded response wins over whatever the live tool returns."""
  _write_recordings(tmp_path, [_recording(response={'result': 4})])
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(case_dir=tmp_path)
  tool = _SpyTool(live_result={'result': 'live'})

  await plugin.before_run_callback(invocation_context=invocation_context)
  replayed = await plugin.before_tool_callback(
      tool=tool, tool_args={'sides': 6}, tool_context=contexts['agent_a']
  )

  assert replayed == {'result': 4}


async def test_before_tool_still_executes_the_underlying_tool(tmp_path):
  """Replay verifies the tool runs; only its response is substituted."""
  _write_recordings(tmp_path, [_recording(args={'sides': 6})])
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(case_dir=tmp_path)
  tool = _SpyTool()

  await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.before_tool_callback(
      tool=tool, tool_args={'sides': 6}, tool_context=contexts['agent_a']
  )

  assert tool.live_calls == [{'sides': 6}]


@pytest.mark.parametrize(
    ('streaming_mode', 'file_name'),
    [('none', _NON_STREAMING_FILE), ('sse', _STREAMING_FILE)],
)
async def test_selected_tool_uses_recorded_response_without_executing(
    tmp_path, streaming_mode, file_name
):
  """Selected tools use the verified response without contacting a backend."""
  _write_recordings(
      tmp_path, [_recording(response={'result': 4})], file_name=file_name
  )
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(
      case_dir=tmp_path,
      streaming_mode=streaming_mode,
      replay_tool_responses={'agent_a': ['roll_die']},
  )
  tool = _SpyTool()

  await plugin.before_run_callback(invocation_context=invocation_context)
  replayed = await plugin.before_tool_callback(
      tool=tool, tool_args={'sides': 6}, tool_context=contexts['agent_a']
  )

  assert replayed == {'result': 4}
  assert tool.live_calls == []


async def test_selected_tool_replays_repeated_calls_in_recorded_order(tmp_path):
  """Repeated selected calls return their recorded responses in order."""
  _write_recordings(
      tmp_path,
      [
          _recording(response={'result': 4}),
          _recording(response={'result': 7}, call_id='fc-2'),
      ],
  )
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(
      case_dir=tmp_path,
      replay_tool_responses={'agent_a': ['roll_die']},
  )
  tool = _SpyTool()

  await plugin.before_run_callback(invocation_context=invocation_context)
  first = await plugin.before_tool_callback(
      tool=tool, tool_args={'sides': 6}, tool_context=contexts['agent_a']
  )
  second = await plugin.before_tool_callback(
      tool=tool, tool_args={'sides': 6}, tool_context=contexts['agent_a']
  )

  assert [first, second] == [{'result': 4}, {'result': 7}]
  assert tool.live_calls == []


async def test_selection_is_scoped_to_agent(tmp_path):
  """A tool selected for one agent still executes for a sibling agent."""
  _write_recordings(
      tmp_path,
      [_recording(agent_name='agent_a'), _recording(agent_name='agent_b')],
  )
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(
      case_dir=tmp_path,
      agent_names=('agent_a', 'agent_b'),
      replay_tool_responses={'agent_a': ['roll_die']},
  )
  selected_tool = _SpyTool()
  other_tool = _SpyTool()

  await plugin.before_run_callback(invocation_context=invocation_context)
  selected_result = await plugin.before_tool_callback(
      tool=selected_tool,
      tool_args={'sides': 6},
      tool_context=contexts['agent_a'],
  )
  other_result = await plugin.before_tool_callback(
      tool=other_tool,
      tool_args={'sides': 6},
      tool_context=contexts['agent_b'],
  )

  assert selected_result == other_result == {'result': 4}
  assert selected_tool.live_calls == []
  assert other_tool.live_calls == [{'sides': 6}]


async def test_selected_tool_with_mismatched_args_never_executes(tmp_path):
  """A mismatched recorded call fails before a selected tool runs."""
  _write_recordings(tmp_path, [_recording(args={'sides': 6})])
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(
      case_dir=tmp_path,
      replay_tool_responses={'agent_a': ['roll_die']},
  )
  tool = _SpyTool()

  await plugin.before_run_callback(invocation_context=invocation_context)
  with pytest.raises(ReplayVerificationError, match='Tool args mismatch'):
    await plugin.before_tool_callback(
        tool=tool, tool_args={'sides': 20}, tool_context=contexts['agent_a']
    )

  assert tool.live_calls == []


async def test_unknown_selected_tool_fails_before_agent_runs(tmp_path):
  """A typo in the selection cannot silently leave a live tool enabled."""
  _write_recordings(tmp_path, [_recording()])
  plugin = ReplayPlugin()
  invocation_context, _ = await _make_invocation(
      case_dir=tmp_path,
      replay_tool_responses={'agent_a': ['roll_dice']},
  )

  with pytest.raises(ReplayConfigError, match='agent_a.roll_dice'):
    await plugin.before_run_callback(invocation_context=invocation_context)


async def test_selected_tool_without_recorded_response_never_executes(tmp_path):
  """A missing recorded response fails without falling back to a live tool."""
  recording = _recording()
  recording.tool_recording.tool_response = None
  _write_recordings(tmp_path, [recording])
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(
      case_dir=tmp_path,
      replay_tool_responses={'agent_a': ['roll_die']},
  )
  tool = _SpyTool()

  await plugin.before_run_callback(invocation_context=invocation_context)
  with pytest.raises(ReplayVerificationError, match='no tool_response'):
    await plugin.before_tool_callback(
        tool=tool, tool_args={'sides': 6}, tool_context=contexts['agent_a']
    )

  assert tool.live_calls == []


async def test_selected_agent_tool_is_rejected(tmp_path):
  """AgentTool has different execution semantics and cannot be response-only."""
  _write_recordings(tmp_path, [_recording(tool_name='child')])
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(
      case_dir=tmp_path,
      replay_tool_responses={'agent_a': ['child']},
  )
  child_tool = AgentTool(agent=testing_utils.create_test_agent(name='child'))

  await plugin.before_run_callback(invocation_context=invocation_context)
  with pytest.raises(ReplayConfigError, match='AgentTool'):
    await plugin.before_tool_callback(
        tool=child_tool,
        tool_args={'sides': 6},
        tool_context=contexts['agent_a'],
    )


async def test_before_run_reads_the_sse_file_in_sse_streaming_mode(tmp_path):
  """streaming_mode selects which recordings file is authoritative."""
  _write_recordings(
      tmp_path,
      [_recording(response={'result': 'non-streaming'})],
      file_name=_NON_STREAMING_FILE,
  )
  _write_recordings(
      tmp_path,
      [_recording(response={'result': 'streaming'})],
      file_name=_STREAMING_FILE,
  )
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(
      case_dir=tmp_path, streaming_mode='sse'
  )

  await plugin.before_run_callback(invocation_context=invocation_context)
  replayed = await plugin.before_tool_callback(
      tool=_SpyTool(),
      tool_args={'sides': 6},
      tool_context=contexts['agent_a'],
  )

  assert replayed == {'result': 'streaming'}


async def test_before_run_reads_the_plain_file_in_non_streaming_mode(tmp_path):
  """The mirror of the sse case, so a swapped file name cannot pass both."""
  _write_recordings(
      tmp_path,
      [_recording(response={'result': 'non-streaming'})],
      file_name=_NON_STREAMING_FILE,
  )
  _write_recordings(
      tmp_path,
      [_recording(response={'result': 'streaming'})],
      file_name=_STREAMING_FILE,
  )
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(
      case_dir=tmp_path, streaming_mode='none'
  )

  await plugin.before_run_callback(invocation_context=invocation_context)
  replayed = await plugin.before_tool_callback(
      tool=_SpyTool(),
      tool_args={'sides': 6},
      tool_context=contexts['agent_a'],
  )

  assert replayed == {'result': 'non-streaming'}


async def test_before_run_unsupported_streaming_mode_raises_value_error(
    tmp_path,
):
  """An unknown streaming mode must fail loudly, not pick a default file."""
  _write_recordings(tmp_path, [_recording()])
  plugin = ReplayPlugin()
  invocation_context, _ = await _make_invocation(
      case_dir=tmp_path, streaming_mode='bidi'
  )

  with pytest.raises(ValueError, match='Unsupported streaming mode: bidi'):
    await plugin.before_run_callback(invocation_context=invocation_context)


async def test_before_run_missing_recordings_file_raises_config_error(
    tmp_path,
):
  """A missing file is a configuration problem, reported with its path."""
  plugin = ReplayPlugin()
  invocation_context, _ = await _make_invocation(case_dir=tmp_path)

  with pytest.raises(ReplayConfigError, match='Recordings file not found'):
    await plugin.before_run_callback(invocation_context=invocation_context)


async def test_before_run_unparsable_recordings_raise_config_error(tmp_path):
  """Schema violations surface as ReplayConfigError, not a pydantic error."""
  (tmp_path / _NON_STREAMING_FILE).write_text(
      'recordings:\n  - user_message_index: 0\n    agent_name: a\n'
      '    tool_recordings: {}\n',
      encoding='utf-8',
  )
  plugin = ReplayPlugin()
  invocation_context, _ = await _make_invocation(case_dir=tmp_path)

  with pytest.raises(ReplayConfigError, match='Failed to load recordings'):
    await plugin.before_run_callback(invocation_context=invocation_context)


async def test_before_tool_without_loaded_state_raises_config_error(tmp_path):
  """Replaying without a preceding before_run is a misuse, not a silent pass."""
  _write_recordings(tmp_path, [_recording()])
  plugin = ReplayPlugin()
  _, contexts = await _make_invocation(case_dir=tmp_path)

  with pytest.raises(ReplayConfigError, match='Replay state not initialized'):
    await plugin.before_tool_callback(
        tool=_SpyTool(),
        tool_args={'sides': 6},
        tool_context=contexts['agent_a'],
    )


async def test_before_tool_tool_name_mismatch_raises_verification_error(
    tmp_path,
):
  """Calling a different tool than recorded fails verification."""
  _write_recordings(tmp_path, [_recording(tool_name='roll_die')])
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(case_dir=tmp_path)

  await plugin.before_run_callback(invocation_context=invocation_context)
  with pytest.raises(ReplayVerificationError) as exc_info:
    await plugin.before_tool_callback(
        tool=_SpyTool(name='flip_coin'),
        tool_args={'sides': 6},
        tool_context=contexts['agent_a'],
    )

  message = str(exc_info.value)
  assert 'Tool name mismatch' in message
  assert 'roll_die' in message
  assert 'flip_coin' in message


async def test_before_tool_args_mismatch_raises_verification_error(tmp_path):
  """The recorded args must match exactly, not just the tool name."""
  _write_recordings(tmp_path, [_recording(args={'sides': 6})])
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(case_dir=tmp_path)

  await plugin.before_run_callback(invocation_context=invocation_context)
  with pytest.raises(ReplayVerificationError) as exc_info:
    await plugin.before_tool_callback(
        tool=_SpyTool(),
        tool_args={'sides': 20},
        tool_context=contexts['agent_a'],
    )

  message = str(exc_info.value)
  assert 'Tool args mismatch' in message
  assert "'sides': 20" in message


async def test_before_tool_recording_without_tool_call_raises_value_error(
    tmp_path,
):
  recording = _recording()
  recording.tool_recording.tool_call = None
  _write_recordings(tmp_path, [recording])
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(case_dir=tmp_path)

  await plugin.before_run_callback(invocation_context=invocation_context)
  with pytest.raises(ValueError, match='no tool_call'):
    await plugin.before_tool_callback(
        tool=_SpyTool(),
        tool_args={'sides': 6},
        tool_context=contexts['agent_a'],
    )


async def test_before_tool_recording_without_tool_response_raises_value_error(
    tmp_path,
):
  recording = _recording()
  recording.tool_recording.tool_response = None
  _write_recordings(tmp_path, [recording])
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(case_dir=tmp_path)
  tool = _SpyTool()

  await plugin.before_run_callback(invocation_context=invocation_context)
  with pytest.raises(ValueError, match='no tool_response'):
    await plugin.before_tool_callback(
        tool=tool,
        tool_args={'sides': 6},
        tool_context=contexts['agent_a'],
    )
  assert tool.live_calls == [{'sides': 6}]


async def test_before_tool_beyond_recorded_calls_raises_verification_error(
    tmp_path,
):
  """An extra tool call past the end of the recordings is a replay failure."""
  _write_recordings(tmp_path, [_recording()])
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(case_dir=tmp_path)
  tool = _SpyTool()

  await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.before_tool_callback(
      tool=tool, tool_args={'sides': 6}, tool_context=contexts['agent_a']
  )

  with pytest.raises(ReplayVerificationError) as exc_info:
    await plugin.before_tool_callback(
        tool=tool, tool_args={'sides': 6}, tool_context=contexts['agent_a']
    )

  message = str(exc_info.value)
  assert 'more tool requests than expected' in message
  assert 'Expected 1' in message


async def test_before_tool_advances_a_separate_index_per_agent(tmp_path):
  """Each agent has its own replay index; a sibling's call must not shift it."""
  _write_recordings(
      tmp_path,
      [
          _recording(
              agent_name='agent_a', args={'sides': 6}, response={'result': 4}
          ),
          _recording(
              agent_name='agent_b', args={'sides': 8}, response={'result': 7}
          ),
          _recording(
              agent_name='agent_a', args={'sides': 20}, response={'result': 17}
          ),
      ],
  )
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(
      case_dir=tmp_path, agent_names=('agent_a', 'agent_b')
  )
  tool = _SpyTool()

  await plugin.before_run_callback(invocation_context=invocation_context)
  first_a = await plugin.before_tool_callback(
      tool=tool, tool_args={'sides': 6}, tool_context=contexts['agent_a']
  )
  first_b = await plugin.before_tool_callback(
      tool=tool, tool_args={'sides': 8}, tool_context=contexts['agent_b']
  )
  second_a = await plugin.before_tool_callback(
      tool=tool, tool_args={'sides': 20}, tool_context=contexts['agent_a']
  )

  assert [first_a, first_b, second_a] == [
      {'result': 4},
      {'result': 7},
      {'result': 17},
  ]


async def test_before_tool_ignores_recordings_for_other_user_messages(
    tmp_path,
):
  """Only the recordings for the configured user message are replayable."""
  _write_recordings(
      tmp_path,
      [
          _recording(
              user_message_index=0,
              args={'sides': 6},
              response={'result': 'first turn'},
          ),
          _recording(
              user_message_index=1,
              args={'sides': 20},
              response={'result': 'second turn'},
          ),
      ],
  )
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(
      case_dir=tmp_path, user_message_index=1
  )
  tool = _SpyTool()

  await plugin.before_run_callback(invocation_context=invocation_context)
  replayed = await plugin.before_tool_callback(
      tool=tool, tool_args={'sides': 20}, tool_context=contexts['agent_a']
  )

  assert replayed == {'result': 'second turn'}
  # The turn-0 recording is not available to this invocation.
  with pytest.raises(ReplayVerificationError, match='Expected 1'):
    await plugin.before_tool_callback(
        tool=tool, tool_args={'sides': 6}, tool_context=contexts['agent_a']
    )


async def test_after_run_discards_the_invocation_state(tmp_path):
  """Cleanup is observable: a later tool call no longer finds replay state."""
  _write_recordings(tmp_path, [_recording(), _recording(call_id='fc-2')])
  plugin = ReplayPlugin()
  invocation_context, contexts = await _make_invocation(case_dir=tmp_path)
  tool = _SpyTool()

  await plugin.before_run_callback(invocation_context=invocation_context)
  await plugin.before_tool_callback(
      tool=tool, tool_args={'sides': 6}, tool_context=contexts['agent_a']
  )
  await plugin.after_run_callback(invocation_context=invocation_context)

  with pytest.raises(ReplayConfigError, match='Replay state not initialized'):
    await plugin.before_tool_callback(
        tool=tool, tool_args={'sides': 6}, tool_context=contexts['agent_a']
    )
