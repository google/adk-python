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

from typing import Optional

from google.adk.agents import Agent
from google.adk.agents import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai.types import Part
from pydantic import BaseModel
from pytest import mark

from .. import testing_utils

function_call_custom = Part.from_function_call(
    name='tool_agent', args={'custom_input': 'test1'}
)

function_call_no_schema = Part.from_function_call(
    name='tool_agent', args={'request': 'test1'}
)

function_response_custom = Part.from_function_response(
    name='tool_agent', response={'custom_output': 'response1'}
)

function_response_no_schema = Part.from_function_response(
    name='tool_agent', response={'result': 'response1'}
)


# Moved SubToolOutput to module level
class SubToolOutput(BaseModel):
    status: str
    value: int


def change_state_callback(callback_context: CallbackContext):
  callback_context.state['state_1'] = 'changed_value'
  print('change_state_callback: ', callback_context.state)


def test_no_schema():
  mock_model = testing_utils.MockModel.create(
      responses=[
          function_call_no_schema,
          'response1',
          'response2',
      ]
  )

  tool_agent = Agent(
      name='tool_agent',
      model=mock_model,
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=tool_agent)],
  )

  runner = testing_utils.InMemoryRunner(root_agent)

  assert testing_utils.simplify_events(runner.run('test1')) == [
      ('root_agent', function_call_no_schema),
      ('root_agent', function_response_no_schema),
      ('root_agent', 'response2'),
  ]


def test_update_state():
  """The agent tool can read and change parent state."""

  mock_model = testing_utils.MockModel.create(
      responses=[
          function_call_no_schema,
          '{"custom_output": "response1"}',
          'response2',
      ]
  )

  tool_agent = Agent(
      name='tool_agent',
      model=mock_model,
      instruction='input: {state_1}',
      before_agent_callback=change_state_callback,
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=tool_agent)],
  )

  runner = testing_utils.InMemoryRunner(root_agent)
  runner.session.state['state_1'] = 'state1_value'

  runner.run('test1')
  assert (
      'input: changed_value' in mock_model.requests[1].config.system_instruction
  )
  assert runner.session.state['state_1'] == 'changed_value'


def test_update_artifacts():
  """The agent tool can read and write artifacts."""

  async def before_tool_agent(callback_context: CallbackContext):
    # Artifact 1 should be available in the tool agent.
    artifact = await callback_context.load_artifact('artifact_1')
    await callback_context.save_artifact(
        'artifact_2', Part.from_text(text=artifact.text + ' 2')
    )

  tool_agent = SequentialAgent(
      name='tool_agent',
      before_agent_callback=before_tool_agent,
  )

  async def before_main_agent(callback_context: CallbackContext):
    await callback_context.save_artifact(
        'artifact_1', Part.from_text(text='test')
    )

  async def after_main_agent(callback_context: CallbackContext):
    # Artifact 2 should be available after the tool agent.
    artifact_2 = await callback_context.load_artifact('artifact_2')
    await callback_context.save_artifact(
        'artifact_3', Part.from_text(text=artifact_2.text + ' 3')
    )

  mock_model = testing_utils.MockModel.create(
      responses=[function_call_no_schema, 'response2']
  )
  root_agent = Agent(
      name='root_agent',
      before_agent_callback=before_main_agent,
      after_agent_callback=after_main_agent,
      tools=[AgentTool(agent=tool_agent)],
      model=mock_model,
  )

  runner = testing_utils.InMemoryRunner(root_agent)
  runner.run('test1')

  artifacts_path = f'test_app/test_user/{runner.session_id}'
  assert runner.runner.artifact_service.artifacts == {
      f'{artifacts_path}/artifact_1': [Part.from_text(text='test')],
      f'{artifacts_path}/artifact_2': [Part.from_text(text='test 2')],
      f'{artifacts_path}/artifact_3': [Part.from_text(text='test 2 3')],
  }


@mark.parametrize(
    'env_variables',
    [
        'GOOGLE_AI',
        # TODO(wanyif): re-enable after fix.
        # 'VERTEX',
    ],
    indirect=True,
)
def test_custom_schema():
  class CustomInput(BaseModel):
    custom_input: str

  class CustomOutput(BaseModel):
    custom_output: str

  mock_model = testing_utils.MockModel.create(
      responses=[
          function_call_custom,
          '{"custom_output": "response1"}',
          'response2',
      ]
  )

  tool_agent = Agent(
      name='tool_agent',
      model=mock_model,
      input_schema=CustomInput,
      output_schema=CustomOutput,
      output_key='tool_output',
  )

  root_agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[AgentTool(agent=tool_agent)],
  )

  runner = testing_utils.InMemoryRunner(root_agent)
  runner.session.state['state_1'] = 'state1_value'

  assert testing_utils.simplify_events(runner.run('test1')) == [
      ('root_agent', function_call_custom),
      ('root_agent', function_response_custom),
      ('root_agent', 'response2'),
  ]

  assert runner.session.state['tool_output'] == {'custom_output': 'response1'}

  assert len(mock_model.requests) == 3
  # The second request is the tool agent request.
  assert mock_model.requests[1].config.response_schema == CustomOutput
  assert mock_model.requests[1].config.response_mime_type == 'application/json'


def test_agent_tool_skip_summarization_in_sub_agent_callback():
    """Tests that skip_summarization set in sub-agent's after_tool_callback propagates raw output."""

    # SubToolOutput is now defined at module level

    def simple_sub_tool(query: str) -> SubToolOutput:
        """A simple tool for the sub-agent."""
        # query is not used, just to make it a valid tool schema
        return SubToolOutput(status="success", value=123)

    expected_raw_output = {"status": "success", "value": 123}

    def sub_agent_after_tool_cb(tool: BaseTool, args: dict, tool_context: ToolContext, tool_response: dict) -> Optional[dict]:
        tool_context.actions.skip_summarization = True
        # Return None to indicate the original tool_response should be used,
        # but with skip_summarization now active on the context.
        return None

    # Mock for the sub-agent's LLM: it should decide to call 'simple_sub_tool'
    mock_sub_agent_model = testing_utils.MockModel.create(
        responses=[
            Part.from_function_call(name='simple_sub_tool', args={'query': 'test'})
            # No further LLM response needed from sub-agent as skip_summarization should make tool output final
        ]
    )

    sub_agent = LlmAgent(
        name='sub_agent_for_skip_test',
        model=mock_sub_agent_model,
        tools=[simple_sub_tool],
        after_tool_callback=sub_agent_after_tool_cb,
        # No output_schema on sub_agent, so it would normally summarize.
    )

    # AgentTool wrapping the sub_agent
    # AgentTool's own skip_summarization is False by default. We are testing the callback mechanism.
    agent_as_tool = AgentTool(agent=sub_agent)

    # Mock for the root agent's LLM: it should decide to call the 'agent_as_tool'
    # The args for agent_as_tool depend on sub_agent.input_schema.
    # Since sub_agent has no input_schema, AgentTool creates a default {'request': types.Schema(type=types.Type.STRING)}
    mock_root_model = testing_utils.MockModel.create(
        responses=[
            Part.from_function_call(name='sub_agent_for_skip_test', args={'request': 'trigger sub agent'}),
            "Root agent summarizing: " # This is what the root agent would say *after* getting AgentTool's output.
                                     # The key is what AgentTool *returned* to it.
        ]
    )

    root_agent = LlmAgent(
        name='root_agent_for_skip_test',
        model=mock_root_model,
        tools=[agent_as_tool],
    )

    runner = testing_utils.InMemoryRunner(root_agent)
    events = runner.run('hello from root')

    simplified_events = testing_utils.simplify_events(events)

    # Expected sequence:
    # 1. Root agent calls AgentTool (sub_agent_for_skip_test)
    # 2. AgentTool (sub_agent_for_skip_test) returns the raw output from simple_sub_tool
    #    due to skip_summarization in sub-agent's callback.
    #    The AgentTool (Step 1 fix) should return the dict directly if it's JSON.
    #    The sub-agent's flow (Step 2 fix) should yield a TextPart with JSON string.
    #    AgentTool then parses this TextPart if it's JSON.
    # 3. Root agent gets this raw output and then generates its final response.

    # Print events for debugging if the test fails
    # print("\nSimplified Events for skip_summarization test:")
    # for e_type, e_content in simplified_events:
    #     print(f"  {e_type}: {e_content}")

    assert simplified_events == [
        ('root_agent_for_skip_test', Part.from_function_call(name='sub_agent_for_skip_test', args={'request': 'trigger sub agent'})),
        # The FunctionResponsePart's 'response' field should contain the raw dict
        ('root_agent_for_skip_test', Part.from_function_response(name='sub_agent_for_skip_test', response=expected_raw_output)),
            ('root_agent_for_skip_test', "Root agent summarizing:"), # Removed trailing space
    ]

    # Also check the actual calls to models to ensure sub-agent didn't try to summarize.
    # mock_sub_agent_model.requests should only contain one request (the one that led to calling simple_sub_tool).
    # It should not have been called again to summarize the output of simple_sub_tool.
    assert len(mock_sub_agent_model.requests) == 1
    # The first (and only) request to sub-agent's model led to the tool call.
    # It should not have any FunctionResponsePart from simple_sub_tool in its history trying to make it summarize.
    # The history for this call would be the initial 'trigger sub agent' converted by AgentTool.
    sub_agent_first_request_messages = mock_sub_agent_model.requests[0].contents
    # print("\nSub Agent Model Request History:")
    # for msg in sub_agent_first_request_messages:
    #    print(f"  Role: {msg.role}, Parts: {msg.parts}")

    # Ensure no FunctionResponsePart for 'simple_sub_tool' was sent back to the sub-agent's LLM
    for message in sub_agent_first_request_messages:
        for part in message.parts:
            if part.function_response and part.function_response.name == 'simple_sub_tool':
                assert False, "Sub-agent's LLM received its own tool's response for summarization, which skip_summarization should prevent."
