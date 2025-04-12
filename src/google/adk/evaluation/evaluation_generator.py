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

import importlib
import uuid

from google.genai import types

from ..agents.base_agent import BaseAgent
from ..agents.llm_agent import Agent
from ..agents.llm_agent import BeforeToolCallback
from ..agents.llm_agent import LlmAgent
from ..artifacts.in_memory_artifact_service import InMemoryArtifactService
from ..runners import Runner
from ..sessions.in_memory_session_service import InMemorySessionService
from ..sessions.session import Session
from .evaluation_constants import EvalConstants


class EvaluationGenerator:
  """Generates evaluation responses for agents."""

  @staticmethod
  def generate_responses(
      eval_dataset,
      agent_module_path,
      repeat_num=3,
      agent_name=None,
      initial_session={},
  ):
    """Returns evaluation responses for the given dataset and agent.

    Args:
      eval_dataset: The dataset that needs to be scraped for responses.
      agent_module_path: Path to the module that contains the root agent.
      repeat_num: Number of time the eval dataset should be repeated. This is
        usually done to remove uncertainty that a single run may bring.
      agent_name: The name of the agent that should be evaluated. This is
        usually the sub-agent.
      initial_session: Initial session for the eval data.
    """
    results = []

    for _ in range(repeat_num):
      for data in eval_dataset:
        results.append(
            EvaluationGenerator._process_query(
                data, agent_module_path, agent_name, initial_session
            )
        )

    return results

  @staticmethod
  def generate_responses_from_session(session_path, eval_dataset):
    """Returns evaluation responses by combining session data with eval data.

    Args:
      session_path: Path to a json file that contains session data.
      eval_dataset: The eval data set that should be combined with the session
        data.
    """
    results = []

    with open(session_path, "r") as f:
      session_data = Session.model_validate_json(f.read())
      print("loaded session", session_path)

    for data in eval_dataset:
      # load session data from session_path
      results.append(
          EvaluationGenerator._process_query_with_session(
              session_data,
              data,
          )
      )

    return results

  @staticmethod
  def _process_query(data, module_name, agent_name=None, initial_session={}):
    """Process a query using the agent and evaluation dataset."""
    module_path = f"{module_name}"
    agent_module = importlib.import_module(module_path)
    root_agent = agent_module.agent.root_agent

    reset_func = getattr(agent_module.agent, "reset_data", None)

    agent_to_evaluate = root_agent
    if agent_name:
      agent_to_evaluate = root_agent.find_agent(agent_name)
      assert agent_to_evaluate, f"Sub-Agent `{agent_name}` not found."

    return EvaluationGenerator._process_query_with_root_agent(
        data, agent_to_evaluate, reset_func, initial_session
    )

  @staticmethod
  def _process_query_with_root_agent(
      data,
      root_agent,
      reset_func,
      initial_session={},
      session_id=None,
      session_service=None,
      artifact_service=None,
  ):
    """Process a query using the agent and evaluation dataset."""

    # we don't know which tools belong to which agent
    # so we just apply to any agents that has certain tool outputs
    all_mock_tools = set()
    for eval_entry in data:
      expected_tool_use = eval_entry.get(EvalConstants.EXPECTED_TOOL_USE, [])
      for expected in expected_tool_use:
        if EvalConstants.MOCK_TOOL_OUTPUT in expected:
          all_mock_tools.add(expected[EvalConstants.TOOL_NAME])

    eval_data_copy = data.copy()
    EvaluationGenerator.apply_before_tool_callback(
        root_agent,
        lambda *args: EvaluationGenerator.before_tool_callback(
            *args, eval_dataset=eval_data_copy
        ),
        all_mock_tools,
    )

    if not session_service:
      session_service = InMemorySessionService()

    app_name = initial_session.get("app_name", "EvaluationGenerator")
    user_id = initial_session.get("user_id", "test_user_id")
    session_id = session_id if session_id else str(uuid.uuid4())

    _ = session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        state=initial_session.get("state", {}),
        session_id=session_id,
    )

    if not artifact_service:
      artifact_service = InMemoryArtifactService()
    runner = Runner(
        app_name=app_name,
        agent=root_agent,
        artifact_service=artifact_service,
        session_service=session_service,
    )

    # Reset agent state for each query
    if callable(reset_func):
      reset_func()

    responses = data.copy()

    for index, eval_entry in enumerate(responses):
      response = None
      query = eval_entry["query"]
      content = types.Content(role="user", parts=[types.Part(text=query)])
      turn_actual_tool_uses = []

      for event in runner.run(
          user_id=user_id, session_id=session_id, new_message=content
      ):
        if event.is_final_response() and event.content and event.content.parts:
          response = event.content.parts[0].text
        elif event.get_function_calls():
          for call in event.get_function_calls():
            turn_actual_tool_uses.append({
                EvalConstants.TOOL_NAME: call.name,
                EvalConstants.TOOL_INPUT: call.args,
            })

      responses[index]["actual_tool_use"] = turn_actual_tool_uses
      responses[index]["response"] = response

    return responses

  @staticmethod
  def _process_query_with_session(session_data, data):
    """Process the queries using the existing session data without invoking the runner."""
    responses = data.copy()

    # Iterate through the provided queries and align them with the session events
    for index, eval_entry in enumerate(responses):
      query = eval_entry["query"]
      actual_tool_uses = []
      response = None

      # Search for the corresponding session events
      for event in session_data.events:
        # Match the query to a user event
        if (
            event.author == "user"
            and event.content
            and event.content.parts
            and event.content.parts[0].text == query
        ):
          # Look for subsequent tool usage or model responses
          for subsequent_event in session_data.events:
            if subsequent_event.invocation_id == event.invocation_id:
              # Extract tool usage
              if subsequent_event.content.parts[0].function_call:
                call = subsequent_event.content.parts[0].function_call
                actual_tool_uses.append(
                    {"tool_name": call.name, "tool_input": call.args}
                )
              # Extract final response
              elif subsequent_event.author != "user":
                response = subsequent_event.content.parts[0].text

      # Update the results for the current query
      responses[index]["actual_tool_use"] = actual_tool_uses
      responses[index]["response"] = response
    return responses

  @staticmethod
  def before_tool_callback(tool, args, tool_context, eval_dataset):
    """Intercept specific tool calls and return predefined outputs

    from eval_dataset.
    """
    for index, eval_entry in enumerate(eval_dataset):
      expected_tool_use = eval_entry.get("expected_tool_use", [])
      for expected in expected_tool_use:
        if (
            EvalConstants.MOCK_TOOL_OUTPUT in expected
            and tool.name == expected[EvalConstants.TOOL_NAME]
            and args == expected.get(EvalConstants.TOOL_INPUT, {})
        ):
          # pop the matched entry so we don't rematch again
          eval_dataset.pop(index)
          return {"result": expected[EvalConstants.MOCK_TOOL_OUTPUT]}

    return None

  @staticmethod
  def apply_before_tool_callback(
      agent: BaseAgent,
      callback: BeforeToolCallback,
      all_mock_tools: set[str],
  ):
    """Recursively apply the before_tool_callback to the root agent and all its subagents."""
    # Check if the agent has tools that are defined by evalset.
    # We use function names to check if tools match
    if not isinstance(agent, Agent) and not isinstance(agent, LlmAgent):
      return

    for tool in agent.canonical_tools:
      tool_name = tool.name
      if tool_name in all_mock_tools:
        agent.before_tool_callback = callback

    # Apply recursively to subagents if they exist
    for sub_agent in agent.sub_agents:
      EvaluationGenerator.apply_before_tool_callback(
          sub_agent, callback, all_mock_tools
      )
