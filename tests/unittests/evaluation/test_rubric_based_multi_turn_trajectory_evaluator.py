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

from google.adk.evaluation.app_details import AgentDetails
from google.adk.evaluation.app_details import AppDetails
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_case import InvocationEvent
from google.adk.evaluation.eval_case import InvocationEvents
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import JudgeModelOptions
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.eval_metrics import RubricsBasedCriterion
from google.adk.evaluation.eval_rubrics import Rubric
from google.adk.evaluation.eval_rubrics import RubricContent
from google.adk.evaluation.rubric_based_multi_turn_trajectory_evaluator import RubricBasedMultiTurnTrajectoryEvaluator
from google.genai import types as genai_types
import pytest

_RUBRICS = [
    Rubric(
        rubric_id="1",
        rubric_content=RubricContent(
            text_property="The agent uses the correct tool."
        ),
        type="TOOL_USAGE",
    ),
    Rubric(
        rubric_id="2",
        rubric_content=RubricContent(
            text_property="The agent fulfills the user intent."
        ),
        type="FULFILL_USER_INTENT",
    ),
]


def _make_evaluator(
    rubrics: list[Rubric] | None = None,
) -> RubricBasedMultiTurnTrajectoryEvaluator:
  """Helper to build an evaluator with the given rubrics."""
  rubrics = rubrics or _RUBRICS
  criterion = RubricsBasedCriterion(
      threshold=0.5,
      rubrics=rubrics,
      judge_model_options=JudgeModelOptions(
          judge_model_config=None,
          num_samples=3,
      ),
  )
  metric = EvalMetric(
      metric_name=PrebuiltMetrics.RUBRIC_BASED_MULTI_TURN_TRAJECTORY_QUALITY_V1.value,
      threshold=0.5,
      criterion=criterion,
  )
  return RubricBasedMultiTurnTrajectoryEvaluator(metric)


def _make_invocation(
    user_text: str,
    agent_text: str | None = None,
    invocation_id: str = "",
    rubrics: list[Rubric] | None = None,
    app_details: AppDetails | None = None,
    intermediate_data: InvocationEvents | None = None,
) -> Invocation:
  """Helper to build an Invocation."""
  return Invocation(
      invocation_id=invocation_id,
      user_content=genai_types.Content(
          parts=[genai_types.Part(text=user_text)]
      ),
      final_response=(
          genai_types.Content(parts=[genai_types.Part(text=agent_text)])
          if agent_text
          else None
      ),
      rubrics=rubrics,
      app_details=app_details,
      intermediate_data=intermediate_data,
  )


class TestFormatAutoRaterPrompt:
  """Tests for format_auto_rater_prompt."""

  def test_basic_dialogue_and_rubrics_in_prompt(self):
    """Tests that user dialogue and rubrics appear in the generated prompt."""
    evaluator = _make_evaluator()
    invocation = _make_invocation(
        user_text="What is the balance?",
        agent_text="Your balance is $100.",
        rubrics=_RUBRICS,
    )
    # Simulate evaluate_invocations dialogue assembly by setting internal state.
    evaluator._formatted_dialogue = "USER TURN 1: What is the balance?"
    evaluator._formatted_instructions = ""
    evaluator._formatted_tools = ""

    prompt = evaluator.format_auto_rater_prompt(invocation, None)

    assert "USER TURN 1: What is the balance?" in prompt
    assert "The agent uses the correct tool." in prompt
    assert "The agent fulfills the user intent." in prompt
    assert "TOOL_USAGE" in prompt
    assert "FULFILL_USER_INTENT" in prompt

  def test_prompt_includes_agent_instructions_and_tools(self):
    """Tests that agent instructions and tools are inserted into the prompt."""
    evaluator = _make_evaluator()
    invocation = _make_invocation(
        user_text="Transfer funds",
        rubrics=_RUBRICS,
    )
    evaluator._formatted_dialogue = "USER TURN 1: Transfer funds"
    evaluator._formatted_instructions = (
        "Agent banking_agent Instructions:\nYou are a banking assistant."
    )
    evaluator._formatted_tools = (
        "Agent: banking_agent\n- transfer_funds: Transfer money between"
        " accounts."
    )

    prompt = evaluator.format_auto_rater_prompt(invocation, None)

    assert "You are a banking assistant." in prompt
    assert "transfer_funds" in prompt


class TestDialogueAssembly:
  """Tests for the dialogue assembly logic in evaluate_invocations.

  These test the internal dialogue construction by calling evaluate_invocations
  and inspecting self._formatted_dialogue.
  """

  @pytest.fixture
  def evaluator(self):
    return _make_evaluator()

  @pytest.mark.asyncio
  async def test_single_turn_user_and_agent(self, evaluator):
    """Tests that a single turn assembles user and agent dialogue."""
    invocations = [
        _make_invocation(
            user_text="Hello",
            agent_text="Hi there!",
            invocation_id="agent1",
            rubrics=_RUBRICS,
        ),
    ]
    # We need to mock the super().evaluate_invocations call since it calls
    # the LLM. Instead, we just test the dialogue assembly part directly.
    evaluator._formatted_dialogue = None

    # Manually run the dialogue assembly portion
    evaluator._assemble_dialogue_history(invocations)

    assert "USER TURN 1: Hello" in evaluator._formatted_dialogue
    assert "AGENT (agent) TURN 1: Hi there!" in evaluator._formatted_dialogue

  @pytest.mark.asyncio
  async def test_multi_turn_dialogue(self, evaluator):
    """Tests dialogue assembly across multiple turns."""
    invocations = [
        _make_invocation(
            user_text="Check my balance",
            agent_text="Your balance is $100.",
            invocation_id="agent1",
            rubrics=_RUBRICS,
        ),
        _make_invocation(
            user_text="Transfer $50",
            agent_text="Transfer complete.",
            invocation_id="agent1",
            rubrics=_RUBRICS,
        ),
    ]
    evaluator._assemble_dialogue_history(invocations)

    assert "USER TURN 1: Check my balance" in evaluator._formatted_dialogue
    assert (
        "AGENT (agent) TURN 1: Your balance is $100."
        in evaluator._formatted_dialogue
    )
    assert "USER TURN 2: Transfer $50" in evaluator._formatted_dialogue
    assert (
        "AGENT (agent) TURN 2: Transfer complete."
        in evaluator._formatted_dialogue
    )

  @pytest.mark.asyncio
  async def test_intermediate_events_with_function_calls(self, evaluator):
    """Tests that intermediate function calls and responses appear in dialogue."""
    tool_call_part = genai_types.Part(
        function_call=genai_types.FunctionCall(
            name="get_balance", args={"account_id": "123"}
        )
    )
    tool_response_part = genai_types.Part(
        function_response=genai_types.FunctionResponse(
            name="get_balance", response={"balance": 100}
        )
    )
    intermediate_data = InvocationEvents(
        invocation_events=[
            InvocationEvent(
                author="banking_agent",
                content=genai_types.Content(parts=[tool_call_part]),
            ),
            InvocationEvent(
                author="banking_agent",
                content=genai_types.Content(parts=[tool_response_part]),
            ),
        ]
    )
    invocations = [
        _make_invocation(
            user_text="What is my balance?",
            agent_text="Your balance is $100.",
            invocation_id="banking_agent",
            rubrics=_RUBRICS,
            intermediate_data=intermediate_data,
        ),
    ]
    evaluator._assemble_dialogue_history(invocations)

    assert "get_balance" in evaluator._formatted_dialogue
    assert '"account_id": "123"' in evaluator._formatted_dialogue
    assert '"balance": 100' in evaluator._formatted_dialogue

  @pytest.mark.asyncio
  async def test_app_details_instructions_and_tools(self, evaluator):
    """Tests that app_details instructions and tools are captured."""
    tool = genai_types.Tool(
        function_declarations=[
            genai_types.FunctionDeclaration(
                name="transfer_funds",
                description="Transfer money between accounts.",
            )
        ]
    )
    app_details = AppDetails(
        agent_details={
            "banking_agent": AgentDetails(
                name="banking_agent",
                instructions="You are a banking assistant.",
                tool_declarations=[tool],
            )
        },
    )
    invocations = [
        _make_invocation(
            user_text="Transfer $50",
            agent_text="Done.",
            invocation_id="banking_agent",
            rubrics=_RUBRICS,
            app_details=app_details,
        ),
    ]
    evaluator._assemble_dialogue_history(invocations)

    assert "You are a banking assistant." in evaluator._formatted_instructions
    assert "transfer_funds" in evaluator._formatted_tools
    assert "Transfer money between accounts." in evaluator._formatted_tools

  @pytest.mark.asyncio
  async def test_invocation_without_user_content(self, evaluator):
    """Tests that invocations with no user text parts are handled gracefully."""
    invocations = [
        Invocation(
            user_content=genai_types.Content(parts=[]),
            final_response=genai_types.Content(
                parts=[genai_types.Part(text="Agent response.")]
            ),
            invocation_id="agent1",
            rubrics=_RUBRICS,
        ),
    ]
    evaluator._assemble_dialogue_history(invocations)

    # No user turn should appear, but agent turn should
    assert "USER TURN" not in evaluator._formatted_dialogue
    assert (
        "AGENT (agent) TURN 1: Agent response." in evaluator._formatted_dialogue
    )
