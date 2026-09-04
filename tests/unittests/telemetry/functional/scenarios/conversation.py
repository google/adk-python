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

"""What every scenario says, is called, and bills for.

The names the graphs are built out of, the token usage the model reports, and
the canned conversations (``Turn`` sequences) it answers with. The graphs
themselves live in the ``*_scenario`` modules that share these.
"""

from __future__ import annotations

from google.genai.types import GenerateContentResponseUsageMetadata
from google.genai.types import Part

USER_PROMPT = "hello"
# The follow-up prompt of a scenario that spans two turns on one session.
SECOND_USER_PROMPT = "and again"
AGENT_NAME = "some_root_agent"
AGENT_DESCRIPTION = "A sample root agent."
BASE_INSTRUCTION = "you are helpful"
# ADK auto-appends agent identity info to the system instruction when the
# agent is invoked as the root of an InMemoryRunner directly.
FULL_SYSTEM_INSTRUCTION = (
    f"{BASE_INSTRUCTION}\n\n"
    f'You are an agent. Your internal name is "{AGENT_NAME}".'
    f' The description about you is "{AGENT_DESCRIPTION}".'
)
FINAL_TEXT = "text response"
# The model both inference instrumentations report. The OTel-instrumented
# configuration runs a real ``Gemini`` over a mocked SDK; the native one a
# ``MockModel`` renamed to match, so the two recordings differ only where the
# instrumentations do and not over the model name.
MODEL_NAME = "gemini-2.5-flash"
# The agent a multi-agent turn is handed on to, mid-turn, by transfer_to_agent.
SPECIALIST_AGENT_NAME = "some_specialist_agent"
SPECIALIST_AGENT_DESCRIPTION = "A sample specialist agent."
TOOL_NAME = "some_tool"
TOOL_DESCRIPTION = "A sample tool."
# What the scenario's tool raises for a case that asks it to fail.
TOOL_ERROR = ValueError("This tool always fails")
TOOL_ARGS = {"arg1": "val1"}
TOOL_RESULT_PREFIX = "processed "
TOOL_RESULT = f"{TOOL_RESULT_PREFIX}{TOOL_ARGS['arg1']}"

# The node scenario uses a workflow node whose output drives the agent's
# input. The workflow itself wraps the same agent.
WORKFLOW_NAME = "my_workflow"
# The root workflow invokes a nested workflow whose sole node produces the
# input for the agent. The nested workflow exercises the `gen_ai.workflow.nested`
# span attribute + metric dimension (only nested workflows carry it).
NESTED_WORKFLOW_NAME = "my_nested_workflow"
NODE_NAME = "some_node"
# The agent the nested workflow runs in place of its plain node.
NESTED_AGENT_NAME = "some_nested_agent"
NESTED_AGENT_DESCRIPTION = "A sample agent inside a nested workflow."
# The agent-tool graph: an agent whose tool wraps another agent, and the Runner
# boundary that tool puts between the two.
AGENT_TOOL_WORKFLOW_NAME = "my_agent_tool_workflow"
DELEGATING_AGENT_NAME = "some_delegating_agent"
DELEGATING_AGENT_DESCRIPTION = "A sample agent that delegates."
DELEGATE_AGENT_NAME = "some_delegate_agent"
DELEGATE_AGENT_DESCRIPTION = "A sample delegate agent."
NODE_RESULT = "some result"
NODE_USER_ID = "some_user"
NODE_APP_NAME = "some_app"

# Token usage reported by the two LLM turns. Every count is distinct, both
# across the two turns and across the buckets within a turn, so that a golden
# pins down which turn and which bucket a number came from: swapping any two of
# them changes the recording. No tool-use tokens: an ordinary FunctionTool's
# result is billed as prompt tokens, and the scenario's tool is one, so that
# bucket is a genuine zero.
#
# `gen_ai.usage.output_tokens` bills candidates + thoughts together, so the
# goldens record an output of 25 for the first turn and 50 for the second, and
# 250 input / 75 output summed over the invocation.
#
# Every turn reports usage: a real provider always does, and without it the two
# instrumentations would diverge for a reason that is about neither of them
# (ADK skips the token metric where the OTel instrumentor records zeros).
FIRST_TURN_PROMPT_TOKEN_COUNT = 100
FIRST_TURN_CACHED_TOKEN_COUNT = 40
FIRST_TURN_CANDIDATES_TOKEN_COUNT = 20
FIRST_TURN_THOUGHTS_TOKEN_COUNT = 5
FIRST_TURN_TOTAL_TOKEN_COUNT = 125
SECOND_TURN_PROMPT_TOKEN_COUNT = 150
SECOND_TURN_CACHED_TOKEN_COUNT = 60
SECOND_TURN_CANDIDATES_TOKEN_COUNT = 35
SECOND_TURN_THOUGHTS_TOKEN_COUNT = 15
SECOND_TURN_TOTAL_TOKEN_COUNT = 200
# Spent by the nested workflow's agent, in the one graph that runs one. Also
# distinct from both turns above, so the nested datapoint and the root one it
# rolls up into cannot be confused for each other.
NESTED_TURN_PROMPT_TOKEN_COUNT = 70
NESTED_TURN_CACHED_TOKEN_COUNT = 30
NESTED_TURN_CANDIDATES_TOKEN_COUNT = 10
NESTED_TURN_THOUGHTS_TOKEN_COUNT = 3
NESTED_TURN_TOTAL_TOKEN_COUNT = 83

FIRST_TURN_USAGE = GenerateContentResponseUsageMetadata(
    prompt_token_count=FIRST_TURN_PROMPT_TOKEN_COUNT,
    cached_content_token_count=FIRST_TURN_CACHED_TOKEN_COUNT,
    candidates_token_count=FIRST_TURN_CANDIDATES_TOKEN_COUNT,
    thoughts_token_count=FIRST_TURN_THOUGHTS_TOKEN_COUNT,
    total_token_count=FIRST_TURN_TOTAL_TOKEN_COUNT,
)
SECOND_TURN_USAGE = GenerateContentResponseUsageMetadata(
    prompt_token_count=SECOND_TURN_PROMPT_TOKEN_COUNT,
    cached_content_token_count=SECOND_TURN_CACHED_TOKEN_COUNT,
    candidates_token_count=SECOND_TURN_CANDIDATES_TOKEN_COUNT,
    thoughts_token_count=SECOND_TURN_THOUGHTS_TOKEN_COUNT,
    total_token_count=SECOND_TURN_TOTAL_TOKEN_COUNT,
)
NESTED_TURN_USAGE = GenerateContentResponseUsageMetadata(
    prompt_token_count=NESTED_TURN_PROMPT_TOKEN_COUNT,
    cached_content_token_count=NESTED_TURN_CACHED_TOKEN_COUNT,
    candidates_token_count=NESTED_TURN_CANDIDATES_TOKEN_COUNT,
    thoughts_token_count=NESTED_TURN_THOUGHTS_TOKEN_COUNT,
    total_token_count=NESTED_TURN_TOTAL_TOKEN_COUNT,
)

# One canned model response: what it answers, and what it bills for it.
Turn = tuple[Part, GenerateContentResponseUsageMetadata]

# The canonical 2-turn conversation: a call to ``some_tool``, then the answer.
TOOL_CALLING_TURNS: tuple[Turn, ...] = (
    (Part.from_function_call(name=TOOL_NAME, args=TOOL_ARGS), FIRST_TURN_USAGE),
    (Part.from_text(text=FINAL_TEXT), SECOND_TURN_USAGE),
)

# The graphs below run more than one agent off the one model, so their turns
# are consumed in the order the graph invokes the agents, one turn each.

# The root transfers mid-turn, then the specialist answers.
MULTI_AGENT_TURNS: tuple[Turn, ...] = (
    (
        Part.from_function_call(
            name="transfer_to_agent",
            args={"agent_name": SPECIALIST_AGENT_NAME},
        ),
        FIRST_TURN_USAGE,
    ),
    (Part.from_text(text=FINAL_TEXT), SECOND_TURN_USAGE),
)

# The delegating agent calls the tool, the delegate the tool starts answers,
# then the delegating agent answers with what came back.
AGENT_TOOL_TURNS: tuple[Turn, ...] = (
    (
        Part.from_function_call(
            name=DELEGATE_AGENT_NAME, args={"request": USER_PROMPT}
        ),
        FIRST_TURN_USAGE,
    ),
    (Part.from_text(text=NODE_RESULT), NESTED_TURN_USAGE),
    (Part.from_text(text=FINAL_TEXT), SECOND_TURN_USAGE),
)

# The nested workflow's agent answers first, since the graph feeds its output
# to the canonical agent, which then spends the usual two turns.
NESTED_WORKFLOW_TURNS: tuple[Turn, ...] = (
    (Part.from_text(text=NODE_RESULT), NESTED_TURN_USAGE),
) + TOOL_CALLING_TURNS

# One streamed model turn: the chunks it arrives in.
StreamedTurn = tuple[Turn, ...]

# The answer, split over the chunks it is streamed in.
STREAMED_TEXT_CHUNKS = ("text ", "response")
assert "".join(STREAMED_TEXT_CHUNKS) == FINAL_TEXT

# The canonical conversation, streamed: the tool call arrives whole, the answer
# in two chunks, each carrying the turn's usage as a real backend repeats it.
# Both instrumentations serve it over the same mocked SDK, so what the two
# recordings disagree about is how each reports a streamed turn, not what the
# aggregation in front of them did with it.
STREAMING_TURNS: tuple[StreamedTurn, ...] = (
    (
        (
            Part.from_function_call(name=TOOL_NAME, args=TOOL_ARGS),
            FIRST_TURN_USAGE,
        ),
    ),
    tuple(
        (Part.from_text(text=chunk), SECOND_TURN_USAGE)
        for chunk in STREAMED_TEXT_CHUNKS
    ),
)
