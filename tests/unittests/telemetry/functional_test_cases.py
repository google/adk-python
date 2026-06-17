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

"""Hand-written expected telemetry shapes for the non-node functional tests.

Each ``EXPECTED_*`` is a complete ``SpanDigest`` tree (with per-span
``LogDigest`` lists nested in) describing what telemetry the canonical
agent + tool + 2-LLM-turn scenario should emit under one specific
combination of:

* ``OTEL_SEMCONV_STABILITY_OPT_IN``
* ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``

The cases are deliberately repetitive and verbose. The point is to give
"at-a-glance" visibility into what telemetry should look like under each
config -- DO NOT factor the construction into helpers.
"""

from __future__ import annotations

from .functional_test_helpers import AGENT_DESCRIPTION
from .functional_test_helpers import AGENT_NAME
from .functional_test_helpers import EXPERIMENTAL_OPT_IN
from .functional_test_helpers import FINAL_TEXT
from .functional_test_helpers import FULL_SYSTEM_INSTRUCTION
from .functional_test_helpers import FunctionalTestCase
from .functional_test_helpers import GEN_AI_CHOICE_EVENT
from .functional_test_helpers import GEN_AI_COMPLETION_DETAILS_EVENT
from .functional_test_helpers import GEN_AI_SYSTEM_MESSAGE_EVENT
from .functional_test_helpers import GEN_AI_USER_MESSAGE_EVENT
from .functional_test_helpers import LogDigest
from .functional_test_helpers import PRESENT
from .functional_test_helpers import SpanDigest
from .functional_test_helpers import TOOL_ARGS
from .functional_test_helpers import TOOL_DESCRIPTION
from .functional_test_helpers import TOOL_NAME
from .functional_test_helpers import TOOL_RESULT
from .functional_test_helpers import USER_PROMPT

# ---------------------------------------------------------------------------
# Stable semconv, OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false
# ---------------------------------------------------------------------------

EXPECTED_STABLE_NO_CAPTURE = SpanDigest(
    name="invocation",
    attributes={},
    children=[
        SpanDigest(
            name="invoke_agent some_root_agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.description": AGENT_DESCRIPTION,
                "gen_ai.agent.name": AGENT_NAME,
                "gen_ai.conversation.id": PRESENT,
            },
            children=[
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.system": "gemini",
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_CHOICE_EVENT,
                                    body={
                                        "content": "<elided>",
                                        "index": 0,
                                        "finish_reason": "STOP",
                                    },
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_SYSTEM_MESSAGE_EVENT,
                                    body={"content": "<elided>"},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={"content": "<elided>"},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                            ],
                            children=[
                                SpanDigest(
                                    name="execute_tool some_tool",
                                    attributes={
                                        "gen_ai.operation.name": "execute_tool",
                                        "gen_ai.tool.description": (
                                            TOOL_DESCRIPTION
                                        ),
                                        "gen_ai.tool.name": TOOL_NAME,
                                        "gen_ai.tool.type": "FunctionTool",
                                        "gcp.vertex.agent.llm_request": "{}",
                                        "gcp.vertex.agent.llm_response": "{}",
                                        "gcp.vertex.agent.tool_call_args": "{}",
                                        "gen_ai.tool.call.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.tool_response": "{}",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.system": "gemini",
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_CHOICE_EVENT,
                                    body={
                                        "content": "<elided>",
                                        "index": 0,
                                        "finish_reason": "STOP",
                                    },
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_SYSTEM_MESSAGE_EVENT,
                                    body={"content": "<elided>"},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={"content": "<elided>"},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={"content": "<elided>"},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={"content": "<elided>"},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Stable semconv, OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
# ---------------------------------------------------------------------------

EXPECTED_STABLE_CAPTURE = SpanDigest(
    name="invocation",
    attributes={},
    children=[
        SpanDigest(
            name="invoke_agent some_root_agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.description": AGENT_DESCRIPTION,
                "gen_ai.agent.name": AGENT_NAME,
                "gen_ai.conversation.id": PRESENT,
            },
            children=[
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.system": "gemini",
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_CHOICE_EVENT,
                                    body={
                                        "content": {
                                            "parts": [{
                                                "function_call": {
                                                    "args": TOOL_ARGS,
                                                    "name": TOOL_NAME,
                                                }
                                            }],
                                            "role": "model",
                                        },
                                        "index": 0,
                                        "finish_reason": "STOP",
                                    },
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_SYSTEM_MESSAGE_EVENT,
                                    body={"content": FULL_SYSTEM_INSTRUCTION},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={
                                        "content": {
                                            "parts": [{"text": USER_PROMPT}],
                                            "role": "user",
                                        }
                                    },
                                    attributes={
                                        "gen_ai.system": "gemini",
                                        "user.id": "test_user",
                                    },
                                ),
                            ],
                            children=[
                                SpanDigest(
                                    name="execute_tool some_tool",
                                    attributes={
                                        "gen_ai.operation.name": "execute_tool",
                                        "gen_ai.tool.description": (
                                            TOOL_DESCRIPTION
                                        ),
                                        "gen_ai.tool.name": TOOL_NAME,
                                        "gen_ai.tool.type": "FunctionTool",
                                        "gcp.vertex.agent.llm_request": "{}",
                                        "gcp.vertex.agent.llm_response": "{}",
                                        "gcp.vertex.agent.tool_call_args": "{}",
                                        "gen_ai.tool.call.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.tool_response": "{}",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.system": "gemini",
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_CHOICE_EVENT,
                                    body={
                                        "content": {
                                            "parts": [{"text": FINAL_TEXT}],
                                            "role": "model",
                                        },
                                        "index": 0,
                                        "finish_reason": "STOP",
                                    },
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_SYSTEM_MESSAGE_EVENT,
                                    body={"content": FULL_SYSTEM_INSTRUCTION},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={
                                        "content": {
                                            "parts": [{
                                                "function_call": {
                                                    "args": TOOL_ARGS,
                                                    "name": TOOL_NAME,
                                                }
                                            }],
                                            "role": "model",
                                        }
                                    },
                                    attributes={
                                        "gen_ai.system": "gemini",
                                        "user.id": "test_user",
                                    },
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={
                                        "content": {
                                            "parts": [{
                                                "function_response": {
                                                    "name": TOOL_NAME,
                                                    "response": {
                                                        "result": TOOL_RESULT
                                                    },
                                                }
                                            }],
                                            "role": "user",
                                        }
                                    },
                                    attributes={
                                        "gen_ai.system": "gemini",
                                        "user.id": "test_user",
                                    },
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={
                                        "content": {
                                            "parts": [{"text": USER_PROMPT}],
                                            "role": "user",
                                        }
                                    },
                                    attributes={
                                        "gen_ai.system": "gemini",
                                        "user.id": "test_user",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Experimental semconv,
# OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=no_content
# ---------------------------------------------------------------------------
# `no_content` is not one of the recognized capturing modes, so it falls into
# the "no content" branch on both the span and the log: function-tool params
# are stripped to None, no input/output messages, no system instructions.

EXPECTED_EXPERIMENTAL_NO_CONTENT = SpanDigest(
    name="invocation",
    attributes={},
    children=[
        SpanDigest(
            name="invoke_agent some_root_agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.description": AGENT_DESCRIPTION,
                "gen_ai.agent.name": AGENT_NAME,
                "gen_ai.conversation.id": PRESENT,
            },
            children=[
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.tool.definitions": [{
                                    "name": TOOL_NAME,
                                    "description": TOOL_DESCRIPTION,
                                    "type": "function",
                                }],
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.tool.definitions": [{
                                            "name": TOOL_NAME,
                                            "description": TOOL_DESCRIPTION,
                                            "type": "function",
                                        }],
                                    },
                                ),
                            ],
                            children=[
                                SpanDigest(
                                    name="execute_tool some_tool",
                                    attributes={
                                        "gen_ai.operation.name": "execute_tool",
                                        "gen_ai.tool.description": (
                                            TOOL_DESCRIPTION
                                        ),
                                        "gen_ai.tool.name": TOOL_NAME,
                                        "gen_ai.tool.type": "FunctionTool",
                                        "gcp.vertex.agent.llm_request": "{}",
                                        "gcp.vertex.agent.llm_response": "{}",
                                        "gcp.vertex.agent.tool_call_args": "{}",
                                        "gen_ai.tool.call.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.tool_response": "{}",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.tool.definitions": [{
                                    "name": TOOL_NAME,
                                    "description": TOOL_DESCRIPTION,
                                    "type": "function",
                                }],
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.tool.definitions": [{
                                            "name": TOOL_NAME,
                                            "description": TOOL_DESCRIPTION,
                                            "type": "function",
                                        }],
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Experimental semconv,
# OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=span_only
# ---------------------------------------------------------------------------
# Span gets full op-details (input/output messages, system instructions, full
# tool definitions). Log carries the no-content view.

# Tool definition with full parameters (only on spans/logs that get content).
_TOOL_DEFINITION_FULL = {
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "parameters": {
        "properties": {"arg1": {"title": "Arg1", "type": "string"}},
        "required": ["arg1"],
        "title": f"{TOOL_NAME}Params",
        "type": "object",
    },
    "type": "function",
}

_TOOL_DEFINITION_NO_CONTENT = {
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "type": "function",
}

_SYSTEM_INSTRUCTIONS = [{"content": FULL_SYSTEM_INSTRUCTION, "type": "text"}]

_TURN_1_INPUT_MESSAGES = [{
    "role": "user",
    "parts": [{"content": USER_PROMPT, "type": "text"}],
}]

_TURN_1_OUTPUT_MESSAGES = [{
    "role": "assistant",
    "parts": [{
        "id": f"{TOOL_NAME}_0",
        "name": TOOL_NAME,
        "arguments": TOOL_ARGS,
        "type": "tool_call",
    }],
    "finish_reason": "stop",
}]

_TURN_2_INPUT_MESSAGES = [
    {
        "role": "user",
        "parts": [{"content": USER_PROMPT, "type": "text"}],
    },
    {
        "role": "assistant",
        "parts": [{
            "id": f"{TOOL_NAME}_0",
            "name": TOOL_NAME,
            "arguments": TOOL_ARGS,
            "type": "tool_call",
        }],
    },
    {
        "role": "user",
        "parts": [{
            "id": f"{TOOL_NAME}_0",
            "response": {"result": TOOL_RESULT},
            "type": "tool_call_response",
        }],
    },
]

_TURN_2_OUTPUT_MESSAGES = [{
    "role": "assistant",
    "parts": [{"content": FINAL_TEXT, "type": "text"}],
    "finish_reason": "stop",
}]


EXPECTED_EXPERIMENTAL_SPAN_ONLY = SpanDigest(
    name="invocation",
    attributes={},
    children=[
        SpanDigest(
            name="invoke_agent some_root_agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.description": AGENT_DESCRIPTION,
                "gen_ai.agent.name": AGENT_NAME,
                "gen_ai.conversation.id": PRESENT,
            },
            children=[
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.input.messages": _TURN_1_INPUT_MESSAGES,
                                "gen_ai.system_instructions": (
                                    _SYSTEM_INSTRUCTIONS
                                ),
                                "gen_ai.tool.definitions": [
                                    _TOOL_DEFINITION_FULL
                                ],
                                "gen_ai.output.messages": (
                                    _TURN_1_OUTPUT_MESSAGES
                                ),
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.tool.definitions": [
                                            _TOOL_DEFINITION_NO_CONTENT
                                        ],
                                    },
                                ),
                            ],
                            children=[
                                SpanDigest(
                                    name="execute_tool some_tool",
                                    attributes={
                                        "gen_ai.operation.name": "execute_tool",
                                        "gen_ai.tool.description": (
                                            TOOL_DESCRIPTION
                                        ),
                                        "gen_ai.tool.name": TOOL_NAME,
                                        "gen_ai.tool.type": "FunctionTool",
                                        "gcp.vertex.agent.llm_request": "{}",
                                        "gcp.vertex.agent.llm_response": "{}",
                                        "gcp.vertex.agent.tool_call_args": "{}",
                                        "gen_ai.tool.call.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.tool_response": "{}",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.input.messages": _TURN_2_INPUT_MESSAGES,
                                "gen_ai.system_instructions": (
                                    _SYSTEM_INSTRUCTIONS
                                ),
                                "gen_ai.tool.definitions": [
                                    _TOOL_DEFINITION_FULL
                                ],
                                "gen_ai.output.messages": (
                                    _TURN_2_OUTPUT_MESSAGES
                                ),
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.tool.definitions": [
                                            _TOOL_DEFINITION_NO_CONTENT
                                        ],
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Experimental semconv,
# OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=event_only
# ---------------------------------------------------------------------------
# Span gets the no-content view (only tool definitions, with params=None).
# Log gets the full op-details (input/output messages, system instructions,
# full tool definitions).

EXPECTED_EXPERIMENTAL_EVENT_ONLY = SpanDigest(
    name="invocation",
    attributes={},
    children=[
        SpanDigest(
            name="invoke_agent some_root_agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.description": AGENT_DESCRIPTION,
                "gen_ai.agent.name": AGENT_NAME,
                "gen_ai.conversation.id": PRESENT,
            },
            children=[
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.tool.definitions": [
                                    _TOOL_DEFINITION_NO_CONTENT
                                ],
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "user.id": "test_user",
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.input.messages": (
                                            _TURN_1_INPUT_MESSAGES
                                        ),
                                        "gen_ai.system_instructions": (
                                            _SYSTEM_INSTRUCTIONS
                                        ),
                                        "gen_ai.tool.definitions": [
                                            _TOOL_DEFINITION_FULL
                                        ],
                                        "gen_ai.output.messages": (
                                            _TURN_1_OUTPUT_MESSAGES
                                        ),
                                    },
                                ),
                            ],
                            children=[
                                SpanDigest(
                                    name="execute_tool some_tool",
                                    attributes={
                                        "gen_ai.operation.name": "execute_tool",
                                        "gen_ai.tool.description": (
                                            TOOL_DESCRIPTION
                                        ),
                                        "gen_ai.tool.name": TOOL_NAME,
                                        "gen_ai.tool.type": "FunctionTool",
                                        "gcp.vertex.agent.llm_request": "{}",
                                        "gcp.vertex.agent.llm_response": "{}",
                                        "gcp.vertex.agent.tool_call_args": "{}",
                                        "gen_ai.tool.call.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.tool_response": "{}",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.tool.definitions": [
                                    _TOOL_DEFINITION_NO_CONTENT
                                ],
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "user.id": "test_user",
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.input.messages": (
                                            _TURN_2_INPUT_MESSAGES
                                        ),
                                        "gen_ai.system_instructions": (
                                            _SYSTEM_INSTRUCTIONS
                                        ),
                                        "gen_ai.tool.definitions": [
                                            _TOOL_DEFINITION_FULL
                                        ],
                                        "gen_ai.output.messages": (
                                            _TURN_2_OUTPUT_MESSAGES
                                        ),
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Experimental semconv,
# OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=span_and_event
# ---------------------------------------------------------------------------
# Both span and log get the full op-details.

EXPECTED_EXPERIMENTAL_SPAN_AND_EVENT = SpanDigest(
    name="invocation",
    attributes={},
    children=[
        SpanDigest(
            name="invoke_agent some_root_agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.description": AGENT_DESCRIPTION,
                "gen_ai.agent.name": AGENT_NAME,
                "gen_ai.conversation.id": PRESENT,
            },
            children=[
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.input.messages": _TURN_1_INPUT_MESSAGES,
                                "gen_ai.system_instructions": (
                                    _SYSTEM_INSTRUCTIONS
                                ),
                                "gen_ai.tool.definitions": [
                                    _TOOL_DEFINITION_FULL
                                ],
                                "gen_ai.output.messages": (
                                    _TURN_1_OUTPUT_MESSAGES
                                ),
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "user.id": "test_user",
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.input.messages": (
                                            _TURN_1_INPUT_MESSAGES
                                        ),
                                        "gen_ai.system_instructions": (
                                            _SYSTEM_INSTRUCTIONS
                                        ),
                                        "gen_ai.tool.definitions": [
                                            _TOOL_DEFINITION_FULL
                                        ],
                                        "gen_ai.output.messages": (
                                            _TURN_1_OUTPUT_MESSAGES
                                        ),
                                    },
                                ),
                            ],
                            children=[
                                SpanDigest(
                                    name="execute_tool some_tool",
                                    attributes={
                                        "gen_ai.operation.name": "execute_tool",
                                        "gen_ai.tool.description": (
                                            TOOL_DESCRIPTION
                                        ),
                                        "gen_ai.tool.name": TOOL_NAME,
                                        "gen_ai.tool.type": "FunctionTool",
                                        "gcp.vertex.agent.llm_request": "{}",
                                        "gcp.vertex.agent.llm_response": "{}",
                                        "gcp.vertex.agent.tool_call_args": "{}",
                                        "gen_ai.tool.call.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.tool_response": "{}",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.input.messages": _TURN_2_INPUT_MESSAGES,
                                "gen_ai.system_instructions": (
                                    _SYSTEM_INSTRUCTIONS
                                ),
                                "gen_ai.tool.definitions": [
                                    _TOOL_DEFINITION_FULL
                                ],
                                "gen_ai.output.messages": (
                                    _TURN_2_OUTPUT_MESSAGES
                                ),
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "user.id": "test_user",
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.input.messages": (
                                            _TURN_2_INPUT_MESSAGES
                                        ),
                                        "gen_ai.system_instructions": (
                                            _SYSTEM_INSTRUCTIONS
                                        ),
                                        "gen_ai.tool.definitions": [
                                            _TOOL_DEFINITION_FULL
                                        ],
                                        "gen_ai.output.messages": (
                                            _TURN_2_OUTPUT_MESSAGES
                                        ),
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# MCP-integration single-turn shape (experimental semconv only).
#
# Used by ``test_functional.py``'s MCP integration test. The scenario is
# a single-turn agent (``MockModel`` returns text immediately) whose only
# tool source is an ``McpToolset`` whose underlying session exposes one
# ``mcp_echo`` tool. ``McpToolset`` calls ``list_tools()`` once per agent
# invocation and materializes the result into a ``FunctionDeclaration``;
# the experimental semconv builder reads that declaration straight from
# ``llm_request.config.tools`` without ever talking to the MCP server
# itself.
#
# Only the experimental path needs a dedicated shape: stable semconv
# doesn't emit ``gen_ai.tool.definitions`` at all, so the MCP integration
# would be indistinguishable from any other tool-bearing agent under
# stable semconv.
#
# In ``EXPECTED_EXPERIMENTAL_SPAN_AND_EVENT_WITH_MCP``, the MCP-resolved
# ``mcp_echo`` definition surfaces in both ``gen_ai.tool.definitions``
# (span attribute) and the same key on the completion-details log
# record. The ``parameters`` block uses standard JSON Schema vocabulary
# (``object``, ``string``) because ``McpTool._get_declaration`` passes
# the MCP ``inputSchema`` through ``parameters_json_schema`` when the
# ``JSON_SCHEMA_FOR_FUNC_DECL`` feature is enabled.
# ---------------------------------------------------------------------------

_MCP_TOOL_NAME = "mcp_echo"
_MCP_TOOL_DESCRIPTION = "Echoes back its input."
_MCP_TOOL_DEFINITION_FULL = {
    "name": _MCP_TOOL_NAME,
    "description": _MCP_TOOL_DESCRIPTION,
    "parameters": {
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "type": "object",
    },
    "type": "function",
}

_MCP_TURN_INPUT_MESSAGES = [{
    "role": "user",
    "parts": [{"content": USER_PROMPT, "type": "text"}],
}]

_MCP_TURN_OUTPUT_MESSAGES = [{
    "role": "assistant",
    "parts": [{"content": FINAL_TEXT, "type": "text"}],
    # ``MockModel`` does not populate ``finish_reason``; it surfaces here as
    # the empty string from ``_to_finish_reason(None)``.
    "finish_reason": "",
}]


EXPECTED_EXPERIMENTAL_SPAN_AND_EVENT_WITH_MCP = SpanDigest(
    name="invocation",
    attributes={},
    children=[
        SpanDigest(
            name="invoke_agent some_root_agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.description": AGENT_DESCRIPTION,
                "gen_ai.agent.name": AGENT_NAME,
                "gen_ai.conversation.id": PRESENT,
            },
            children=[
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.input.messages": (
                                    _MCP_TURN_INPUT_MESSAGES
                                ),
                                "gen_ai.system_instructions": [{
                                    "content": FULL_SYSTEM_INSTRUCTION,
                                    "type": "text",
                                }],
                                "gen_ai.tool.definitions": [
                                    _MCP_TOOL_DEFINITION_FULL
                                ],
                                "gen_ai.output.messages": (
                                    _MCP_TURN_OUTPUT_MESSAGES
                                ),
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "user.id": "test_user",
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.input.messages": (
                                            _MCP_TURN_INPUT_MESSAGES
                                        ),
                                        "gen_ai.system_instructions": [{
                                            "content": FULL_SYSTEM_INSTRUCTION,
                                            "type": "text",
                                        }],
                                        "gen_ai.tool.definitions": [
                                            _MCP_TOOL_DEFINITION_FULL
                                        ],
                                        "gen_ai.output.messages": (
                                            _MCP_TURN_OUTPUT_MESSAGES
                                        ),
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Parametrization list.
# ---------------------------------------------------------------------------

ALL_CASES: list[FunctionalTestCase] = [
    FunctionalTestCase(
        test_id="stable-no-capture",
        semconv_opt_in=None,
        capture_content="false",
        expected_root=EXPECTED_STABLE_NO_CAPTURE,
    ),
    FunctionalTestCase(
        test_id="stable-capture",
        semconv_opt_in=None,
        capture_content="true",
        expected_root=EXPECTED_STABLE_CAPTURE,
    ),
    FunctionalTestCase(
        test_id="experimental-no-content",
        semconv_opt_in=EXPERIMENTAL_OPT_IN,
        capture_content="no_content",
        expected_root=EXPECTED_EXPERIMENTAL_NO_CONTENT,
    ),
    FunctionalTestCase(
        test_id="experimental-span-only",
        semconv_opt_in=EXPERIMENTAL_OPT_IN,
        capture_content="span_only",
        expected_root=EXPECTED_EXPERIMENTAL_SPAN_ONLY,
    ),
    FunctionalTestCase(
        test_id="experimental-event-only",
        semconv_opt_in=EXPERIMENTAL_OPT_IN,
        capture_content="event_only",
        expected_root=EXPECTED_EXPERIMENTAL_EVENT_ONLY,
    ),
    FunctionalTestCase(
        test_id="experimental-span-and-event",
        semconv_opt_in=EXPERIMENTAL_OPT_IN,
        capture_content="span_and_event",
        expected_root=EXPECTED_EXPERIMENTAL_SPAN_AND_EVENT,
    ),
]
