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

from typing import Any

from a2a.types import Message as A2AMessage
from google.adk.a2a.agent.config import A2aRemoteAgentConfig
from google.adk.a2a.agent.config import ParametersConfig
from google.adk.a2a.agent.config import RequestInterceptor
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

# Only these session state keys are forwarded to the remote agent as A2A
# request metadata. Keeping the list explicit prevents accidentally leaking
# unrelated state (credentials, internal flags, large blobs, etc.) across the
# service boundary.
ALLOWED_FORWARD_KEYS: frozenset[str] = frozenset({"user_name"})


async def _forward_state_as_a2a_metadata(
    ctx: InvocationContext,
    a2a_request: A2AMessage,
    parameters: ParametersConfig,
) -> tuple[A2AMessage, ParametersConfig]:
    """Forward whitelisted session state keys through A2A request metadata."""
    payload: dict[str, Any] = {
        key: value
        for key, value in ctx.session.state.items()
        if key in ALLOWED_FORWARD_KEYS
    }
    if payload:
        parameters.request_metadata = {
            **(parameters.request_metadata or {}),
            **payload,
        }
    return a2a_request, parameters


greet_agent = RemoteA2aAgent(
    name="greet_agent",
    description="Greets the user using a name taken from session state.",
    agent_card=(
        f"http://localhost:8001/a2a/greet_agent{AGENT_CARD_WELL_KNOWN_PATH}"
    ),
    config=A2aRemoteAgentConfig(
        request_interceptors=[
            RequestInterceptor(before_request=_forward_state_as_a2a_metadata),
        ]
    ),
)


def _seed_state(callback_context: CallbackContext) -> None:
    """Seed demo session state so the remote agent has something to greet."""
    callback_context.state.setdefault("user_name", "Alice")


root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    instruction=(
        "You are a helpful assistant. When the user asks to be greeted,"
        " delegate to the greet_agent sub-agent."
    ),
    sub_agents=[greet_agent],
    before_agent_callback=_seed_state,
)
