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

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent


def _inject_metadata_into_state(callback_context: CallbackContext) -> None:
    """Expand incoming A2A metadata into the session state.

    ADK's request_converter places A2A request metadata under
    `run_config.custom_metadata['a2a_metadata']`. We copy those entries into
    session state so that the agent's `instruction` template can resolve
    placeholders like `{user_name}` from values the caller provided.
    """
    run_config = callback_context.run_config
    if run_config is None or not run_config.custom_metadata:
        return
    a2a_metadata = run_config.custom_metadata.get("a2a_metadata") or {}
    for key, value in a2a_metadata.items():
        callback_context.state[key] = value


root_agent = Agent(
    model="gemini-2.5-flash",
    name="greet_agent",
    description="Greets the user using a name provided in session state.",
    instruction=(
        "Greet the user exactly in the following format, without any extra"
        " text:\n"
        "Hello {user_name}! How are you doing today?"
    ),
    before_agent_callback=_inject_metadata_into_state,
)
