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

"""A customer-support agent with two tools, one of them destructive.

The agent runs on a local Ollama model so the example exercises real model
behavior (see README.md). agent-hooks governance is wired in ``main.py``: the
``delete_account`` tool call is denied, and ``lookup_account`` results are
redacted, before the model ever sees them.
"""

from __future__ import annotations

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm


def lookup_account(user_id: str) -> dict:
  """Looks up a customer account.

  Args:
    user_id: The id of the account to look up.

  Returns:
    The account record, including fields the governance policy will redact.
  """
  return {
      "user_id": user_id,
      "name": "Alice Example",
      "email": "alice@example.com",
      "api_key": "EXAMPLE_NOT_A_REAL_KEY",
      "plan": "pro",
  }


def delete_account(user_id: str) -> dict:
  """Permanently deletes a customer account.

  This is a destructive tool; the governance policy denies it before it runs.

  Args:
    user_id: The id of the account to delete.

  Returns:
    A confirmation record (never reached under the governance policy).
  """
  return {"user_id": user_id, "status": "deleted"}


root_agent = LlmAgent(
    name="support_agent",
    model=LiteLlm(model="ollama_chat/qwen2.5:latest"),
    description="A customer-support agent guarded by agent-hooks.",
    instruction=(
        "You are a customer-support assistant. Always use the available tools"
        " to fulfill the user's request: call lookup_account to read an account"
        " and call delete_account when the user asks to delete one. After a"
        " tool returns, summarize its result for the user. If a tool result"
        " reports that it was blocked by policy, tell the user you were not"
        " allowed to perform that action."
    ),
    tools=[lookup_account, delete_account],
)
