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


from google.adk.agents.llm_agent import Agent
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.genai import types


def reimburse(purpose: str, amount: float) -> str:
  """Reimburse the amount of money to the employee."""
  return {
      'status': 'ok',
  }


approval_agent = RemoteA2aAgent(
    name='approval_agent',
    description='Help approve the reimburse if the amount is greater than 100.',
    agent_card=(
        f'http://localhost:8001/a2a/human_in_loop{AGENT_CARD_WELL_KNOWN_PATH}'
    ),
)


root_agent = Agent(
    model='gemini-2.0-flash',
    name='reimbursement_agent',
    instruction="""
      You are an agent whose job is to handle the reimbursement process for
      the employees. If the amount is less than $100, you will automatically
      approve the reimbursement. And call reimburse() to reimburse the amount to the employee.

      If the amount is greater than $100. You will hand over the request to
      approval_agent to handle the reimburse.
""",
    tools=[reimburse],
    sub_agents=[approval_agent],
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
)
