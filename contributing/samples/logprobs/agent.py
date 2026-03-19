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

"""Sample agent demonstrating log probability usage.

This agent shows how to access log probabilities from language model responses.
The after_model_callback appends confidence information to demonstrate how
logprobs can be extracted and used.
"""

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_response import LlmResponse
from google.genai import types


async def append_logprobs_to_response(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse:
  """After-model callback that appends log probability information to response.

  This callback demonstrates how to access avg_logprobs and logprobs_result
  from the LlmResponse and append the information to the response content.

  Args:
    callback_context: The current callback context
    llm_response: The LlmResponse containing logprobs data

  Returns:
    Modified LlmResponse with logprobs information appended
  """
  # Build log probability analysis
  if llm_response.avg_logprobs is None:
    print("⚠️ No log probability data available")
    logprobs_info = (
        "\n\n[LOG PROBABILITY ANALYSIS]\n⚠️ No log probability data available"
    )
  else:
    print(f"📊 Average log probability: {llm_response.avg_logprobs:.4f}")

    # Build confidence analysis
    confidence_level = (
        "High"
        if llm_response.avg_logprobs >= -0.5
        else "Medium"
        if llm_response.avg_logprobs >= -1.0
        else "Low"
    )

    logprobs_info = f"""

[LOG PROBABILITY ANALYSIS]
📊 Average Log Probability: {llm_response.avg_logprobs:.4f}
🎯 Confidence Level: {confidence_level}
📈 Confidence Score: {100 * (2 ** llm_response.avg_logprobs):.1f}%"""

    # Optionally include detailed logprobs_result information
    if (
        llm_response.logprobs_result
        and llm_response.logprobs_result.top_candidates
    ):
      logprobs_info += (
          "\n🔍 Top alternatives analyzed:"
          f" {len(llm_response.logprobs_result.top_candidates)}"
      )

  # Append logprobs analysis to the response
  if llm_response.content and llm_response.content.parts:
    llm_response.content.parts.append(types.Part(text=logprobs_info))

  return llm_response


# Create a simple agent that demonstrates logprobs usage
root_agent = Agent(
    model="gemini-2.0-flash",
    name="logprobs_demo_agent",
    description=(
        "A simple agent that demonstrates log probability extraction and"
        " display."
    ),
    instruction="""
    You are a helpful AI assistant. Answer user questions normally and naturally.

    After you respond, you'll see log probability analysis appended to your response.
    You don't need to include the log probability analysis in your response yourself.
    """,
    generate_content_config=types.GenerateContentConfig(
        response_logprobs=True,  # Enable log probability collection
        logprobs=5,  # Collect top 5 alternatives for analysis
        temperature=0.7,  # Moderate temperature for varied responses
    ),
    after_model_callback=append_logprobs_to_response,
)
