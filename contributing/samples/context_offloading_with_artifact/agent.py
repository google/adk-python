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
"""Sales Data Assistant Agent demonstrating context offloading with artifacts.

This agent simulates querying large sales reports. To avoid cluttering
the LLM context window with large amounts of data, queried reports are
saved as artifacts rather than returned directly in function responses.
Tools are used to inject artifact content into the LLM context only when
needed:
- QueryLargeDataTool injects content immediately after a report is generated.
- CustomLoadArtifactsTool injects content when load_artifacts is called, and
  also provides artifact summaries to the LLM based on artifact metadata.
"""

import json
import logging
import random

from google.adk import Agent
from google.adk.apps import App
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.load_artifacts_tool import LoadArtifactsTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from typing_extensions import override

logger = logging.getLogger('google_adk.' + __name__)


class CustomLoadArtifactsTool(LoadArtifactsTool):
  """A custom tool to load artifacts that also provides summaries.

  This tool extends LoadArtifactsTool to read custom metadata from artifacts
  and provide summaries to the LLM in the system instructions, allowing the
  model to know what artifacts are available (e.g., "Sales report for APAC").
  It also injects artifact content into the LLM request when load_artifacts
  is called by the model.
  """

  @override
  async def _append_artifacts_to_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ):
    artifact_names = await tool_context.list_artifacts()
    if not artifact_names:
      return

    summaries = {}
    for name in artifact_names:
      version_info = await tool_context.get_artifact_version(name)
      if version_info and version_info.custom_metadata:
        summaries[name] = version_info.custom_metadata.get('summary')

    artifacts_with_summaries = [
        f'{name}: {summaries.get(name)}'
        if name in summaries and summaries.get(name)
        else name
        for name in artifact_names
    ]

    # Tell the model about the available artifacts.
    llm_request.append_instructions([
        f"""You have access to artifacts: {json.dumps(artifacts_with_summaries)}.
If you need to answer a question that requires artifact content, first check if
the content was very recently added to the conversation (e.g., in the last
turn). If it is, use that content directly to answer. If the content is not
available in the recent conversation history, you MUST call `load_artifacts`
to retrieve it before answering.
"""
    ])

    # Attach the content of the artifacts if the model requests them.
    # This only adds the content to the model request, instead of the session.
    if llm_request.contents and llm_request.contents[-1].parts:
      function_response = llm_request.contents[-1].parts[0].function_response
      if function_response and function_response.name == 'load_artifacts':
        artifact_names = function_response.response['artifact_names']
        if not artifact_names:
          return
        for artifact_name in artifact_names:
          # Try session-scoped first (default behavior)
          artifact = await tool_context.load_artifact(artifact_name)

          # If not found and name doesn't already have user: prefix,
          # try cross-session artifacts with user: prefix
          if artifact is None and not artifact_name.startswith('user:'):
            prefixed_name = f'user:{artifact_name}'
            artifact = await tool_context.load_artifact(prefixed_name)

          if artifact is None:
            logger.warning('Artifact "%s" not found, skipping', artifact_name)
            continue
          llm_request.contents.append(
              types.Content(
                  role='user',
                  parts=[
                      types.Part.from_text(
                          text=f'Artifact {artifact_name} is:'
                      ),
                      artifact,
                  ],
              )
          )


async def query_large_data(query: str, tool_context: ToolContext) -> dict:
  """Generates a mock sales report for a given region and saves it as an artifact.

  This function simulates querying a large dataset. It generates a mock report
  for North America, EMEA, or APAC, saves it as a text artifact, and includes
  a data summary in the artifact's custom metadata.
  Example queries: "Get sales data for North America", "EMEA sales report".

  Args:
    query: The user query, expected to contain a region name.
    tool_context: The tool context for saving artifacts.

  Returns:
    A dictionary containing a confirmation message and the artifact name.
  """
  region = 'Unknown'
  if 'north america' in query.lower():
    region = 'North America'
  elif 'emea' in query.lower():
    region = 'EMEA'
  elif 'apac' in query.lower():
    region = 'APAC'
  else:
    return {
        'message': f"Sorry, I don't have data for query: {query}",
        'artifact_name': None,
    }

  # simulate large data - Generate a mock sales report
  report_content = f"""SALES REPORT: {region} Q3 2025
=========================================
Total Revenue: ${random.uniform(500, 2000):.2f}M
Units Sold: {random.randint(100000, 500000)}
Key Products: Gadget Pro, Widget Max, Thingy Plus
Highlights:
- Strong growth in Gadget Pro driven by new marketing campaign.
- Widget Max sales are stable.
- Thingy Plus saw a 15% increase in market share.

Regional Breakdown:
""" + ''.join([
      f'Sub-region {i+1} performance metric: {random.random()*100:.2f}\n'
      for i in range(500)
  ])
  data_summary = f'Sales report for {region} Q3 2025'
  artifact_name = f"{region.replace(' ', '_')}_sales_report_q3_2025.txt"

  await tool_context.save_artifact(
      artifact_name,
      types.Part.from_text(text=report_content),
      custom_metadata={'summary': data_summary},
  )
  return {
      'message': (
          f'Sales data for {region} for Q3 2025 is saved as artifact'
          f" '{artifact_name}'."
      ),
      'artifact_name': artifact_name,
  }


class QueryLargeDataTool(FunctionTool):
  """A tool that queries large data and saves it as an artifact.

  This tool wraps the query_large_data function. Its process_llm_request
  method checks if query_large_data was just called. If so, it loads the
  artifact that was just created and injects its content into the LLM
  request, so the model can use the data immediately in the next turn.
  """

  def __init__(self):
    super().__init__(query_large_data)

  @override
  async def process_llm_request(
      self,
      *,
      tool_context: ToolContext,
      llm_request: LlmRequest,
  ) -> None:
    await super().process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )
    if llm_request.contents and llm_request.contents[-1].parts:
      function_response = llm_request.contents[-1].parts[0].function_response
      if function_response and function_response.name == 'query_large_data':
        artifact_name = function_response.response.get('artifact_name')
        if artifact_name:
          artifact = await tool_context.load_artifact(artifact_name)
          if artifact:
            llm_request.contents.append(
                types.Content(
                    role='user',
                    parts=[
                        types.Part.from_text(
                            text=f'Artifact {artifact_name} is:'
                        ),
                        artifact,
                    ],
                )
            )


root_agent = Agent(
    model='gemini-2.5-flash',
    name='context_offloading_with_artifact',
    description='An assistant for querying large sales reports.',
    instruction="""
      You are a sales data assistant. You can query large sales reports by
      region (North America, EMEA, APAC) using the query_large_data tool.
      If you are asked to compare data between regions, make sure you have
      queried the data for all required regions first, and then use the
      load_artifacts tool if you need to access reports from previous turns.
    """,
    tools=[
        QueryLargeDataTool(),
        CustomLoadArtifactsTool(),
    ],
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(  # avoid false alarm about rolling dice.
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
)


app = App(
    name='context_offloading_with_artifact',
    root_agent=root_agent,
)
