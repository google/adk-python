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

import json
from typing import Any
from typing import Dict
from typing import List

from adk_answering_agent.settings import ADK_GCP_SA_KEY
from adk_answering_agent.settings import GEMINI_API_DATASTORE_ID
from adk_answering_agent.utils import error_response
from google.adk.agents.llm_agent import Agent
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import discoveryengine_v1beta as discoveryengine
from google.oauth2 import service_account


def search_gemini_api_docs(queries: List[str]) -> Dict[str, Any]:
  """Searches Gemini API docs using Vertex AI Search.

  Args:
    queries: The list of queries to search.

  Returns:
    A dictionary containing the status of the request and the list of search
    results, which contains the title, url and snippets.
  """
  try:
    adk_gcp_sa_key_info = json.loads(ADK_GCP_SA_KEY)
    client = discoveryengine.SearchServiceClient(
        credentials=service_account.Credentials.from_service_account_info(
            adk_gcp_sa_key_info
        )
    )
  except (TypeError, ValueError) as e:
    return error_response(f"Error creating Vertex AI Search client: {e}")

  serving_config = f"{GEMINI_API_DATASTORE_ID}/servingConfigs/default_config"
  results = []
  try:
    for query in queries:
      request = discoveryengine.SearchRequest(
          serving_config=serving_config,
          query=query,
          page_size=20,
      )
      response = client.search(request=request)
      for item in response.results:
        snippets = []
        for snippet in item.document.derived_struct_data.get("snippets", []):
          snippets.append(snippet.get("snippet"))

        results.append({
            "title": item.document.derived_struct_data.get("title"),
            "url": item.document.derived_struct_data.get("link"),
            "snippets": snippets,
        })
  except GoogleAPICallError as e:
    return error_response(f"Error from Vertex AI Search: {e}")
  return {"status": "success", "results": results}


root_agent = Agent(
    model="gemini-2.5-pro",
    name="gemini_assistant",
    description="Answer questions about Gemini API.",
    instruction="""
    You are a helpful assistant that responds to questions about Gemini API based on information
    found in the document store. You can access the document store using the `search_gemini_api_docs` tool.

    When user asks a question, here are the steps:
    1. Use the `search_gemini_api_docs` tool to find relevant information before answering.
      * You can call the tool with multiple queries to find all the relevant information.
    2. Provide a response based on the information you found in the document store. Reference the source document in the response.

    IMPORTANT:
      * Your response should be based on the information you found in the document store. Do not invent
        information that is not in the document store. Do not invent citations which are not in the document store.
      * If you can't find the answer or information in the document store, just respond with "I can't find the answer or information in the document store".
      * If you uses citation from the document store, please always provide a footnote referencing the source document format it as: "[1] URL of the document".
    """,
    tools=[search_gemini_api_docs],
)
