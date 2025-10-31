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

import os
from typing import Any
from typing import Dict
from typing import Optional

from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.google_tool import GoogleTool
from google.adk.tools.spanner import query_tool
from google.adk.tools.spanner import search_tool
from google.adk.tools.spanner.settings import Capabilities
from google.adk.tools.spanner.settings import SpannerToolSettings
from google.adk.tools.spanner.spanner_credentials import SpannerCredentialsConfig
from google.adk.tools.tool_context import ToolContext
import google.auth
from google.auth.credentials import Credentials
from pydantic import BaseModel

# Define an appropriate credential type
# Set to None to use the application default credentials (ADC) for a quick
# development.
CREDENTIALS_TYPE = None


# Define Spanner tool config with read capability set to allowed.
tool_settings = SpannerToolSettings(capabilities=[Capabilities.DATA_READ])

if CREDENTIALS_TYPE == AuthCredentialTypes.OAUTH2:
  # Initiaze the tools to do interactive OAuth
  # The environment variables OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET
  # must be set
  credentials_config = SpannerCredentialsConfig(
      client_id=os.getenv("OAUTH_CLIENT_ID"),
      client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
      scopes=[
          "https://www.googleapis.com/auth/spanner.admin",
          "https://www.googleapis.com/auth/spanner.data",
      ],
  )
elif CREDENTIALS_TYPE == AuthCredentialTypes.SERVICE_ACCOUNT:
  # Initialize the tools to use the credentials in the service account key.
  # If this flow is enabled, make sure to replace the file path with your own
  # service account key file
  # https://cloud.google.com/iam/docs/service-account-creds#user-managed-keys
  creds, _ = google.auth.load_credentials_from_file("service_account_key.json")
  credentials_config = SpannerCredentialsConfig(credentials=creds)
else:
  # Initialize the tools to use the application default credentials.
  # https://cloud.google.com/docs/authentication/provide-credentials-adc
  application_default_credentials, _ = google.auth.default()
  credentials_config = SpannerCredentialsConfig(
      credentials=application_default_credentials
  )


### Section 1: Extending the built-in Spanner Toolset for Custom Use Cases ###
# This example illustrates how to extend the built-in Spanner toolset to create
# a customized Spanner tool. This method is advantageous when you need to deal
# with a specific use case:
#
# 1.  Streamline the end user experience by pre-configuring the tool with fixed
#     parameters (such as a specific database, instance, or project) and a
#     dedicated SQL query, making it perfect for a single, focused use case
#     like vector search on a specific table.
# 2.  Enhance functionality by adding custom logic to manage tool inputs,
#     execution, and result processing, providing greater control over the
#     tool's behavior.
class SpannerRagSetting(BaseModel):
  """Customized Spanner RAG settings for an example use case."""

  # Replace the following settings for your Spanner database used in the sample.
  project_id: str = "<PROJECT_ID>"
  instance_id: str = "<INSTANCE_ID>"
  database_id: str = "<DATABASE_ID>"

  # Follow the instructions in README.md, the table name is "products" and the
  # Spanner embedding model name is "EmbeddingsModel" in this sample.
  table_name: str = "products"
  # Learn more about Spanner Vertex AI integration for embedding and Spanner
  # vector search.
  # https://cloud.google.com/spanner/docs/ml-tutorial-embeddings
  # https://cloud.google.com/spanner/docs/vector-search/overview
  embedding_model_name: str = "EmbeddingsModel"

  selected_columns: list[str] = [
      "productId",
      "productName",
      "productDescription",
  ]
  embedding_column_name: str = "productDescriptionEmbedding"

  additional_filter_expression: str = "inventoryCount > 0"
  vector_distance_function: str = "EUCLIDEAN_DISTANCE"
  top_k: int = 3


RAG_SETTINGS = SpannerRagSetting()


### (Option 1) Use the built-in similarity_search tool ###
# Create a wrapped function tool for the agent on top of the built-in
# similarity_search tool in the Spanner toolset.
# This customized tool is used to perform a Spanner KNN vector search on a
# embedded knowledge base stored in a Spanner database table.
def wrapped_spanner_similarity_search(
    search_query: str,
    credentials: Credentials,  # GoogleTool handles `credentials` automatically
    settings: SpannerToolSettings,  # GoogleTool handles `settings` automatically
    tool_context: ToolContext,  # GoogleTool handles `tool_context` automatically
) -> str:
  """Perform a similarity search on the product catalog.

  Args:
    search_query: The search query to find relevant content.

  Returns:
      Relevant product catalog content with sources
  """
  columns = RAG_SETTINGS.selected_columns.copy()

  # Instead of fixing all parameters, you can also expose some of them for
  # the LLM to decide.
  return search_tool.similarity_search(
      RAG_SETTINGS.project_id,
      RAG_SETTINGS.instance_id,
      RAG_SETTINGS.database_id,
      RAG_SETTINGS.table_name,
      search_query,
      RAG_SETTINGS.embedding_column_name,
      columns,
      {
          "spanner_embedding_model_name": RAG_SETTINGS.embedding_model_name,
      },
      credentials,
      settings,
      tool_context,
      RAG_SETTINGS.additional_filter_expression,
      {
          "top_k": RAG_SETTINGS.top_k,
          "distance_type": RAG_SETTINGS.vector_distance_function,
      },
  )


### (Option 2) Use the built-in execute_sql tool ###
# Create a wrapped function tool for the agent on top of the built-in
# execute_sql tool in the Spanner toolset.
# This customized tool is used to perform a Spanner KNN vector search on a
# embedded knowledge base stored in a Spanner database table.
#
# Compared with similarity_search, using execute_sql (a lower level tool) means
# that you have more control, but you also need to do more work (e.g. to write
# the SQL query from scratch). Consider using this option if your scenario is
# more complicated than a plain similarity search.
def wrapped_spanner_execute_sql_tool(
    search_query: str,
    credentials: Credentials,  # GoogleTool handles `credentials` automatically
    settings: SpannerToolSettings,  # GoogleTool handles `settings` automatically
    tool_context: ToolContext,  # GoogleTool handles `tool_context` automatically
) -> str:
  """Perform a similarity search on the product catalog.

  Args:
    search_query: The search query to find relevant content.

  Returns:
      Relevant product catalog content with sources
  """

  embedding_query = f"""SELECT embeddings.values
      FROM ML.PREDICT(
        MODEL {RAG_SETTINGS.embedding_model_name},
        (SELECT "{search_query}" as content)
      )
    """

  distance_alias = "distance"
  columns = [f"{column}" for column in RAG_SETTINGS.selected_columns]
  columns += [f"""{RAG_SETTINGS.vector_distance_function}(
        {RAG_SETTINGS.embedding_column_name},
        ({embedding_query})) AS {distance_alias}
    """]
  columns = ", ".join(columns)

  knn_query = f"""
      SELECT {columns}
      FROM {RAG_SETTINGS.table_name}
      WHERE {RAG_SETTINGS.additional_filter_expression}
      ORDER BY {distance_alias}
      LIMIT {RAG_SETTINGS.top_k}
    """

  # Customized tool based on the built-in Spanner toolset.
  return query_tool.execute_sql(
      project_id=RAG_SETTINGS.project_id,
      instance_id=RAG_SETTINGS.instance_id,
      database_id=RAG_SETTINGS.database_id,
      query=knn_query,
      credentials=credentials,
      settings=settings,
      tool_context=tool_context,
  )


def inspect_tool_params(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
) -> Optional[Dict]:
  """A callback function to inspect tool parameters before execution."""
  print("Inspect for tool: " + tool.name)

  actual_search_query_in_args = args.get("search_query")
  # Inspect the `search_query` when calling the tool for tutorial purposes.
  print(f"Tool args `search_query`: {actual_search_query_in_args}")

  pass


### Section 2: Create the root agent ###
root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="spanner_knowledge_base_agent",
    description=(
        "Agent to answer questions about product-specific recommendations."
    ),
    instruction="""
    You are a helpful assistant that answers user questions about product-specific recommendations.
    1. Always use the `wrapped_spanner_similarity_search` tool to find relevant information.
    2. If no relevant information is found, say you don't know.
    3. Present all the relevant information naturally and well formatted in your response.
    """,
    tools=[
        # # (Option 1)
        # # Add customized Spanner tool based on the built-in similarity_search
        # # in the Spanner toolset.
        GoogleTool(
            func=wrapped_spanner_similarity_search,
            credentials_config=credentials_config,
            tool_settings=tool_settings,
        ),
        # # (Option 2)
        # # Add customized Spanner tool based on the built-in execute_sql in
        # # the Spanner toolset.
        # GoogleTool(
        #     func=wrapped_spanner_execute_sql_tool,
        #     credentials_config=credentials_config,
        #     tool_settings=tool_settings,
        # ),
    ],
    before_tool_callback=inspect_tool_params,
)
