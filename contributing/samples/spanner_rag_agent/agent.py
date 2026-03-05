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

import os

from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.tools.spanner.settings import Capabilities
from google.adk.tools.spanner.settings import SpannerToolSettings
from google.adk.tools.spanner.settings import SpannerVectorStoreSettings
from google.adk.tools.spanner.spanner_credentials import SpannerCredentialsConfig
from google.adk.tools.spanner.spanner_toolset import SpannerToolset
import google.auth

# Define an appropriate credential type
# Set to None to use the application default credentials (ADC) for a quick
# development.
CREDENTIALS_TYPE = None


if CREDENTIALS_TYPE == AuthCredentialTypes.OAUTH2:
  # Initialize the tools to do interactive OAuth
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

# Follow the instructions in README.md to set up the example Spanner database.
# Replace the following settings with your specific Spanner database.

# Define Spanner vector store settings.
vector_store_settings = SpannerVectorStoreSettings(
    project_id="<PROJECT_ID>",
    instance_id="<INSTANCE_ID>",
    database_id="<DATABASE_ID>",
    table_name="products",
    content_column="productDescription",
    embedding_column="productDescriptionEmbedding",
    vector_length=768,
    vertex_ai_embedding_model_name="text-embedding-005",
    selected_columns=[
        "productId",
        "productName",
        "productDescription",
    ],
    nearest_neighbors_algorithm="EXACT_NEAREST_NEIGHBORS",
    top_k=3,
    distance_type="COSINE",
    additional_filter="inventoryCount > 0",
)

# Define Spanner tool config with the vector store settings.
tool_settings = SpannerToolSettings(
    capabilities=[Capabilities.DATA_READ],
    vector_store_settings=vector_store_settings,
)

# Get the Spanner toolset with the Spanner tool settings and credentials config.
# Filter the tools to only include the `vector_store_similarity_search` tool.
spanner_toolset = SpannerToolset(
    credentials_config=credentials_config,
    spanner_tool_settings=tool_settings,
    # Comment to include all allowed tools.
    tool_filter=["vector_store_similarity_search"],
)


root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="spanner_knowledge_base_agent",
    description=(
        "Agent to answer questions about product-specific recommendations."
    ),
    instruction="""
    You are a helpful assistant that answers user questions about product-specific recommendations.
    1. Always use the `vector_store_similarity_search` tool to find information.
    2. Directly present all the information results from the `vector_store_similarity_search` tool naturally and well formatted in your response.
    3. If no information result is returned by the `vector_store_similarity_search` tool, say you don't know.
    """,
    # Use the Spanner toolset for vector similarity search.
    tools=[spanner_toolset],
)
