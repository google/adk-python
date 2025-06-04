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

from google.adk.tools.application_integration_tool.application_integration_toolset import ApplicationIntegrationToolset

jira_tool = ApplicationIntegrationToolset(
    project="your-gcp-project-id",  # replace with your GCP project ID
    location="your-regions",  # replace your regions
    connection="your-integration-connection-name",  # replace with your connection name
    entity_operations={
        "Issues": ["GET", "LIST"],
    },
    actions=[
        "get_issue_by_key",
    ],
    tool_name="jira_conversation_tool",
    tool_instructions="""
    
    This tool is to call an integration to search for issues in JIRA
    
    """,
)
