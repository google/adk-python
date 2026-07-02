#!/usr/bin/env python3
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

"""Runs compliance checks on ADK source files.

This script is used as a pre-commit hook and in CI to enforce coding standards.
"""

import argparse
import os
import re
import sys

# Legacy files that are temporarily excluded from the mTLS check.
# Do not add new files to this list. All new code must support mTLS.
_EXCLUDED_FROM_MTLS = {
    'contributing/samples/environment_and_skills/e2b_environment/agent.py',
    'contributing/samples/integrations/bigquery_mcp/agent.py',
    'contributing/samples/integrations/bigtable/agent.py',
    'contributing/samples/integrations/data_agent/agent.py',
    'contributing/samples/integrations/gcp_auth/agent.py',
    'contributing/samples/integrations/gcs/agent.py',
    'contributing/samples/integrations/gcs_admin/agent.py',
    'contributing/samples/integrations/integration_connector_euc_agent/agent.py',
    'contributing/samples/integrations/oauth_calendar_agent/agent.py',
    'contributing/samples/integrations/spanner/agent.py',
    'contributing/samples/integrations/spanner_admin/agent.py',
    'contributing/samples/integrations/spanner_rag_agent/agent.py',
    'contributing/samples/mcp/mcp_service_account_agent/agent.py',
    'contributing/samples/models/interactions_api/main.py',
    'contributing/samples/multimodal/static_non_text_content/agent.py',
    'src/google/adk/auth/auth_credential.py',
    'src/google/adk/integrations/api_registry/api_registry.py',
    'src/google/adk/integrations/bigquery/bigquery_credentials.py',
    'src/google/adk/integrations/bigquery/data_insights_tool.py',
    'src/google/adk/integrations/bigquery/metadata_tool.py',
    'src/google/adk/integrations/gcs/gcs_credentials.py',
    'src/google/adk/plugins/bigquery_agent_analytics_plugin.py',
    'src/google/adk/tools/_google_credentials.py',
    'src/google/adk/tools/apihub_tool/clients/apihub_client.py',
    'src/google/adk/tools/application_integration_tool/application_integration_toolset.py',
    'src/google/adk/tools/application_integration_tool/clients/connections_client.py',
    'src/google/adk/tools/application_integration_tool/clients/integration_client.py',
    'src/google/adk/tools/bigtable/bigtable_credentials.py',
    'src/google/adk/tools/data_agent/credentials.py',
    'src/google/adk/tools/data_agent/data_agent_tool.py',
    'src/google/adk/tools/google_api_tool/google_api_toolset.py',
    'src/google/adk/tools/google_api_tool/googleapi_to_openapi_converter.py',
    'src/google/adk/tools/mcp_tool/mcp_session_manager.py',
    'src/google/adk/tools/openapi_tool/auth/auth_helpers.py',
    'src/google/adk/tools/openapi_tool/auth/credential_exchangers/service_account_exchanger.py',
    'src/google/adk/tools/pubsub/pubsub_credentials.py',
    'src/google/adk/tools/spanner/spanner_credentials.py',
    'tests/unittests/auth/test_credential_manager.py',
    'tests/unittests/cli/utils/test_gcp_utils.py',
    'tests/unittests/flows/llm_flows/test_functions_request_euc.py',
    'tests/unittests/integrations/api_registry/test_api_registry.py',
    'tests/unittests/integrations/bigquery/test_bigquery_credentials.py',
    'tests/unittests/tools/apihub_tool/clients/test_apihub_client.py',
    'tests/unittests/tools/application_integration_tool/clients/test_connections_client.py',
    'tests/unittests/tools/application_integration_tool/clients/test_integration_client.py',
    'tests/unittests/tools/application_integration_tool/test_application_integration_toolset.py',
    'tests/unittests/tools/data_agent/test_data_agent_tool.py',
    'tests/unittests/tools/google_api_tool/test_docs_batchupdate.py',
    'tests/unittests/tools/google_api_tool/test_google_api_toolset.py',
    'tests/unittests/tools/google_api_tool/test_googleapi_to_openapi_converter.py',
    'tests/unittests/tools/openapi_tool/auth/credential_exchangers/test_service_account_exchanger.py',
    'tests/unittests/tools/openapi_tool/openapi_spec_parser/test_openapi_toolset.py',
    'tests/unittests/tools/openapi_tool/openapi_spec_parser/test_rest_api_tool.py',
    'tests/unittests/tools/spanner/test_spanner_credentials.py',
    'tests/unittests/tools/test_base_google_credentials_manager.py',
    'tests/unittests/tools/test_google_tool.py',
    'tests/unittests/workflow/utils/test_workflow_hitl_utils.py',
}


def check_logger(content: str) -> bool:
  # Forbidden: getLogger(__name__) without the 'google_adk.' prefix.
  pattern = re.compile(r'logger\s*=\s*logging\.getLogger\(__name__\)')
  return not pattern.search(content)


def check_future_annotations(content: str, filename: str) -> bool:
  # Exclude: __init__.py, version.py, tests/, contributing/samples/
  if (
      filename.endswith('__init__.py')
      or filename.endswith('version.py')
      or 'tests/' in filename
      or 'contributing/samples/' in filename
  ):
    return True
  return 'from __future__ import annotations' in content


def check_cli_import(content: str, filename: str) -> bool:
  # Exclude: cli/, apihub_toolset.py, tests/, contributing/samples/
  if (
      'cli/' in filename
      or filename.endswith('apihub_toolset.py')
      or 'tests/' in filename
      or 'contributing/samples/' in filename
  ):
    return True
  # Pattern: ^from.*\bcli\b.*import.*$ (multiline)
  pattern = re.compile(r'^from.*\bcli\b.*import.*$', re.MULTILINE)
  return not pattern.search(content)


def check_mtls(content: str, filename: str) -> bool:
  if filename in _EXCLUDED_FROM_MTLS:
    return True
  # Pattern for googleapis: https?://[a-zA-Z0-9.-]+\.googleapis\.com
  endpoint_pattern = re.compile(r'https?://[a-zA-Z0-9.-]+\.googleapis\.com')
  if endpoint_pattern.search(content):
    return '.mtls.googleapis.com' in content
  return True


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('files', nargs='*', help='Files to check')
  args = parser.parse_args()

  failed = False
  for f in args.files:
    # Skip directories if they are passed accidentally
    if not os.path.isfile(f):
      continue
    try:
      with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    except Exception as e:  # pylint: disable=broad-except
      print(f'Error reading {f}: {e}')
      continue

    # Run checks
    if not check_logger(content):
      print(
          f"❌ {f}: Found forbidden use of 'logger ="
          " logging.getLogger(__name__)'. Please use 'logger ="
          ' logging.getLogger("google_adk." + __name__)\' instead.'
      )
      failed = True

    if not check_future_annotations(content, f):
      print(f"❌ {f}: Missing 'from __future__ import annotations'.")
      failed = True

    if not check_cli_import(content, f):
      print(
          f'❌ {f}: Do not import from the cli package outside of the cli'
          ' package.'
      )
      failed = True

    if not check_mtls(content, f):
      print(
          f'❌ {f}: Found hardcoded googleapis.com endpoints without mTLS'
          ' support.'
      )
      failed = True

  if failed:
    sys.exit(1)
  sys.exit(0)


if __name__ == '__main__':
  main()
