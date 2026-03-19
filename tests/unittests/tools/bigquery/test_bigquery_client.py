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

from __future__ import annotations

import os
from unittest import mock

import google.adk
from google.adk.tools.bigquery.client import DP_USER_AGENT
from google.adk.tools.bigquery.client import get_bigquery_client
from google.adk.tools.bigquery.client import get_dataplex_catalog_client
from google.api_core.gapic_v1 import client_info as gapic_client_info
import google.auth
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import dataplex_v1
from google.cloud.bigquery import client as bigquery_client
from google.oauth2.credentials import Credentials


def test_bigquery_client_default():
  """Test the default BigQuery client properties."""
  # Trigger the BigQuery client creation
  client = get_bigquery_client(
      project="test-gcp-project",
      credentials=mock.create_autospec(Credentials, instance=True),
  )

  # Verify that the client has the desired project set
  assert client.project == "test-gcp-project"
  assert client.location is None


def test_bigquery_client_project_set_explicit():
  """Test BigQuery client creation does not invoke default auth."""
  # Let's simulate that no environment variables are set, so that any project
  # set in there does not interfere with this test
  with mock.patch.dict(os.environ, {}, clear=True):
    with mock.patch.object(
        google.auth, "default", autospec=True
    ) as mock_default_auth:
      # Simulate exception from default auth
      mock_default_auth.side_effect = DefaultCredentialsError(
          "Your default credentials were not found"
      )

      # Trigger the BigQuery client creation
      client = get_bigquery_client(
          project="test-gcp-project",
          credentials=mock.create_autospec(Credentials, instance=True),
      )

      # If we are here that already means client creation did not call default
      # auth (otherwise we would have run into DefaultCredentialsError set
      # above). For the sake of explicitness, trivially assert that the default
      # auth was not called, and yet the project was set correctly
      mock_default_auth.assert_not_called()
      assert client.project == "test-gcp-project"


def test_bigquery_client_project_set_with_default_auth():
  """Test BigQuery client creation invokes default auth to set the project."""
  # Let's simulate that no environment variables are set, so that any project
  # set in there does not interfere with this test
  with mock.patch.dict(os.environ, {}, clear=True):
    with mock.patch.object(
        google.auth, "default", autospec=True
    ) as mock_default_auth:
      # Simulate credentials
      mock_creds = mock.create_autospec(Credentials, instance=True)

      # Simulate output of the default auth
      mock_default_auth.return_value = (mock_creds, "test-gcp-project")

      # Trigger the BigQuery client creation
      client = get_bigquery_client(
          project=None,
          credentials=mock_creds,
      )

      # Verify that default auth was called once to set the client project
      mock_default_auth.assert_called_once()
      assert client.project == "test-gcp-project"


def test_bigquery_client_project_set_with_env():
  """Test BigQuery client creation sets the project from environment variable."""
  # Let's simulate the project set in environment variables
  with mock.patch.dict(
      os.environ, {"GOOGLE_CLOUD_PROJECT": "test-gcp-project"}, clear=True
  ):
    with mock.patch.object(
        google.auth, "default", autospec=True
    ) as mock_default_auth:
      # Simulate exception from default auth
      mock_default_auth.side_effect = DefaultCredentialsError(
          "Your default credentials were not found"
      )

      # Trigger the BigQuery client creation
      client = get_bigquery_client(
          project=None,
          credentials=mock.create_autospec(Credentials, instance=True),
      )

      # If we are here that already means client creation did not call default
      # auth (otherwise we would have run into DefaultCredentialsError set
      # above). For the sake of explicitness, trivially assert that the default
      # auth was not called, and yet the project was set correctly
      mock_default_auth.assert_not_called()
      assert client.project == "test-gcp-project"


def test_bigquery_client_user_agent_default():
  """Test BigQuery client default user agent."""
  with mock.patch.object(
      bigquery_client, "Connection", autospec=True
  ) as mock_connection:
    # Trigger the BigQuery client creation
    get_bigquery_client(
        project="test-gcp-project",
        credentials=mock.create_autospec(Credentials, instance=True),
    )

    # Verify that the tracking user agent was set
    client_info_arg = mock_connection.call_args[1].get("client_info")
    assert client_info_arg is not None
    expected_user_agents = {
        "adk-bigquery-tool",
        f"google-adk/{google.adk.__version__}",
    }
    actual_user_agents = set(client_info_arg.user_agent.split())
    assert expected_user_agents.issubset(actual_user_agents)


def test_bigquery_client_user_agent_custom():
  """Test BigQuery client custom user agent."""
  with mock.patch.object(
      bigquery_client, "Connection", autospec=True
  ) as mock_connection:
    # Trigger the BigQuery client creation
    get_bigquery_client(
        project="test-gcp-project",
        credentials=mock.create_autospec(Credentials, instance=True),
        user_agent="custom_user_agent",
    )

    # Verify that the tracking user agent was set
    client_info_arg = mock_connection.call_args[1].get("client_info")
    assert client_info_arg is not None
    expected_user_agents = {
        "adk-bigquery-tool",
        f"google-adk/{google.adk.__version__}",
        "custom_user_agent",
    }
    actual_user_agents = set(client_info_arg.user_agent.split())
    assert expected_user_agents.issubset(actual_user_agents)


def test_bigquery_client_user_agent_custom_list():
  """Test BigQuery client custom user agent."""
  with mock.patch.object(
      bigquery_client, "Connection", autospec=True
  ) as mock_connection:
    # Trigger the BigQuery client creation
    get_bigquery_client(
        project="test-gcp-project",
        credentials=mock.create_autospec(Credentials, instance=True),
        user_agent=["custom_user_agent1", "custom_user_agent2"],
    )

    # Verify that the tracking user agents were set
    client_info_arg = mock_connection.call_args[1].get("client_info")
    assert client_info_arg is not None
    expected_user_agents = {
        "adk-bigquery-tool",
        f"google-adk/{google.adk.__version__}",
        "custom_user_agent1",
        "custom_user_agent2",
    }
    actual_user_agents = set(client_info_arg.user_agent.split())
    assert expected_user_agents.issubset(actual_user_agents)


def test_bigquery_client_location_custom():
  """Test BigQuery client custom location."""
  # Trigger the BigQuery client creation
  client = get_bigquery_client(
      project="test-gcp-project",
      credentials=mock.create_autospec(Credentials, instance=True),
      location="us-central1",
  )

  # Verify that the client has the desired project set
  assert client.project == "test-gcp-project"
  assert client.location == "us-central1"


# Tests for Dataplex Catalog Client
# ------------------------------------------------------------------------------


# Mock the CatalogServiceClient class directly
@mock.patch.object(dataplex_v1, "CatalogServiceClient", autospec=True)
def test_dataplex_client_default(mock_catalog_service_client):
  """Test get_dataplex_catalog_client with default user agent."""
  mock_creds = mock.create_autospec(Credentials, instance=True)

  client = get_dataplex_catalog_client(credentials=mock_creds)

  mock_catalog_service_client.assert_called_once()
  _, kwargs = mock_catalog_service_client.call_args

  assert kwargs["credentials"] == mock_creds
  client_info = kwargs["client_info"]
  assert isinstance(client_info, gapic_client_info.ClientInfo)
  assert client_info.user_agent == DP_USER_AGENT

  # Ensure the function returns the mock instance
  assert client == mock_catalog_service_client.return_value


@mock.patch.object(dataplex_v1, "CatalogServiceClient", autospec=True)
def test_dataplex_client_custom_user_agent_str(mock_catalog_service_client):
  """Test get_dataplex_catalog_client with a custom user agent string."""
  mock_creds = mock.create_autospec(Credentials, instance=True)
  custom_ua = "catalog_ua/1.0"
  expected_ua = f"{DP_USER_AGENT} {custom_ua}"

  get_dataplex_catalog_client(credentials=mock_creds, user_agent=custom_ua)

  mock_catalog_service_client.assert_called_once()
  _, kwargs = mock_catalog_service_client.call_args
  client_info = kwargs["client_info"]
  assert client_info.user_agent == expected_ua


@mock.patch.object(dataplex_v1, "CatalogServiceClient", autospec=True)
def test_dataplex_client_custom_user_agent_list(mock_catalog_service_client):
  """Test get_dataplex_catalog_client with a custom user agent list."""
  mock_creds = mock.create_autospec(Credentials, instance=True)
  custom_ua_list = ["catalog_ua", "catalog_ua_2.0"]
  expected_ua = f"{DP_USER_AGENT} {' '.join(custom_ua_list)}"

  get_dataplex_catalog_client(credentials=mock_creds, user_agent=custom_ua_list)

  mock_catalog_service_client.assert_called_once()
  _, kwargs = mock_catalog_service_client.call_args
  client_info = kwargs["client_info"]
  assert client_info.user_agent == expected_ua


@mock.patch.object(dataplex_v1, "CatalogServiceClient", autospec=True)
def test_dataplex_client_custom_user_agent_list_with_none(
    mock_catalog_service_client,
):
  """Test get_dataplex_catalog_client with a list containing None."""
  mock_creds = mock.create_autospec(Credentials, instance=True)
  custom_ua_list = ["catalog_ua", None, "catalog_ua_2.0"]
  expected_ua = f"{DP_USER_AGENT} catalog_ua catalog_ua_2.0"

  get_dataplex_catalog_client(credentials=mock_creds, user_agent=custom_ua_list)

  mock_catalog_service_client.assert_called_once()
  _, kwargs = mock_catalog_service_client.call_args
  client_info = kwargs["client_info"]
  assert client_info.user_agent == expected_ua
