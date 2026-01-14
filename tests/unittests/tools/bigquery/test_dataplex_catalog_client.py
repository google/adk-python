from __future__ import annotations

from typing import List
from typing import Optional
from typing import Union
from unittest import mock

import google.adk
from google.adk.tools.bigquery.client import DP_USER_AGENT
from google.adk.tools.bigquery.client import get_dataplex_catalog_client
from google.api_core import client_options as client_options_lib
from google.api_core.gapic_v1 import client_info as gapic_client_info
from google.cloud import dataplex_v1
from google.oauth2.credentials import Credentials


# Mock the CatalogServiceClient class directly
@mock.patch.object(dataplex_v1, "CatalogServiceClient", autospec=True)
def test_dataplex_client_default(mock_catalog_service_client):
  """Test get_dataplex_catalog_client with default user agent."""
  mock_creds = mock.create_autospec(Credentials, instance=True)

  # Call the function under test
  client = get_dataplex_catalog_client(credentials=mock_creds)

  # Assert that CatalogServiceClient constructor was called once
  mock_catalog_service_client.assert_called_once()
  _, kwargs = mock_catalog_service_client.call_args

  # Check the arguments passed to the CatalogServiceClient constructor
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
