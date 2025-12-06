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

import base64
import json
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.tools.apihub_tool.clients.apihub_client import APIHubClient
import pytest
from requests.exceptions import HTTPError

# Mock data for API responses
MOCK_API_LIST = {
    "apis": [
        {"name": "projects/test-project/locations/us-central1/apis/api1"},
        {"name": "projects/test-project/locations/us-central1/apis/api2"},
    ]
}
MOCK_API_DETAIL = {
    "name": "projects/test-project/locations/us-central1/apis/api1",
    "versions": [
        "projects/test-project/locations/us-central1/apis/api1/versions/v1"
    ],
}
MOCK_API_VERSION = {
    "name": "projects/test-project/locations/us-central1/apis/api1/versions/v1",
    "specs": [
        "projects/test-project/locations/us-central1/apis/api1/versions/v1/specs/spec1"
    ],
}
MOCK_SPEC_CONTENT = {"contents": base64.b64encode(b"spec content").decode()}


# Test cases
class TestAPIHubClient:

  @pytest.fixture
  def client(self):
    return APIHubClient(access_token="mocked_token")

  @pytest.fixture
  def service_account_config(self):
    return json.dumps({
        "type": "service_account",
        "project_id": "test",
        "token_uri": "test.com",
        "client_email": "test@example.com",
        "private_key": "1234",
    })

  @patch("requests.get")
  def test_list_apis(self, mock_get, client):
    mock_get.return_value.json.return_value = MOCK_API_LIST
    mock_get.return_value.status_code = 200

    apis = client.list_apis("test-project", "us-central1")
    assert apis == MOCK_API_LIST["apis"]
    mock_get.assert_called_once_with(
        "https://apihub.googleapis.com/v1/projects/test-project/locations/us-central1/apis",
        headers={
            "accept": "application/json, text/plain, */*",
            "Authorization": "Bearer mocked_token",
        },
    )

  @patch("requests.get")
  def test_list_apis_empty(self, mock_get, client):
    mock_get.return_value.json.return_value = {"apis": []}
    mock_get.return_value.status_code = 200

    apis = client.list_apis("test-project", "us-central1")
    assert apis == []

  @patch("requests.get")
  def test_list_apis_error(self, mock_get, client):
    mock_get.return_value.raise_for_status.side_effect = HTTPError

    with pytest.raises(HTTPError):
      client.list_apis("test-project", "us-central1")

  @patch("requests.get")
  def test_get_api(self, mock_get, client):
    mock_get.return_value.json.return_value = MOCK_API_DETAIL
    mock_get.return_value.status_code = 200
    api = client.get_api(
        "projects/test-project/locations/us-central1/apis/api1"
    )
    assert api == MOCK_API_DETAIL
    mock_get.assert_called_once_with(
        "https://apihub.googleapis.com/v1/projects/test-project/locations/us-central1/apis/api1",
        headers={
            "accept": "application/json, text/plain, */*",
            "Authorization": "Bearer mocked_token",
        },
    )

  @patch("requests.get")
  def test_get_api_error(self, mock_get, client):
    mock_get.return_value.raise_for_status.side_effect = HTTPError
    with pytest.raises(HTTPError):
      client.get_api("projects/test-project/locations/us-central1/apis/api1")

  @patch("requests.get")
  def test_get_api_version(self, mock_get, client):
    mock_get.return_value.json.return_value = MOCK_API_VERSION
    mock_get.return_value.status_code = 200
    api_version = client.get_api_version(
        "projects/test-project/locations/us-central1/apis/api1/versions/v1"
    )
    assert api_version == MOCK_API_VERSION
    mock_get.assert_called_once_with(
        "https://apihub.googleapis.com/v1/projects/test-project/locations/us-central1/apis/api1/versions/v1",
        headers={
            "accept": "application/json, text/plain, */*",
            "Authorization": "Bearer mocked_token",
        },
    )

  @patch("requests.get")
  def test_get_api_version_error(self, mock_get, client):
    mock_get.return_value.raise_for_status.side_effect = HTTPError
    with pytest.raises(HTTPError):
      client.get_api_version(
          "projects/test-project/locations/us-central1/apis/api1/versions/v1"
      )

  @patch("requests.get")
  def test_get_spec_content(self, mock_get, client):
    mock_get.return_value.json.return_value = MOCK_SPEC_CONTENT
    mock_get.return_value.status_code = 200
    spec_content = client.get_spec_content(
        "projects/test-project/locations/us-central1/apis/api1/versions/v1/specs/spec1"
    )
    assert spec_content == "spec content"
    mock_get.assert_called_once_with(
        "https://apihub.googleapis.com/v1/projects/test-project/locations/us-central1/apis/api1/versions/v1/specs/spec1:contents",
        headers={
            "accept": "application/json, text/plain, */*",
            "Authorization": "Bearer mocked_token",
        },
    )

  @patch("requests.get")
  def test_get_spec_content_empty(self, mock_get, client):
    mock_get.return_value.json.return_value = {"contents": ""}
    mock_get.return_value.status_code = 200
    spec_content = client.get_spec_content(
        "projects/test-project/locations/us-central1/apis/api1/versions/v1/specs/spec1"
    )
    assert spec_content == ""

  @patch("requests.get")
  def test_get_spec_content_error(self, mock_get, client):
    mock_get.return_value.raise_for_status.side_effect = HTTPError
    with pytest.raises(HTTPError):
      client.get_spec_content(
          "projects/test-project/locations/us-central1/apis/api1/versions/v1/specs/spec1"
      )

  @pytest.mark.parametrize(
      "url_or_path, expected",
      [
          (
              "projects/test-project/locations/us-central1/apis/api1",
              (
                  "projects/test-project/locations/us-central1/apis/api1",
                  None,
                  None,
              ),
          ),
          (
              "projects/test-project/locations/us-central1/apis/api1/versions/v1",
              (
                  "projects/test-project/locations/us-central1/apis/api1",
                  "projects/test-project/locations/us-central1/apis/api1/versions/v1",
                  None,
              ),
          ),
          (
              "projects/test-project/locations/us-central1/apis/api1/versions/v1/specs/spec1",
              (
                  "projects/test-project/locations/us-central1/apis/api1",
                  "projects/test-project/locations/us-central1/apis/api1/versions/v1",
                  "projects/test-project/locations/us-central1/apis/api1/versions/v1/specs/spec1",
              ),
          ),
          (
              "https://console.cloud.google.com/apigee/api-hub/projects/test-project/locations/us-central1/apis/api1/versions/v1?project=test-project",
              (
                  "projects/test-project/locations/us-central1/apis/api1",
                  "projects/test-project/locations/us-central1/apis/api1/versions/v1",
                  None,
              ),
          ),
          (
              "https://console.cloud.google.com/apigee/api-hub/projects/test-project/locations/us-central1/apis/api1/versions/v1/specs/spec1?project=test-project",
              (
                  "projects/test-project/locations/us-central1/apis/api1",
                  "projects/test-project/locations/us-central1/apis/api1/versions/v1",
                  "projects/test-project/locations/us-central1/apis/api1/versions/v1/specs/spec1",
              ),
          ),
          (
              "/projects/test-project/locations/us-central1/apis/api1/versions/v1",
              (
                  "projects/test-project/locations/us-central1/apis/api1",
                  "projects/test-project/locations/us-central1/apis/api1/versions/v1",
                  None,
              ),
          ),
          (  # Added trailing slashes
              "projects/test-project/locations/us-central1/apis/api1/",
              (
                  "projects/test-project/locations/us-central1/apis/api1",
                  None,
                  None,
              ),
          ),
          (  # case location name
              "projects/test-project/locations/LOCATION/apis/api1/",
              (
                  "projects/test-project/locations/LOCATION/apis/api1",
                  None,
                  None,
              ),
          ),
          (
              "projects/p1/locations/l1/apis/a1/versions/v1/specs/s1",
              (
                  "projects/p1/locations/l1/apis/a1",
                  "projects/p1/locations/l1/apis/a1/versions/v1",
                  "projects/p1/locations/l1/apis/a1/versions/v1/specs/s1",
              ),
          ),
      ],
  )
  def test_extract_resource_name(self, client, url_or_path, expected):
    result = client._extract_resource_name(url_or_path)
    assert result == expected

  @pytest.mark.parametrize(
      "url_or_path, expected_error_message",
      [
          (
              "invalid-path",
              "Project ID not found in URL or path in APIHubClient.",
          ),
          (
              "projects/test-project",
              "Location not found in URL or path in APIHubClient.",
          ),
          (
              "projects/test-project/locations/us-central1",
              "API id not found in URL or path in APIHubClient.",
          ),
      ],
  )
  def test_extract_resource_name_invalid(
      self, client, url_or_path, expected_error_message
  ):
    with pytest.raises(ValueError, match=expected_error_message):
      client._extract_resource_name(url_or_path)

  @patch(
      "google.adk.tools.apihub_tool.clients.apihub_client.default_service_credential"
  )
  @patch(
      "google.adk.tools.apihub_tool.clients.apihub_client.service_account.Credentials.from_service_account_info"
  )
  def test_get_access_token_use_default_credential(
      self,
      mock_from_service_account_info,
      mock_default_service_credential,
  ):
    mock_credential = MagicMock()
    mock_credential.token = "default_token"
    mock_default_service_credential.return_value = (
        mock_credential,
        "project_id",
    )
    mock_config_credential = MagicMock()
    mock_config_credential.token = "config_token"
    mock_from_service_account_info.return_value = mock_config_credential

    client = APIHubClient()
    token = client._get_access_token()
    assert token == "default_token"
    # Verify default_service_credential is called with the correct scopes parameter
    mock_default_service_credential.assert_called_once_with(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    mock_credential.refresh.assert_called_once()
    assert client.credential_cache == mock_credential

  @patch(
      "google.adk.tools.apihub_tool.clients.apihub_client.default_service_credential"
  )
  @patch(
      "google.adk.tools.apihub_tool.clients.apihub_client.service_account.Credentials.from_service_account_info"
  )
  def test_get_access_token_use_configured_service_account(
      self,
      mock_from_service_account_info,
      mock_default_service_credential,
      service_account_config,
  ):
    mock_credential = MagicMock()
    mock_credential.token = "default_token"
    mock_default_service_credential.return_value = (
        mock_credential,
        "project_id",
    )
    mock_config_credential = MagicMock()
    mock_config_credential.token = "config_token"
    mock_from_service_account_info.return_value = mock_config_credential

    client = APIHubClient(service_account_json=service_account_config)
    token = client._get_access_token()

    assert token == "config_token"
    mock_from_service_account_info.assert_called_once_with(
        json.loads(service_account_config),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    mock_config_credential.refresh.assert_called_once()
    assert client.credential_cache == mock_config_credential

  @patch(
      "google.adk.tools.apihub_tool.clients.apihub_client.default_service_credential"
  )
  def test_get_access_token_not_expired_use_cached_token(
      self, mock_default_credential
  ):
    mock_credentials = MagicMock()
    mock_credentials.token = "default_service_account_token"
    mock_default_credential.return_value = (mock_credentials, "")

    client = APIHubClient()
    # Call #1: Setup cache
    token = client._get_access_token()
    assert token == "default_service_account_token"
    mock_default_credential.assert_called_once()

    # Call #2: Reuse cache
    mock_credentials.reset_mock()
    mock_credentials.expired = False
    token = client._get_access_token()
    assert token == "default_service_account_token"
    mock_credentials.refresh.assert_not_called()

  @patch(
      "google.adk.tools.apihub_tool.clients.apihub_client.default_service_credential"
  )
  def test_get_access_token_expired_refresh(self, mock_default_credential):
    mock_credentials = MagicMock()
    mock_credentials.token = "default_service_account_token"
    mock_default_credential.return_value = (mock_credentials, "")
    client = APIHubClient()

    # Call #1: Setup cache
    token = client._get_access_token()
    assert token == "default_service_account_token"
    mock_default_credential.assert_called_once()

    # Call #2: Cache expired
    mock_credentials.reset_mock()
    mock_credentials.expired = True
    token = client._get_access_token()
    mock_credentials.refresh.assert_called_once()
    assert token == "default_service_account_token"

  @patch(
      "google.adk.tools.apihub_tool.clients.apihub_client.default_service_credential"
  )
  def test_get_access_token_no_credentials(
      self, mock_default_service_credential
  ):
    mock_default_service_credential.return_value = (None, None)
    with pytest.raises(
        ValueError,
        match=(
            "Please provide a service account or an access token to API Hub"
            " client."
        ),
    ):
      # no service account client
      APIHubClient()._get_access_token()

  @patch("requests.get")
  def test_get_spec_content_api_level(self, mock_get, client):
    mock_get.side_effect = [
        MagicMock(status_code=200, json=lambda: MOCK_API_DETAIL),  # For get_api
        MagicMock(
            status_code=200, json=lambda: MOCK_API_VERSION
        ),  # For get_api_version
        MagicMock(
            status_code=200, json=lambda: MOCK_SPEC_CONTENT
        ),  # For get_spec_content
    ]

    content = client.get_spec_content(
        "projects/test-project/locations/us-central1/apis/api1"
    )
    assert content == "spec content"
    # Check calls - get_api, get_api_version, then get_spec_content
    assert mock_get.call_count == 3

  @patch("requests.get")
  def test_get_spec_content_version_level(self, mock_get, client):
    mock_get.side_effect = [
        MagicMock(
            status_code=200, json=lambda: MOCK_API_VERSION
        ),  # For get_api_version
        MagicMock(
            status_code=200, json=lambda: MOCK_SPEC_CONTENT
        ),  # For get_spec_content
    ]

    content = client.get_spec_content(
        "projects/test-project/locations/us-central1/apis/api1/versions/v1"
    )
    assert content == "spec content"
    assert mock_get.call_count == 2  # get_api_version and get_spec_content

  @patch("requests.get")
  def test_get_spec_content_spec_level(self, mock_get, client):
    mock_get.return_value.json.return_value = MOCK_SPEC_CONTENT
    mock_get.return_value.status_code = 200

    content = client.get_spec_content(
        "projects/test-project/locations/us-central1/apis/api1/versions/v1/specs/spec1"
    )
    assert content == "spec content"
    mock_get.assert_called_once()  # Only get_spec_content should be called

  @patch("requests.get")
  def test_get_spec_content_no_versions(self, mock_get, client):
    mock_get.return_value.json.return_value = {
        "name": "projects/test-project/locations/us-central1/apis/api1",
        "versions": [],
    }  # No versions
    mock_get.return_value.status_code = 200
    with pytest.raises(
        ValueError,
        match=(
            "No versions found in API Hub resource:"
            " projects/test-project/locations/us-central1/apis/api1"
        ),
    ):
      client.get_spec_content(
          "projects/test-project/locations/us-central1/apis/api1"
      )

  @patch("requests.get")
  def test_get_spec_content_no_specs(self, mock_get, client):
    mock_get.side_effect = [
        MagicMock(status_code=200, json=lambda: MOCK_API_DETAIL),
        MagicMock(
            status_code=200,
            json=lambda: {
                "name": "projects/test-project/locations/us-central1/apis/api1/versions/v1",
                "specs": [],
            },
        ),  # No specs
    ]

    with pytest.raises(
        ValueError,
        match=(
            "No specs found in API Hub version:"
            " projects/test-project/locations/us-central1/apis/api1/versions/v1"
        ),
    ):
      client.get_spec_content(
          "projects/test-project/locations/us-central1/apis/api1/versions/v1"
      )

  @patch("requests.get")
  def test_get_spec_content_invalid_path(self, mock_get, client):
    with pytest.raises(
        ValueError,
        match=(
            "Project ID not found in URL or path in APIHubClient. Input"
            " path is 'invalid-path'."
        ),
    ):
      client.get_spec_content("invalid-path")


if __name__ == "__main__":
  pytest.main([__file__])
