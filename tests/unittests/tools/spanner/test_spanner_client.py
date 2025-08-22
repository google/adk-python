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

from __future__ import annotations

import os
import re
from unittest import mock

from google.adk.tools.spanner.client import get_spanner_client
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2.credentials import Credentials
import pytest


def test_spanner_client_project():
  """Test spanner client project."""
  # Trigger the spanner client creation
  client = get_spanner_client(
      project="test-gcp-project",
      credentials=mock.create_autospec(Credentials, instance=True),
  )

  # Verify that the client has the desired project set
  assert client.project == "test-gcp-project"


def test_spanner_client_project_set_explicit():
  """Test spanner client creation does not invoke default auth."""
  # Let's simulate that no environment variables are set, so that any project
  # set in there does not interfere with this test
  with mock.patch.dict(os.environ, {}, clear=True):
    with mock.patch("google.auth.default", autospec=True) as mock_default_auth:
      # Simulate exception from default auth
      mock_default_auth.side_effect = DefaultCredentialsError(
          "Your default credentials were not found"
      )

      # Trigger the spanner client creation
      client = get_spanner_client(
          project="test-gcp-project",
          credentials=mock.create_autospec(Credentials, instance=True),
      )

      # If we are here that already means client creation did not call default
      # auth (otherwise we would have run into DefaultCredentialsError set
      # above). For the sake of explicitness, trivially assert that the default
      # auth was not called, and yet the project was set correctly
      mock_default_auth.assert_not_called()
      assert client.project == "test-gcp-project"


def test_spanner_client_project_set_with_default_auth():
  """Test spanner client creation invokes default auth to set the project."""
  # Let's simulate that no environment variables are set, so that any project
  # set in there does not interfere with this test
  with mock.patch.dict(os.environ, {}, clear=True):
    with mock.patch("google.auth.default", autospec=True) as mock_default_auth:
      # Simulate credentials
      mock_creds = mock.create_autospec(Credentials, instance=True)

      # Simulate output of the default auth
      mock_default_auth.return_value = (mock_creds, "test-gcp-project")

      # Trigger the spanner client creation
      client = get_spanner_client(
          project=None,
          credentials=mock_creds,
      )

      # Verify that default auth was called once to set the client project
      mock_default_auth.assert_called_once()
      assert client.project == "test-gcp-project"


def test_spanner_client_project_set_with_env():
  """Test spanner client creation sets the project from environment variable."""
  # Let's simulate the project set in environment variables
  with mock.patch.dict(
      os.environ, {"GOOGLE_CLOUD_PROJECT": "test-gcp-project"}, clear=True
  ):
    with mock.patch("google.auth.default", autospec=True) as mock_default_auth:
      # Simulate exception from default auth
      mock_default_auth.side_effect = DefaultCredentialsError(
          "Your default credentials were not found"
      )

      # Trigger the spanner client creation
      client = get_spanner_client(
          project=None,
          credentials=mock.create_autospec(Credentials, instance=True),
      )

      # If we are here that already means client creation did not call default
      # auth (otherwise we would have run into DefaultCredentialsError set
      # above). For the sake of explicitness, trivially assert that the default
      # auth was not called, and yet the project was set correctly
      mock_default_auth.assert_not_called()
      assert client.project == "test-gcp-project"


def test_spanner_client_user_agent():
  """Test spanner client user agent."""
  # Patch the Client constructor
  with mock.patch(
      "google.cloud.spanner.Client", autospec=True
  ) as mock_client_class:
    # The mock instance that will be returned by spanner.Client()
    mock_instance = mock_client_class.return_value
    # The real spanner.Client instance has a `_client_info` attribute.
    # We need to add it to our mock instance so that the user_agent can be set.
    mock_instance._client_info = mock.Mock()

    # Call the function that creates the client
    client = get_spanner_client(
        project="test-gcp-project",
        credentials=mock.create_autospec(Credentials, instance=True),
    )

    # Verify that the Spanner Client was instantiated.
    mock_client_class.assert_called_once_with(
        project="test-gcp-project",
        credentials=mock.ANY,
    )

    # Verify that the user_agent was set on the client instance.
    # The client returned by get_spanner_client is the mock instance.
    assert re.search(
        r"adk-spanner-tool google-adk/([0-9A-Za-z._\-+/]+)",
        client._client_info.user_agent,
    )
