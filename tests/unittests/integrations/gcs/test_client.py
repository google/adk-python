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

from unittest import mock

from google.adk.integrations.gcs import client
from google.auth.credentials import Credentials
from google.cloud import storage
import google.oauth2.credentials


def test_get_gcs_client():
  """Test get_gcs_client function."""
  with mock.patch.object(storage, "Client", autospec=True) as MockGCSClient:
    mock_creds = mock.create_autospec(Credentials, instance=True)
    client.get_gcs_client(project="test-project", credentials=mock_creds)
    MockGCSClient.assert_called_once_with(
        project="test-project",
        credentials=mock_creds,
        client_info=mock.ANY,
    )


def test_get_gcs_client_is_never_shared_between_credentials():
  """Test each client is authenticated as the credentials it was built for."""

  def fake_storage_client(**kwargs):
    made = mock.Mock()
    # Record only the token. Keeping the credentials object itself alive would
    # stop its address being reused, which is the collision under test.
    made.token = kwargs["credentials"].token
    return made

  # Patched with a plain function rather than a Mock, because a Mock retains
  # every credentials object it was called with in call_args_list.
  with mock.patch.object(storage, "Client", new=fake_storage_client):
    for i in range(200):
      # A short-lived credentials object per call, as a tool invocation makes.
      credentials = google.oauth2.credentials.Credentials(token=f"token-{i}")
      gcs_client = client.get_gcs_client(credentials=credentials)
      assert gcs_client.token == f"token-{i}"


def test_get_gcs_client_returns_a_new_client_per_call():
  """Test the same credentials do not hand out one shared client."""
  with mock.patch.object(storage, "Client", autospec=True) as MockGCSClient:
    MockGCSClient.side_effect = lambda **kwargs: mock.Mock()
    mock_creds = mock.create_autospec(Credentials, instance=True)

    client1 = client.get_gcs_client(
        project="test-project", credentials=mock_creds
    )
    client2 = client.get_gcs_client(
        project="test-project", credentials=mock_creds
    )

    assert client1 is not client2
    assert MockGCSClient.call_count == 2
