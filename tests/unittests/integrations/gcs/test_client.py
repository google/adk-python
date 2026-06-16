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


def test_get_gcs_client_cache():
  """Test get_gcs_client caches and reuses the client instance."""
  client._client_cache.clear()  # pylint: disable=protected-access

  with mock.patch.object(storage, "Client", autospec=True) as MockGCSClient:
    mock_creds = mock.create_autospec(Credentials, instance=True)

    # First call - cache miss
    client1 = client.get_gcs_client(
        project="test-project", credentials=mock_creds
    )

    # Second call - cache hit
    client2 = client.get_gcs_client(
        project="test-project", credentials=mock_creds
    )

    assert client1 is client2
    MockGCSClient.assert_called_once_with(
        project="test-project",
        credentials=mock_creds,
        client_info=mock.ANY,
    )
