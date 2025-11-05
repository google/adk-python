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

from unittest import mock

from google.adk.tools.bigtable import client
from google.auth.credentials import Credentials


def test_get_bigtable_data_client():
  """Test get_bigtable_client function."""
  with mock.patch(
      "google.cloud.bigtable.data.BigtableDataClient"
  ) as MockBigtableDataClient:
    mock_creds = mock.create_autospec(Credentials, instance=True)
    client.get_bigtable_data_client(
        project="test-project", credentials=mock_creds
    )
    MockBigtableDataClient.assert_called_once_with(
        project="test-project",
        credentials=mock_creds,
        client_info=mock.ANY,
    )


def test_get_bigtable_admin_client():
  """Test get_bigtable_admin_client function."""
  with mock.patch("google.cloud.bigtable.Client") as BigtableDataClient:
    mock_creds = mock.create_autospec(Credentials, instance=True)
    client.get_bigtable_admin_client(
        project="test-project", credentials=mock_creds
    )
    # Admin client is a BigtableDataClient created with admin=True.
    BigtableDataClient.assert_called_once_with(
        project="test-project",
        admin=True,
        credentials=mock_creds,
        client_info=mock.ANY,
    )
