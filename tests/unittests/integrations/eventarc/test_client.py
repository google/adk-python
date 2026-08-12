# mypy: ignore-errors
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
import unittest
from unittest import mock

from google.adk.integrations.eventarc import _client as client
import google.auth.compute_engine.credentials
import google.auth.credentials
import google.auth.identity_pool
import google.auth.impersonated_credentials
import google.auth.pluggable
import google.oauth2.credentials
import google.oauth2.service_account


class TestEventarcClient(unittest.IsolatedAsyncioTestCase):

  def test_get_credential_id(self):
    # Service Account
    sa_creds = google.oauth2.service_account.Credentials(
        signer=mock.Mock(),
        service_account_email="test@test.com",
        token_uri="https://oauth2.mtls.googleapis.com/token",
    )
    self.assertEqual(client._get_credential_id(sa_creds), "test@test.com")

    # Impersonated (Uses service_account_email under the hood in google-auth)
    imp_creds = google.auth.impersonated_credentials.Credentials(
        source_credentials=mock.Mock(),
        target_principal="imp@test.com",
        target_scopes=[],
    )
    self.assertEqual(client._get_credential_id(imp_creds), "imp@test.com")

    # Compute Engine (ADC)
    gce_creds = google.auth.compute_engine.credentials.Credentials()
    self.assertEqual(
        client._get_credential_id(gce_creds), "ComputeEngineCredentials"
    )

    # Fallback
    fallback_creds = mock.create_autospec(
        google.auth.credentials.Credentials, instance=True
    )
    # create_autospec dynamically configures the mock class __module__, but we ensure
    # it doesn't accidentally match Compute Engine.
    self.assertEqual(
        client._get_credential_id(fallback_creds), str(id(fallback_creds))
    )

    # Identity Pool (File)
    ip_file_creds1 = google.auth.identity_pool.Credentials(
        audience="aud1",
        subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        credential_source={"file": "path1"},
    )
    ip_file_creds2 = google.auth.identity_pool.Credentials(
        audience="aud1",
        subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        credential_source={"file": "path1"},
    )
    ip_file_creds_diff = google.auth.identity_pool.Credentials(
        audience="aud1",
        subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        credential_source={"file": "path2"},
    )
    self.assertEqual(
        client._get_credential_id(ip_file_creds1),
        client._get_credential_id(ip_file_creds2),
    )
    self.assertNotEqual(
        client._get_credential_id(ip_file_creds1),
        client._get_credential_id(ip_file_creds_diff),
    )
    self.assertTrue(
        client._get_credential_id(ip_file_creds1).startswith(
            "ExternalAccount:aud1:"
        )
    )

    # Identity Pool (Supplier)
    supplier1 = lambda context: "token"
    supplier2 = lambda context: "token"
    ip_sup_creds1 = google.auth.identity_pool.Credentials(
        audience="aud1",
        subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        subject_token_supplier=supplier1,
    )
    ip_sup_creds2 = google.auth.identity_pool.Credentials(
        audience="aud1",
        subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        subject_token_supplier=supplier1,
    )
    ip_sup_creds_diff = google.auth.identity_pool.Credentials(
        audience="aud1",
        subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        subject_token_supplier=supplier2,
    )
    self.assertEqual(
        client._get_credential_id(ip_sup_creds1),
        client._get_credential_id(ip_sup_creds2),
    )
    self.assertNotEqual(
        client._get_credential_id(ip_sup_creds1),
        client._get_credential_id(ip_sup_creds_diff),
    )

    # Pluggable
    plug_creds1 = google.auth.pluggable.Credentials(
        audience="aud1",
        subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        token_url="https://sts.mtls.googleapis.com/v1/token",
        credential_source={"executable": {"command": "cmd1"}},
    )
    plug_creds2 = google.auth.pluggable.Credentials(
        audience="aud1",
        subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        token_url="https://sts.mtls.googleapis.com/v1/token",
        credential_source={"executable": {"command": "cmd1"}},
    )
    plug_creds_diff = google.auth.pluggable.Credentials(
        audience="aud1",
        subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        token_url="https://sts.mtls.googleapis.com/v1/token",
        credential_source={"executable": {"command": "cmd2"}},
    )
    self.assertEqual(
        client._get_credential_id(plug_creds1),
        client._get_credential_id(plug_creds2),
    )
    self.assertNotEqual(
        client._get_credential_id(plug_creds1),
        client._get_credential_id(plug_creds_diff),
    )

    # User Credentials (with refresh token)
    user_creds1 = google.oauth2.credentials.Credentials(
        token="token1",
        refresh_token="refresh1",
        token_uri="uri1",
        client_id="client1",
        client_secret="secret1",
    )
    user_creds2 = google.oauth2.credentials.Credentials(
        token="token2",
        refresh_token="refresh1",
        token_uri="uri1",
        client_id="client1",
        client_secret="secret1",
    )
    user_creds_diff = google.oauth2.credentials.Credentials(
        token="token1",
        refresh_token="refresh2",
        token_uri="uri1",
        client_id="client1",
        client_secret="secret1",
    )
    self.assertEqual(
        client._get_credential_id(user_creds1),
        client._get_credential_id(user_creds2),
    )
    self.assertNotEqual(
        client._get_credential_id(user_creds1),
        client._get_credential_id(user_creds_diff),
    )
    self.assertTrue(
        client._get_credential_id(user_creds1).startswith("UserCredentials:")
    )

    # Downscoped Credentials (Mocked to avoid build dependency on google.auth.downscoped)
    class DownscopedCredentialsForTest:
      pass

    DownscopedCredentialsForTest.__module__ = "google.auth.downscoped"

    source_creds = google.oauth2.service_account.Credentials(
        signer=mock.Mock(),
        service_account_email="sa1@p1.iam.gserviceaccount.com",
        token_uri="https://oauth2.mtls.googleapis.com/token",
    )

    boundary1 = mock.Mock()
    boundary1.to_json.return_value = {"rules": ["rule1"]}
    boundary_diff = mock.Mock()
    boundary_diff.to_json.return_value = {"rules": ["rule2"]}

    ds_creds1 = DownscopedCredentialsForTest()
    ds_creds1._source_credentials = source_creds
    ds_creds1._credential_access_boundary = boundary1

    ds_creds2 = DownscopedCredentialsForTest()
    ds_creds2._source_credentials = source_creds
    ds_creds2._credential_access_boundary = boundary1

    ds_creds_diff = DownscopedCredentialsForTest()
    ds_creds_diff._source_credentials = source_creds
    ds_creds_diff._credential_access_boundary = boundary_diff

    self.assertEqual(
        client._get_credential_id(ds_creds1),
        client._get_credential_id(ds_creds2),
    )
    self.assertNotEqual(
        client._get_credential_id(ds_creds1),
        client._get_credential_id(ds_creds_diff),
    )
    cred_id = client._get_credential_id(ds_creds1)
    self.assertTrue(
        cred_id.startswith("Downscoped:sa1@p1.iam.gserviceaccount.com:")
    )

  @mock.patch.object(client, "eventarc_publishing_v1", autospec=True)
  async def test_get_publisher_client_cache(self, mock_eventarc_publishing):
    # Reset cache
    client._publisher_client_cache.clear()

    creds = mock.create_autospec(
        google.auth.credentials.Credentials, instance=True
    )

    mock_client_cls = mock.Mock()
    mock_eventarc_publishing.PublisherAsyncClient = mock_client_cls

    # Return a new mock instance each time
    mock_client_cls.side_effect = lambda **kwargs: mock.Mock()

    # First call creates the client
    c1 = await client.get_publisher_client(credentials=creds, project_id="p1")
    mock_client_cls.assert_called_once()

    # Second call returns cached client
    c2 = await client.get_publisher_client(credentials=creds, project_id="p1")
    mock_client_cls.assert_called_once()
    self.assertIs(c1, c2)

    # Different project creates new client
    c3 = await client.get_publisher_client(credentials=creds, project_id="p2")
    self.assertEqual(mock_client_cls.call_count, 2)
    self.assertIsNot(c1, c3)

  @mock.patch.object(client, "eventarc_publishing_v1", autospec=True)
  async def test_remove_publisher_client(self, mock_eventarc_publishing):
    client._publisher_client_cache.clear()

    mock_client_cls = mock.Mock()
    mock_eventarc_publishing.PublisherAsyncClient = mock_client_cls
    mock_client = mock.Mock()
    mock_client.transport = mock.Mock()
    mock_client_cls.return_value = mock_client

    creds = mock.create_autospec(
        google.auth.credentials.Credentials, instance=True
    )
    c1 = await client.get_publisher_client(credentials=creds, project_id="p1")
    self.assertEqual(len(client._publisher_client_cache), 1)

    # Remove client
    await client.remove_publisher_client(credentials=creds, project_id="p1")
    self.assertEqual(len(client._publisher_client_cache), 0)
    mock_client.transport.close.assert_called_once()

    # Remove again is safe
    await client.remove_publisher_client(credentials=creds, project_id="p1")

  @mock.patch.object(client, "eventarc_publishing_v1", autospec=True)
  async def test_publisher_client_cache_lru_eviction(
      self, mock_eventarc_publishing
  ):
    """Verifies LRU eviction and transport closing when cache is full."""
    client._publisher_client_cache.clear()

    mock_client_cls = mock.Mock()
    mock_eventarc_publishing.PublisherAsyncClient = mock_client_cls

    # Track created mock clients and mock their transports
    clients_list = []

    def create_mock_client(**kwargs):
      mc = mock.Mock()
      mc.transport = mock.Mock()
      clients_list.append(mc)
      return mc

    mock_client_cls.side_effect = create_mock_client

    creds = mock.create_autospec(
        google.auth.credentials.Credentials, instance=True
    )

    # Fill cache to MAX_SIZE
    for i in range(client._CACHE_MAX_SIZE):
      await client.get_publisher_client(
          credentials=creds, project_id=f"project-{i}"
      )

    # Hit project-0 to make it recently used
    await client.get_publisher_client(credentials=creds, project_id="project-0")

    # Now project-1 should be the oldest.
    # Add another client to trigger eviction
    next_proj = f"project-{client._CACHE_MAX_SIZE}"
    await client.get_publisher_client(credentials=creds, project_id=next_proj)

    self.assertEqual(
        len(client._publisher_client_cache), client._CACHE_MAX_SIZE
    )

    # project-1 client should be evicted (which is index 1 in clients_list)
    clients_list[1].transport.close.assert_called_once()

    mock_client_cls.reset_mock()
    # project-1 should be evicted
    await client.get_publisher_client(credentials=creds, project_id="project-1")
    mock_client_cls.assert_called_once()

    mock_client_cls.reset_mock()
    # project-0 should still be in cache
    await client.get_publisher_client(credentials=creds, project_id="project-0")
    mock_client_cls.assert_not_called()

  @mock.patch.object(client, "eventarc_publishing_v1", autospec=True)
  async def test_get_publisher_client_cache_external_account(
      self, mock_eventarc_publishing
  ):
    client._publisher_client_cache.clear()
    mock_client_cls = mock.Mock()
    mock_eventarc_publishing.PublisherAsyncClient = mock_client_cls
    mock_client_cls.side_effect = lambda **kwargs: mock.Mock()

    creds1 = google.auth.identity_pool.Credentials(
        audience="aud1",
        subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        credential_source={"file": "path1"},
    )
    creds2 = google.auth.identity_pool.Credentials(
        audience="aud1",
        subject_token_type="urn:ietf:params:oauth:token-type:jwt",
        credential_source={"file": "path1"},
    )

    c1 = await client.get_publisher_client(credentials=creds1, project_id="p1")
    mock_client_cls.assert_called_once()

    c2 = await client.get_publisher_client(credentials=creds2, project_id="p1")
    # Should be a cache hit, so call count remains 1
    mock_client_cls.assert_called_once()
    self.assertIs(c1, c2)

  @mock.patch.object(client, "eventarc_publishing_v1", autospec=True)
  async def test_get_publisher_client_cache_user_credentials(
      self, mock_eventarc_publishing
  ):
    client._publisher_client_cache.clear()
    mock_client_cls = mock.Mock()
    mock_eventarc_publishing.PublisherAsyncClient = mock_client_cls
    mock_client_cls.side_effect = lambda **kwargs: mock.Mock()

    creds1 = google.oauth2.credentials.Credentials(
        token="token1",
        refresh_token="refresh1",
        token_uri="uri1",
        client_id="client1",
        client_secret="secret1",
    )
    creds2 = google.oauth2.credentials.Credentials(
        token="token2",
        refresh_token="refresh1",
        token_uri="uri1",
        client_id="client1",
        client_secret="secret1",
    )

    c1 = await client.get_publisher_client(credentials=creds1, project_id="p1")
    mock_client_cls.assert_called_once()

    c2 = await client.get_publisher_client(credentials=creds2, project_id="p1")
    # Should be a cache hit, so call count remains 1
    mock_client_cls.assert_called_once()
    self.assertIs(c1, c2)


if __name__ == "__main__":
  unittest.main()
