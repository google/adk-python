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

"""Tests for BigQueryCheckpointStore."""

from datetime import datetime
from datetime import timezone
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.durable.stores.bigquery_checkpoint_store import BigQueryCheckpointStore
import pytest


class TestBigQueryCheckpointStore:
  """Tests for BigQueryCheckpointStore."""

  @pytest.fixture
  def store(self):
    """Create a store instance for testing."""
    return BigQueryCheckpointStore(
        project="test-project",
        dataset="test_dataset",
        gcs_bucket="test-bucket",
    )

  def test_init(self, store):
    """Test store initialization."""
    assert store._project == "test-project"
    assert store._dataset == "test_dataset"
    assert store._gcs_bucket == "test-bucket"
    assert store._sessions_table == "sessions"
    assert store._checkpoints_table == "checkpoints"
    assert store._location == "US"

  def test_table_ids(self, store):
    """Test table ID generation."""
    assert store._sessions_table_id == "test-project.test_dataset.sessions"
    assert (
        store._checkpoints_table_id == "test-project.test_dataset.checkpoints"
    )

  def test_gcs_uri_generation(self, store):
    """Test GCS URI generation."""
    uri = store._get_gcs_uri("session-123", 5)
    assert uri == "gs://test-bucket/checkpoints/session-123/5.json.gz"

  @pytest.mark.asyncio
  async def test_create_session(self, store):
    """Test session creation."""
    mock_client = MagicMock()
    mock_client.insert_rows_json.return_value = []

    with patch.object(store, "_get_bq_client", return_value=mock_client):
      with patch.object(
          store, "get_session", new_callable=AsyncMock
      ) as mock_get:
        mock_get.return_value = None

        session = await store.create_session(
            session_id="test-session",
            agent_name="test_agent",
            metadata={"key": "value"},
        )

        assert session.session_id == "test-session"
        assert session.agent_name == "test_agent"
        assert session.status == "active"
        assert session.current_checkpoint_seq == 0
        assert session.metadata == {"key": "value"}

        mock_client.insert_rows_json.assert_called_once()

  @pytest.mark.asyncio
  async def test_create_session_already_exists(self, store):
    """Test session creation when session already exists."""
    with patch.object(store, "get_session", new_callable=AsyncMock) as mock_get:
      from google.adk.durable.stores.base_checkpoint_store import SessionMetadata

      mock_get.return_value = SessionMetadata(
          session_id="test-session",
          status="active",
          agent_name="test_agent",
          created_at=datetime.now(timezone.utc),
          updated_at=datetime.now(timezone.utc),
          current_checkpoint_seq=0,
      )

      with pytest.raises(ValueError, match="already exists"):
        await store.create_session(
            session_id="test-session",
            agent_name="test_agent",
        )

  @pytest.mark.asyncio
  async def test_write_checkpoint(self, store):
    """Test checkpoint writing."""
    mock_bq_client = MagicMock()
    mock_bq_client.insert_rows_json.return_value = []
    mock_bq_client.query.return_value.result.return_value = None

    mock_storage_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_storage_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    with patch.object(store, "_get_bq_client", return_value=mock_bq_client):
      with patch.object(
          store, "_get_storage_client", return_value=mock_storage_client
      ):
        with patch.object(
            store, "get_session", new_callable=AsyncMock
        ) as mock_get:
          from google.adk.durable.stores.base_checkpoint_store import SessionMetadata

          mock_get.return_value = SessionMetadata(
              session_id="test-session",
              status="active",
              agent_name="test_agent",
              created_at=datetime.now(timezone.utc),
              updated_at=datetime.now(timezone.utc),
              current_checkpoint_seq=0,
          )

          checkpoint = await store.write_checkpoint(
              session_id="test-session",
              checkpoint_seq=1,
              state_blob=b'{"state": "data"}',
              agent_state={"key": "value"},
              trigger="async_boundary",
          )

          assert checkpoint.session_id == "test-session"
          assert checkpoint.checkpoint_seq == 1
          assert checkpoint.trigger == "async_boundary"
          assert checkpoint.agent_state == {"key": "value"}

          # Verify GCS upload was called
          mock_blob.upload_from_string.assert_called_once()

          # Verify BQ insert was called
          mock_bq_client.insert_rows_json.assert_called_once()

  @pytest.mark.asyncio
  async def test_write_checkpoint_invalid_seq(self, store):
    """Test checkpoint writing with invalid sequence number."""
    with patch.object(store, "get_session", new_callable=AsyncMock) as mock_get:
      from google.adk.durable.stores.base_checkpoint_store import SessionMetadata

      mock_get.return_value = SessionMetadata(
          session_id="test-session",
          status="active",
          agent_name="test_agent",
          created_at=datetime.now(timezone.utc),
          updated_at=datetime.now(timezone.utc),
          current_checkpoint_seq=5,
      )

      with pytest.raises(ValueError, match="must be greater"):
        await store.write_checkpoint(
            session_id="test-session",
            checkpoint_seq=3,  # Less than current (5)
            state_blob=b"data",
        )

  @pytest.mark.asyncio
  async def test_write_checkpoint_session_not_found(self, store):
    """Test checkpoint writing when session doesn't exist."""
    with patch.object(store, "get_session", new_callable=AsyncMock) as mock_get:
      mock_get.return_value = None

      with pytest.raises(ValueError, match="not found"):
        await store.write_checkpoint(
            session_id="nonexistent",
            checkpoint_seq=1,
            state_blob=b"data",
        )

  @pytest.mark.asyncio
  async def test_acquire_lease_success(self, store):
    """Test successful lease acquisition."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.num_dml_affected_rows = 1
    mock_client.query.return_value.result.return_value = mock_result

    with patch.object(store, "_get_bq_client", return_value=mock_client):
      result = await store.acquire_lease(
          session_id="test-session",
          lease_id="lease-123",
          timeout_seconds=300,
      )

      assert result is True
      mock_client.query.assert_called_once()

  @pytest.mark.asyncio
  async def test_acquire_lease_failure(self, store):
    """Test failed lease acquisition (another lease active)."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.num_dml_affected_rows = 0
    mock_client.query.return_value.result.return_value = mock_result

    with patch.object(store, "_get_bq_client", return_value=mock_client):
      result = await store.acquire_lease(
          session_id="test-session",
          lease_id="lease-123",
          timeout_seconds=300,
      )

      assert result is False

  @pytest.mark.asyncio
  async def test_release_lease(self, store):
    """Test lease release."""
    mock_client = MagicMock()
    mock_client.query.return_value.result.return_value = None

    with patch.object(store, "_get_bq_client", return_value=mock_client):
      await store.release_lease(
          session_id="test-session",
          lease_id="lease-123",
      )

      mock_client.query.assert_called_once()

  @pytest.mark.asyncio
  async def test_renew_lease_success(self, store):
    """Test successful lease renewal."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.num_dml_affected_rows = 1
    mock_client.query.return_value.result.return_value = mock_result

    with patch.object(store, "_get_bq_client", return_value=mock_client):
      result = await store.renew_lease(
          session_id="test-session",
          lease_id="lease-123",
          timeout_seconds=600,
      )

      assert result is True

  @pytest.mark.asyncio
  async def test_renew_lease_failure(self, store):
    """Test failed lease renewal (lease not held)."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.num_dml_affected_rows = 0
    mock_client.query.return_value.result.return_value = mock_result

    with patch.object(store, "_get_bq_client", return_value=mock_client):
      result = await store.renew_lease(
          session_id="test-session",
          lease_id="lease-123",
          timeout_seconds=600,
      )

      assert result is False
