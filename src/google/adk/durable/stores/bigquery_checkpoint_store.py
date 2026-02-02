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

"""BigQuery + GCS implementation of durable session checkpoint store."""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone
import hashlib
import json
import logging
from typing import Any
from typing import Dict
from typing import Optional
import uuid

from ...utils.feature_decorator import experimental
from .base_checkpoint_store import Checkpoint
from .base_checkpoint_store import DurableSessionStore
from .base_checkpoint_store import SessionMetadata

logger = logging.getLogger("google_adk." + __name__)


@experimental
class BigQueryCheckpointStore(DurableSessionStore):
  """Checkpoint store using BigQuery for metadata and GCS for state blobs.

  This implementation stores:
  - Session metadata and checkpoint records in BigQuery tables
  - Large state blobs in Google Cloud Storage

  Prerequisites:
  - BigQuery dataset with sessions and checkpoints tables
  - GCS bucket for state blobs
  - Appropriate IAM permissions

  Example:
    ```python
    store = BigQueryCheckpointStore(
        project="my-project",
        dataset="adk_metadata",
        gcs_bucket="my-project-adk-checkpoints",
    )

    # Create a session
    await store.create_session(
        session_id="sess-123",
        agent_name="my_agent",
    )

    # Write a checkpoint
    await store.write_checkpoint(
        session_id="sess-123",
        checkpoint_seq=1,
        state_blob=b"...",
    )

    # Read it back
    checkpoint, blob = await store.read_latest_checkpoint(session_id="sess-123")
    ```
  """

  def __init__(
      self,
      *,
      project: str,
      dataset: str,
      gcs_bucket: str,
      sessions_table: str = "sessions",
      checkpoints_table: str = "checkpoints",
      location: str = "US",
  ):
    """Initialize the BigQuery checkpoint store.

    Args:
      project: GCP project ID.
      dataset: BigQuery dataset name.
      gcs_bucket: GCS bucket name for state blobs.
      sessions_table: Name of the sessions table.
      checkpoints_table: Name of the checkpoints table.
      location: BigQuery dataset location.
    """
    self._project = project
    self._dataset = dataset
    self._gcs_bucket = gcs_bucket
    self._sessions_table = sessions_table
    self._checkpoints_table = checkpoints_table
    self._location = location

    # Lazy-loaded clients
    self._bq_client = None
    self._storage_client = None

  @property
  def _sessions_table_id(self) -> str:
    return f"{self._project}.{self._dataset}.{self._sessions_table}"

  @property
  def _checkpoints_table_id(self) -> str:
    return f"{self._project}.{self._dataset}.{self._checkpoints_table}"

  def _get_bq_client(self):
    """Lazy-load BigQuery client."""
    if self._bq_client is None:
      from google.cloud import bigquery

      self._bq_client = bigquery.Client(
          project=self._project, location=self._location
      )
    return self._bq_client

  def _get_storage_client(self):
    """Lazy-load Cloud Storage client."""
    if self._storage_client is None:
      from google.cloud import storage

      self._storage_client = storage.Client(project=self._project)
    return self._storage_client

  def _get_gcs_uri(self, session_id: str, checkpoint_seq: int) -> str:
    """Generate a GCS URI for a checkpoint blob."""
    return f"gs://{self._gcs_bucket}/checkpoints/{session_id}/{checkpoint_seq}.json.gz"

  async def create_session(
      self,
      *,
      session_id: str,
      agent_name: str,
      metadata: Optional[Dict[str, Any]] = None,
  ) -> SessionMetadata:
    """Create a new durable session."""
    now = datetime.now(timezone.utc)

    # Check if session already exists
    existing = await self.get_session(session_id=session_id)
    if existing:
      raise ValueError(f"Session {session_id} already exists")

    # Insert session record using DML (not streaming) for immediate updatability
    client = self._get_bq_client()
    from google.cloud import bigquery

    insert_query = f"""
    INSERT INTO `{self._sessions_table_id}`
    (session_id, status, agent_name, created_at, updated_at,
     current_checkpoint_seq, active_lease_id, lease_expiry, ttl_expiry, metadata)
    VALUES
    (@session_id, @status, @agent_name, @created_at, @updated_at,
     @current_checkpoint_seq, @active_lease_id, @lease_expiry, @ttl_expiry,
     PARSE_JSON(@metadata))
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
            bigquery.ScalarQueryParameter("status", "STRING", "active"),
            bigquery.ScalarQueryParameter("agent_name", "STRING", agent_name),
            bigquery.ScalarQueryParameter(
                "created_at", "TIMESTAMP", now.isoformat()
            ),
            bigquery.ScalarQueryParameter(
                "updated_at", "TIMESTAMP", now.isoformat()
            ),
            bigquery.ScalarQueryParameter("current_checkpoint_seq", "INT64", 0),
            bigquery.ScalarQueryParameter("active_lease_id", "STRING", None),
            bigquery.ScalarQueryParameter("lease_expiry", "TIMESTAMP", None),
            bigquery.ScalarQueryParameter("ttl_expiry", "TIMESTAMP", None),
            bigquery.ScalarQueryParameter(
                "metadata", "STRING", json.dumps(metadata) if metadata else None
            ),
        ]
    )
    client.query(insert_query, job_config=job_config).result()

    logger.info("Created durable session: %s", session_id)

    return SessionMetadata(
        session_id=session_id,
        status="active",
        agent_name=agent_name,
        created_at=now,
        updated_at=now,
        current_checkpoint_seq=0,
        metadata=metadata,
    )

  async def get_session(self, *, session_id: str) -> Optional[SessionMetadata]:
    """Get session metadata."""
    query = f"""
    SELECT
      session_id,
      status,
      agent_name,
      created_at,
      updated_at,
      current_checkpoint_seq,
      active_lease_id,
      lease_expiry,
      ttl_expiry,
      metadata
    FROM `{self._sessions_table_id}`
    WHERE session_id = @session_id
    """

    client = self._get_bq_client()
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
        ]
    )
    results = client.query(query, job_config=job_config).result()

    for row in results:
      return SessionMetadata(
          session_id=row.session_id,
          status=row.status,
          agent_name=row.agent_name,
          created_at=row.created_at,
          updated_at=row.updated_at,
          current_checkpoint_seq=row.current_checkpoint_seq,
          active_lease_id=row.active_lease_id,
          lease_expiry=row.lease_expiry,
          ttl_expiry=row.ttl_expiry,
          metadata=row.metadata if isinstance(row.metadata, dict) else (json.loads(row.metadata) if row.metadata else None),
      )

    return None

  async def update_session_status(
      self, *, session_id: str, status: str
  ) -> None:
    """Update the status of a session."""
    now = datetime.now(timezone.utc)

    query = f"""
    UPDATE `{self._sessions_table_id}`
    SET status = @status, updated_at = @updated_at
    WHERE session_id = @session_id
    """

    client = self._get_bq_client()
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
            bigquery.ScalarQueryParameter("status", "STRING", status),
            bigquery.ScalarQueryParameter(
                "updated_at", "TIMESTAMP", now.isoformat()
            ),
        ]
    )
    client.query(query, job_config=job_config).result()
    logger.debug("Updated session %s status to %s", session_id, status)

  async def write_checkpoint(
      self,
      *,
      session_id: str,
      checkpoint_seq: int,
      state_blob: bytes,
      agent_state: Optional[Dict[str, Any]] = None,
      trigger: str = "async_boundary",
  ) -> Checkpoint:
    """Write a checkpoint with two-phase commit."""
    import gzip

    now = datetime.now(timezone.utc)

    # Verify session exists and checkpoint_seq is valid
    session = await self.get_session(session_id=session_id)
    if not session:
      raise ValueError(f"Session {session_id} not found")

    if checkpoint_seq <= session.current_checkpoint_seq:
      raise ValueError(
          f"checkpoint_seq {checkpoint_seq} must be greater than current"
          f" {session.current_checkpoint_seq}"
      )

    # Compute hash of the state blob
    sha256 = hashlib.sha256(state_blob).hexdigest()
    size_bytes = len(state_blob)

    # Phase 1: Upload to GCS
    gcs_uri = self._get_gcs_uri(session_id, checkpoint_seq)
    blob_path = gcs_uri.replace(f"gs://{self._gcs_bucket}/", "")

    storage_client = self._get_storage_client()
    bucket = storage_client.bucket(self._gcs_bucket)
    blob = bucket.blob(blob_path)

    compressed = gzip.compress(state_blob)
    blob.upload_from_string(compressed, content_type="application/gzip")
    logger.debug(
        "Uploaded checkpoint blob to %s (%d bytes compressed)",
        gcs_uri,
        len(compressed),
    )

    # Phase 2: Insert checkpoint record using DML
    client = self._get_bq_client()
    from google.cloud import bigquery

    insert_query = f"""
    INSERT INTO `{self._checkpoints_table_id}`
    (session_id, checkpoint_seq, created_at, gcs_state_uri, sha256,
     size_bytes, agent_state_json, trigger)
    VALUES
    (@session_id, @checkpoint_seq, @created_at, @gcs_state_uri, @sha256,
     @size_bytes, PARSE_JSON(@agent_state_json), @trigger)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
            bigquery.ScalarQueryParameter(
                "checkpoint_seq", "INT64", checkpoint_seq
            ),
            bigquery.ScalarQueryParameter(
                "created_at", "TIMESTAMP", now.isoformat()
            ),
            bigquery.ScalarQueryParameter("gcs_state_uri", "STRING", gcs_uri),
            bigquery.ScalarQueryParameter("sha256", "STRING", sha256),
            bigquery.ScalarQueryParameter("size_bytes", "INT64", size_bytes),
            bigquery.ScalarQueryParameter(
                "agent_state_json", "STRING",
                json.dumps(agent_state) if agent_state else None
            ),
            bigquery.ScalarQueryParameter("trigger", "STRING", trigger),
        ]
    )
    try:
      client.query(insert_query, job_config=job_config).result()
    except Exception as e:
      # Rollback: delete the GCS blob
      blob.delete()
      raise RuntimeError(f"Failed to insert checkpoint record: {e}")

    # Phase 3: Update session's current_checkpoint_seq
    from google.cloud import bigquery

    update_query = f"""
    UPDATE `{self._sessions_table_id}`
    SET current_checkpoint_seq = @checkpoint_seq, updated_at = @updated_at
    WHERE session_id = @session_id
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
            bigquery.ScalarQueryParameter(
                "checkpoint_seq", "INT64", checkpoint_seq
            ),
            bigquery.ScalarQueryParameter(
                "updated_at", "TIMESTAMP", now.isoformat()
            ),
        ]
    )
    client.query(update_query, job_config=job_config).result()

    logger.info(
        "Wrote checkpoint %d for session %s (%d bytes, sha256=%s)",
        checkpoint_seq,
        session_id,
        size_bytes,
        sha256[:16],
    )

    return Checkpoint(
        session_id=session_id,
        checkpoint_seq=checkpoint_seq,
        created_at=now,
        gcs_state_uri=gcs_uri,
        sha256=sha256,
        size_bytes=size_bytes,
        agent_state=agent_state,
        trigger=trigger,
    )

  async def read_latest_checkpoint(
      self, *, session_id: str
  ) -> Optional[tuple[Checkpoint, bytes]]:
    """Read the latest checkpoint for a session."""
    session = await self.get_session(session_id=session_id)
    if not session or session.current_checkpoint_seq == 0:
      return None

    return await self.read_checkpoint(
        session_id=session_id, checkpoint_seq=session.current_checkpoint_seq
    )

  async def read_checkpoint(
      self, *, session_id: str, checkpoint_seq: int
  ) -> Optional[tuple[Checkpoint, bytes]]:
    """Read a specific checkpoint."""
    import gzip

    query = f"""
    SELECT
      session_id,
      checkpoint_seq,
      created_at,
      gcs_state_uri,
      sha256,
      size_bytes,
      agent_state_json,
      trigger
    FROM `{self._checkpoints_table_id}`
    WHERE session_id = @session_id AND checkpoint_seq = @checkpoint_seq
    """

    client = self._get_bq_client()
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
            bigquery.ScalarQueryParameter(
                "checkpoint_seq", "INT64", checkpoint_seq
            ),
        ]
    )
    results = client.query(query, job_config=job_config).result()

    checkpoint_row = None
    for row in results:
      checkpoint_row = row
      break

    if not checkpoint_row:
      return None

    # Download blob from GCS
    gcs_uri = checkpoint_row.gcs_state_uri
    blob_path = gcs_uri.replace(f"gs://{self._gcs_bucket}/", "")

    storage_client = self._get_storage_client()
    bucket = storage_client.bucket(self._gcs_bucket)
    blob = bucket.blob(blob_path)

    compressed = blob.download_as_bytes()
    state_blob = gzip.decompress(compressed)

    # Verify integrity
    actual_sha256 = hashlib.sha256(state_blob).hexdigest()
    if actual_sha256 != checkpoint_row.sha256:
      raise RuntimeError(
          "Checkpoint integrity check failed: expected"
          f" {checkpoint_row.sha256}, got {actual_sha256}"
      )

    checkpoint = Checkpoint(
        session_id=checkpoint_row.session_id,
        checkpoint_seq=checkpoint_row.checkpoint_seq,
        created_at=checkpoint_row.created_at,
        gcs_state_uri=checkpoint_row.gcs_state_uri,
        sha256=checkpoint_row.sha256,
        size_bytes=checkpoint_row.size_bytes,
        agent_state=(
            checkpoint_row.agent_state_json if isinstance(checkpoint_row.agent_state_json, dict)
            else (json.loads(checkpoint_row.agent_state_json) if checkpoint_row.agent_state_json else None)
        ),
        trigger=checkpoint_row.trigger,
    )

    logger.debug(
        "Read checkpoint %d for session %s (%d bytes)",
        checkpoint_seq,
        session_id,
        len(state_blob),
    )

    return checkpoint, state_blob

  async def acquire_lease(
      self, *, session_id: str, lease_id: str, timeout_seconds: int
  ) -> bool:
    """Attempt to acquire a lease on a session."""
    now = datetime.now(timezone.utc)
    expiry = now + timedelta(seconds=timeout_seconds)

    # Atomic update: only succeed if no active lease or lease expired
    query = f"""
    UPDATE `{self._sessions_table_id}`
    SET
      active_lease_id = @lease_id,
      lease_expiry = @lease_expiry,
      updated_at = @updated_at
    WHERE
      session_id = @session_id
      AND (active_lease_id IS NULL OR lease_expiry < @now)
    """

    client = self._get_bq_client()
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
            bigquery.ScalarQueryParameter("lease_id", "STRING", lease_id),
            bigquery.ScalarQueryParameter(
                "lease_expiry", "TIMESTAMP", expiry.isoformat()
            ),
            bigquery.ScalarQueryParameter(
                "updated_at", "TIMESTAMP", now.isoformat()
            ),
            bigquery.ScalarQueryParameter("now", "TIMESTAMP", now.isoformat()),
        ]
    )
    result = client.query(query, job_config=job_config).result()

    # Check if the update affected any rows
    if result.num_dml_affected_rows and result.num_dml_affected_rows > 0:
      logger.info("Acquired lease %s on session %s", lease_id, session_id)
      return True
    else:
      logger.debug("Failed to acquire lease on session %s", session_id)
      return False

  async def release_lease(self, *, session_id: str, lease_id: str) -> None:
    """Release a lease on a session."""
    now = datetime.now(timezone.utc)

    query = f"""
    UPDATE `{self._sessions_table_id}`
    SET
      active_lease_id = NULL,
      lease_expiry = NULL,
      updated_at = @updated_at
    WHERE session_id = @session_id AND active_lease_id = @lease_id
    """

    client = self._get_bq_client()
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
            bigquery.ScalarQueryParameter("lease_id", "STRING", lease_id),
            bigquery.ScalarQueryParameter(
                "updated_at", "TIMESTAMP", now.isoformat()
            ),
        ]
    )
    client.query(query, job_config=job_config).result()
    logger.info("Released lease %s on session %s", lease_id, session_id)

  async def renew_lease(
      self, *, session_id: str, lease_id: str, timeout_seconds: int
  ) -> bool:
    """Renew an existing lease."""
    now = datetime.now(timezone.utc)
    expiry = now + timedelta(seconds=timeout_seconds)

    query = f"""
    UPDATE `{self._sessions_table_id}`
    SET lease_expiry = @lease_expiry, updated_at = @updated_at
    WHERE session_id = @session_id AND active_lease_id = @lease_id
    """

    client = self._get_bq_client()
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
            bigquery.ScalarQueryParameter("lease_id", "STRING", lease_id),
            bigquery.ScalarQueryParameter(
                "lease_expiry", "TIMESTAMP", expiry.isoformat()
            ),
            bigquery.ScalarQueryParameter(
                "updated_at", "TIMESTAMP", now.isoformat()
            ),
        ]
    )
    result = client.query(query, job_config=job_config).result()

    if result.num_dml_affected_rows and result.num_dml_affected_rows > 0:
      logger.debug("Renewed lease %s on session %s", lease_id, session_id)
      return True
    return False

  async def list_checkpoints(
      self, *, session_id: str, limit: int = 10
  ) -> list[Checkpoint]:
    """List checkpoints for a session."""
    query = f"""
    SELECT
      session_id,
      checkpoint_seq,
      created_at,
      gcs_state_uri,
      sha256,
      size_bytes,
      agent_state_json,
      trigger
    FROM `{self._checkpoints_table_id}`
    WHERE session_id = @session_id
    ORDER BY checkpoint_seq DESC
    LIMIT @limit
    """

    client = self._get_bq_client()
    from google.cloud import bigquery

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ]
    )
    results = client.query(query, job_config=job_config).result()

    checkpoints = []
    for row in results:
      checkpoints.append(
          Checkpoint(
              session_id=row.session_id,
              checkpoint_seq=row.checkpoint_seq,
              created_at=row.created_at,
              gcs_state_uri=row.gcs_state_uri,
              sha256=row.sha256,
              size_bytes=row.size_bytes,
              agent_state=(
                  row.agent_state_json if isinstance(row.agent_state_json, dict)
                  else (json.loads(row.agent_state_json) if row.agent_state_json else None)
              ),
              trigger=row.trigger,
          )
      )

    return checkpoints

  async def delete_session(self, *, session_id: str) -> None:
    """Delete a session and all its checkpoints."""
    # Delete checkpoints from GCS
    checkpoints = await self.list_checkpoints(session_id=session_id, limit=1000)
    storage_client = self._get_storage_client()
    bucket = storage_client.bucket(self._gcs_bucket)

    for checkpoint in checkpoints:
      blob_path = checkpoint.gcs_state_uri.replace(
          f"gs://{self._gcs_bucket}/", ""
      )
      blob = bucket.blob(blob_path)
      try:
        blob.delete()
      except Exception as e:
        logger.warning("Failed to delete blob %s: %s", blob_path, e)

    # Delete checkpoint records
    client = self._get_bq_client()
    from google.cloud import bigquery

    delete_checkpoints = f"""
    DELETE FROM `{self._checkpoints_table_id}`
    WHERE session_id = @session_id
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
        ]
    )
    client.query(delete_checkpoints, job_config=job_config).result()

    # Delete session record
    delete_session = f"""
    DELETE FROM `{self._sessions_table_id}`
    WHERE session_id = @session_id
    """
    client.query(delete_session, job_config=job_config).result()

    logger.info(
        "Deleted session %s and %d checkpoints", session_id, len(checkpoints)
    )
