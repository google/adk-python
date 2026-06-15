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

"""Tests for GcsArtifactService path traversal prevention.

Covers _validate_gcs_path_segment() directly and verifies that all four
public operations (save, load, delete, list) reject traversal inputs
before reaching the GCS backend.
"""

import pytest
from unittest.mock import MagicMock, patch

from google.adk.artifacts.gcs_artifact_service import GcsArtifactService
from google.adk.errors.input_validation_error import InputValidationError
from google.genai import types


@pytest.fixture
def gcs_service():
    """Create a GcsArtifactService with a mocked GCS client."""
    with patch("google.cloud.storage.Client") as mock_client:
        mock_bucket = MagicMock()
        mock_client.return_value.bucket.return_value = mock_bucket
        service = GcsArtifactService(bucket_name="test-bucket")
        yield service


# ---------------------------------------------------------------------------
# Unit tests for _validate_gcs_path_segment
# ---------------------------------------------------------------------------
class TestGcsPathSegmentValidation:
    """Tests for _validate_gcs_path_segment input validation."""

    def test_valid_user_id_passes(self):
        """Normal user IDs should pass validation."""
        GcsArtifactService._validate_gcs_path_segment("user-123", "user_id")
        GcsArtifactService._validate_gcs_path_segment("alice@example.com", "user_id")
        GcsArtifactService._validate_gcs_path_segment("user_with_underscores", "user_id")

    def test_traversal_user_id_blocked(self):
        """user_id containing ../ should be rejected."""
        with pytest.raises(InputValidationError, match="must not contain path separators"):
            GcsArtifactService._validate_gcs_path_segment("../other-user", "user_id")

    def test_traversal_double_dot_blocked(self):
        """user_id that is exactly '..' should be rejected."""
        with pytest.raises(InputValidationError, match="must not contain traversal segments"):
            GcsArtifactService._validate_gcs_path_segment("..", "user_id")

    def test_single_dot_blocked(self):
        """user_id that is exactly '.' should be rejected."""
        with pytest.raises(InputValidationError, match="must not contain traversal segments"):
            GcsArtifactService._validate_gcs_path_segment(".", "user_id")

    def test_slash_in_user_id_blocked(self):
        """user_id containing forward slash should be rejected."""
        with pytest.raises(InputValidationError, match="must not contain path separators"):
            GcsArtifactService._validate_gcs_path_segment("user/evil", "user_id")

    def test_backslash_in_user_id_blocked(self):
        """user_id containing backslash should be rejected."""
        with pytest.raises(InputValidationError, match="must not contain path separators"):
            GcsArtifactService._validate_gcs_path_segment("user\\evil", "user_id")

    def test_null_byte_blocked(self):
        """user_id containing null byte should be rejected."""
        with pytest.raises(InputValidationError, match="must not contain null bytes"):
            GcsArtifactService._validate_gcs_path_segment("user\x00evil", "user_id")

    def test_empty_value_blocked(self):
        """Empty user_id should be rejected."""
        with pytest.raises(InputValidationError, match="must not be empty"):
            GcsArtifactService._validate_gcs_path_segment("", "user_id")

    def test_app_name_traversal_blocked(self):
        """app_name containing traversal should be rejected."""
        with pytest.raises(InputValidationError, match="must not contain path separators"):
            GcsArtifactService._validate_gcs_path_segment("../other-app", "app_name")

    def test_session_id_traversal_blocked(self):
        """session_id containing traversal should be rejected."""
        with pytest.raises(InputValidationError, match="must not contain path separators"):
            GcsArtifactService._validate_gcs_path_segment("../other-session", "session_id")


# ---------------------------------------------------------------------------
# Operation-level tests: save, load, delete, list
# Each verifies that traversal in user_id is rejected BEFORE any GCS call.
# ---------------------------------------------------------------------------
class TestSaveArtifactTraversal:
    """Verify save_artifact rejects path-traversal inputs."""

    @pytest.mark.asyncio
    async def test_save_rejects_traversal_user_id(self, gcs_service):
        """save_artifact with user_id='../victim' should raise before GCS write."""
        artifact = types.Part.from_bytes(data=b"secret", mime_type="text/plain")
        with pytest.raises(InputValidationError, match="must not contain path separators"):
            await gcs_service.save_artifact(
                app_name="app",
                user_id="../victim-user",
                filename="user:secret-data",
                artifact=artifact,
            )

    @pytest.mark.asyncio
    async def test_save_rejects_traversal_app_name(self, gcs_service):
        """save_artifact with app_name='../other-app' should raise."""
        artifact = types.Part.from_bytes(data=b"data", mime_type="text/plain")
        with pytest.raises(InputValidationError, match="must not contain path separators"):
            await gcs_service.save_artifact(
                app_name="../other-app",
                user_id="user-1",
                filename="user:file",
                artifact=artifact,
            )

    @pytest.mark.asyncio
    async def test_save_rejects_null_byte_user_id(self, gcs_service):
        """save_artifact with null byte in user_id should raise."""
        artifact = types.Part.from_bytes(data=b"data", mime_type="text/plain")
        with pytest.raises(InputValidationError, match="must not contain null bytes"):
            await gcs_service.save_artifact(
                app_name="app",
                user_id="user\x00evil",
                filename="user:file",
                artifact=artifact,
            )


class TestLoadArtifactTraversal:
    """Verify load_artifact rejects path-traversal inputs."""

    @pytest.mark.asyncio
    async def test_load_rejects_traversal_user_id(self, gcs_service):
        """load_artifact with user_id='../victim' should raise before GCS read."""
        with pytest.raises(InputValidationError, match="must not contain path separators"):
            await gcs_service.load_artifact(
                app_name="app",
                user_id="../victim-user",
                filename="user:secret-data",
            )

    @pytest.mark.asyncio
    async def test_load_rejects_traversal_session_id(self, gcs_service):
        """load_artifact with traversal in session_id should raise."""
        with pytest.raises(InputValidationError, match="must not contain path separators"):
            await gcs_service.load_artifact(
                app_name="app",
                user_id="user-1",
                filename="file.txt",
                session_id="../other-session",
            )


class TestDeleteArtifactTraversal:
    """Verify delete_artifact rejects path-traversal inputs."""

    @pytest.mark.asyncio
    async def test_delete_rejects_traversal_user_id(self, gcs_service):
        """delete_artifact with user_id='../victim' should raise before GCS delete."""
        with pytest.raises(InputValidationError, match="must not contain path separators"):
            await gcs_service.delete_artifact(
                app_name="app",
                user_id="../victim-user",
                filename="user:secret-data",
            )

    @pytest.mark.asyncio
    async def test_delete_rejects_backslash_user_id(self, gcs_service):
        """delete_artifact with backslash in user_id should raise."""
        with pytest.raises(InputValidationError, match="must not contain path separators"):
            await gcs_service.delete_artifact(
                app_name="app",
                user_id="user\\evil",
                filename="user:file",
            )


class TestListArtifactKeysTraversal:
    """Verify list_artifact_keys rejects path-traversal inputs."""

    @pytest.mark.asyncio
    async def test_list_rejects_traversal_user_id(self, gcs_service):
        """list_artifact_keys with user_id='../victim' should raise before GCS list."""
        with pytest.raises(InputValidationError, match="must not contain path separators"):
            await gcs_service.list_artifact_keys(
                app_name="app",
                user_id="../victim-user",
            )

    @pytest.mark.asyncio
    async def test_list_rejects_dot_dot_user_id(self, gcs_service):
        """list_artifact_keys with user_id='..' should raise."""
        with pytest.raises(InputValidationError, match="must not contain traversal segments"):
            await gcs_service.list_artifact_keys(
                app_name="app",
                user_id="..",
            )
