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

"""Tests for GcsArtifactService path traversal prevention."""

import pytest

from google.adk.artifacts.gcs_artifact_service import GcsArtifactService
from google.adk.errors.input_validation_error import InputValidationError


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
