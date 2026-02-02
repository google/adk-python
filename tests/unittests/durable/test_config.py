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

"""Tests for DurableSessionConfig."""

from google.adk.durable.config import DurableSessionConfig
from pydantic import ValidationError
import pytest


class TestDurableSessionConfig:
  """Tests for DurableSessionConfig model."""

  def test_default_config(self):
    """Test default configuration values."""
    config = DurableSessionConfig()

    assert config.is_durable is False
    assert config.checkpoint_policy == "async_boundary"
    assert config.checkpoint_store is None
    assert config.lease_timeout_seconds == 300
    assert config.max_checkpoint_size_bytes == 10 * 1024 * 1024

  def test_enabled_config(self):
    """Test enabled configuration."""
    config = DurableSessionConfig(
        is_durable=True,
        checkpoint_policy="every_turn",
        lease_timeout_seconds=600,
    )

    assert config.is_durable is True
    assert config.checkpoint_policy == "every_turn"
    assert config.lease_timeout_seconds == 600

  def test_checkpoint_policies(self):
    """Test valid checkpoint policies."""
    for policy in ["async_boundary", "every_turn", "manual"]:
      config = DurableSessionConfig(checkpoint_policy=policy)
      assert config.checkpoint_policy == policy

  def test_invalid_checkpoint_policy(self):
    """Test that invalid checkpoint policies raise validation error."""
    with pytest.raises(ValidationError):
      DurableSessionConfig(checkpoint_policy="invalid_policy")

  def test_lease_timeout_bounds(self):
    """Test lease timeout validation bounds."""
    # Valid minimum
    config = DurableSessionConfig(lease_timeout_seconds=60)
    assert config.lease_timeout_seconds == 60

    # Valid maximum
    config = DurableSessionConfig(lease_timeout_seconds=3600)
    assert config.lease_timeout_seconds == 3600

    # Below minimum
    with pytest.raises(ValidationError):
      DurableSessionConfig(lease_timeout_seconds=59)

    # Above maximum
    with pytest.raises(ValidationError):
      DurableSessionConfig(lease_timeout_seconds=3601)

  def test_max_checkpoint_size_bounds(self):
    """Test max checkpoint size validation."""
    # Valid minimum
    config = DurableSessionConfig(max_checkpoint_size_bytes=1024)
    assert config.max_checkpoint_size_bytes == 1024

    # Below minimum
    with pytest.raises(ValidationError):
      DurableSessionConfig(max_checkpoint_size_bytes=1023)

  def test_extra_fields_forbidden(self):
    """Test that extra fields are not allowed."""
    with pytest.raises(ValidationError):
      DurableSessionConfig(unknown_field="value")

  def test_config_serialization(self):
    """Test config can be serialized to dict."""
    config = DurableSessionConfig(
        is_durable=True,
        checkpoint_policy="every_turn",
        lease_timeout_seconds=120,
    )

    data = config.model_dump()

    assert data["is_durable"] is True
    assert data["checkpoint_policy"] == "every_turn"
    assert data["lease_timeout_seconds"] == 120
    assert data["checkpoint_store"] is None
