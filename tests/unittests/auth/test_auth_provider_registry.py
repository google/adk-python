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

"""Unit tests for the AuthProviderRegistry."""

from google.adk.auth.auth_provider_registry import AuthProviderRegistry
from google.adk.auth.base_auth_provider import BaseAuthProvider
from pydantic import BaseModel


class SchemeA(BaseModel):
  pass


class SchemeB(BaseModel):
  pass


class TestAuthProviderRegistry:
  """Test cases for AuthProviderRegistry."""

  def test_register_and_get_provider(self, mocker):
    """Test registering and retrieving providers for different auth scheme types."""
    registry = AuthProviderRegistry()
    provider_a = mocker.create_autospec(BaseAuthProvider, instance=True)
    provider_b = mocker.create_autospec(BaseAuthProvider, instance=True)

    registry.register(SchemeA, provider_a)
    registry.register(SchemeB, provider_b)

    assert registry.get_provider(SchemeA()) is provider_a
    assert registry.get_provider(SchemeB()) is provider_b

  def test_get_unregistered_provider_returns_none(self):
    """Test that get_provider returns None for unregistered scheme types."""
    registry = AuthProviderRegistry()
    assert registry.get_provider(SchemeA()) is None

  def test_register_duplicate_type_overwrites_existing(self, mocker):
    """Test that registering a provider for an existing type overwrites the previous one."""
    registry = AuthProviderRegistry()
    provider_1 = mocker.create_autospec(BaseAuthProvider, instance=True)
    provider_2 = mocker.create_autospec(BaseAuthProvider, instance=True)

    registry.register(SchemeA, provider_1)
    registry.register(SchemeA, provider_2)

    assert registry.get_provider(SchemeA()) is provider_2
