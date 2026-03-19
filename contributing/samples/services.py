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
"""Example of Python-based service registration."""

from __future__ import annotations

from dummy_services import FooMemoryService
from google.adk.cli.service_registry import get_service_registry


def foo_memory_factory(uri: str, **kwargs) -> FooMemoryService:
  """Factory for FooMemoryService."""
  return FooMemoryService(uri=uri, **kwargs)


# Register the foo memory service with scheme "foo".
# To use this memory service, set --memory_service_uri=foo:// in the ADK CLI.
get_service_registry().register_memory_service("foo", foo_memory_factory)

# The BarMemoryService is registered in services.yaml with scheme "bar".
# To use it, set --memory_service_uri=bar:// in the ADK CLI.
