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

"""Tests for the ContextCacheConfig reading shared by prefix-marking models."""

from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.models._prompt_cache import resolve_cache_config
from google.adk.models._prompt_cache import use_one_hour_ttl
from google.adk.models.llm_request import LlmRequest
import pytest


def _request(cache_config=None, previous_prompt_tokens=None):
  return LlmRequest(
      model="test-model",
      cache_config=cache_config,
      cacheable_contents_token_count=previous_prompt_tokens,
  )


def test_no_cache_config_resolves_to_none():
  assert resolve_cache_config(_request()) is None


def test_no_cache_config_resolves_to_none_with_a_known_prompt_size():
  """The size is read only after a config is known to exist."""
  assert resolve_cache_config(_request(previous_prompt_tokens=10_000)) is None


def test_cache_config_resolves_before_the_first_prompt_size_is_known():
  cache_config = ContextCacheConfig(min_tokens=5000)

  assert resolve_cache_config(_request(cache_config)) is cache_config


@pytest.mark.parametrize(
    "previous_prompt_tokens,expected",
    [(0, None), (4999, None), (5000, "config"), (5001, "config")],
)
def test_min_tokens_gates_on_the_previous_prompt_size(
    previous_prompt_tokens, expected
):
  cache_config = ContextCacheConfig(min_tokens=5000)

  resolved = resolve_cache_config(
      _request(cache_config, previous_prompt_tokens)
  )

  assert resolved is (cache_config if expected else None)


def test_a_prompt_size_of_zero_is_a_size_not_an_absent_one():
  """Zero is below any positive minimum, and is not "not measured yet"."""
  cache_config = ContextCacheConfig(min_tokens=1)

  assert resolve_cache_config(_request(cache_config, 0)) is None


@pytest.mark.parametrize(
    "ttl_seconds,expected",
    [(1, False), (300, False), (3599, False), (3600, True), (86400, True)],
)
def test_only_an_hour_or_more_asks_for_the_long_cache(ttl_seconds, expected):
  assert (
      use_one_hour_ttl(ContextCacheConfig(ttl_seconds=ttl_seconds)) is expected
  )
