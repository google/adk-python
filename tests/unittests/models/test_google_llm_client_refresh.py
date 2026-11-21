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

"""Tests for refreshing the genai clients cached by `Gemini`."""

# pylint: disable=protected-access

import asyncio
from functools import cached_property
from unittest import mock

from google.adk.models.apigee_llm import ApigeeLlm
from google.adk.models.google_llm import Gemini
from google.adk.utils import _event_loop_cache
from google.genai import Client
from pydantic import ValidationError
import pytest


class _Clock:

  def __init__(self):
    self.now = 5000.0

  def __call__(self):
    return self.now


@pytest.fixture(name='clock')
def _clock(monkeypatch):
  clock = _Clock()
  monkeypatch.setattr(_event_loop_cache, '_now', clock)
  return clock


@pytest.fixture(name='client_factory', autouse=True)
def _client_factory():
  """Makes every `genai.Client` the model builds a distinct mock."""
  with mock.patch(
      'google.genai.Client',
      side_effect=lambda **_: mock.MagicMock(vertexai=False),
  ) as factory:
    yield factory


def _gemini(**kwargs):
  return Gemini(model='gemini-2.5-flash', **kwargs)


def test_default_never_rebuilds_clients(clock):
  model = _gemini()
  api_client = model.api_client
  live_client = model._live_api_client

  clock.now += 10**9

  assert model.client_max_age is None
  assert model.api_client is api_client
  assert model._live_api_client is live_client


def test_client_is_reused_before_max_age_and_rebuilt_at_it(clock):
  model = _gemini(client_max_age=3600)
  first = model.api_client

  clock.now += 3599
  assert model.api_client is first

  clock.now += 1
  second = model.api_client
  assert second is not first
  assert model.api_client is second


def test_live_client_ages_out_like_the_api_client(clock):
  model = _gemini(client_max_age=60)
  first = model._live_api_client

  clock.now += 59
  assert model._live_api_client is first

  clock.now += 1
  assert model._live_api_client is not first


def test_each_client_ages_from_its_own_build(clock):
  model = _gemini(client_max_age=60)
  api_client = model.api_client
  clock.now += 40
  live_client = model._live_api_client
  clock.now += 20

  assert model.api_client is not api_client
  assert model._live_api_client is live_client


def test_expired_client_is_dropped_not_closed(clock):
  model = _gemini(client_max_age=60)
  first = model.api_client
  clock.now += 60

  _ = model.api_client

  assert first.mock_calls == []


def test_invalidate_clients_rebuilds_both_clients():
  model = _gemini()
  api_client = model.api_client
  live_client = model._live_api_client

  model.invalidate_clients()

  assert model.api_client is not api_client
  assert model._live_api_client is not live_client


def test_invalidate_clients_before_any_use_is_harmless(client_factory):
  model = _gemini()

  model.invalidate_clients()

  assert client_factory.call_count == 0
  assert model.api_client is model.api_client


def test_invalidate_clients_does_not_close_the_old_client():
  model = _gemini()
  first = model.api_client

  model.invalidate_clients()

  assert first.mock_calls == []


def test_invalidate_clients_resets_expiry():
  model = _gemini(client_max_age=60)
  first = model.api_client

  model.invalidate_clients()

  second = model.api_client
  assert second is not first
  assert model.api_client is second


def test_invalidate_clients_applies_to_every_event_loop():
  model = _gemini()

  async def read():
    return model.api_client

  async def scenario():
    in_loop = model.api_client
    model.invalidate_clients()
    return in_loop, model.api_client

  before, after = asyncio.run(scenario())
  assert after is not before
  # Another loop never sees a client built before the invalidation either.
  assert asyncio.run(read()) is not before


def test_loops_keep_separate_clients_and_age_independently(clock):
  model = _gemini(client_max_age=60)
  results = {}

  async def in_loop_a():
    first = model.api_client
    clock.now += 60
    results['a'] = (first, model.api_client)

  async def in_loop_b():
    results['b'] = model.api_client

  asyncio.run(in_loop_a())
  asyncio.run(in_loop_b())

  a_first, a_second = results['a']
  assert a_second is not a_first
  assert results['b'] not in (a_first, a_second)


def test_in_flight_loop_keeps_working_client_across_invalidation():
  """A coroutine holding the old client is unaffected by a rebuild."""
  model = _gemini()
  seen = {}

  async def long_request():
    client = model.api_client
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    seen['old'] = client
    seen['old_still_usable'] = client.mock_calls == []

  async def scenario():
    task = asyncio.create_task(long_request())
    await asyncio.sleep(0)
    model.invalidate_clients()
    seen['new'] = model.api_client
    await task

  asyncio.run(scenario())
  assert seen['new'] is not seen['old']
  assert seen['old_still_usable']


def test_provided_client_is_never_replaced(clock):
  provided = mock.MagicMock(spec=Client)
  model = _gemini(client=provided, client_max_age=1)

  clock.now += 100
  model.invalidate_clients()

  assert model.api_client is provided
  assert model._live_api_client is provided


@pytest.mark.parametrize('value', [0, -1, -0.5, float('nan'), 'soon', [60]])
def test_invalid_client_max_age_is_rejected(value):
  with pytest.raises(ValidationError):
    _gemini(client_max_age=value)


@pytest.mark.parametrize('value', [0.001, 1, 3600, 86400.5])
def test_valid_client_max_age_is_accepted(value):
  assert _gemini(client_max_age=value).client_max_age == value


def test_client_max_age_is_not_serialized():
  assert 'client_max_age' not in _gemini(client_max_age=60).model_dump()


def test_subclass_overriding_api_client_keeps_its_own_client(clock):
  pinned = mock.MagicMock(vertexai=False)

  class PinnedGemini(Gemini):

    @cached_property
    def api_client(self):
      return pinned

  model = PinnedGemini(model='gemini-2.5-flash', client_max_age=1)
  assert model.api_client is pinned
  live_client = model._live_api_client

  clock.now += 100
  assert model.api_client is pinned
  model.invalidate_clients()

  # The override owns its client; the live client is still refreshed.
  assert model.api_client is pinned
  assert model._live_api_client is not live_client


def test_subclass_override_of_api_client_without_caching_is_untouched():
  class DynamicGemini(Gemini):

    @property
    def api_client(self):
      return mock.MagicMock(vertexai=False)

  model = DynamicGemini(model='gemini-2.5-flash', client_max_age=60)

  model.invalidate_clients()

  assert model.api_client is not model.api_client


def test_apigee_subclass_honors_max_age_and_invalidate(clock):
  model = ApigeeLlm(
      model='apigee/gemini-2.5-flash',
      proxy_url='https://proxy.example.com',
  )
  model.client_max_age = 60
  first = model.api_client

  clock.now += 60
  second = model.api_client
  assert second is not first

  model.invalidate_clients()
  assert model.api_client is not second
