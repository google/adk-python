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

"""Tests for paging `list_sessions` with `page_size` and `page_token`.

Every test that takes `session_service` runs against each backend registered in
`_conformance`.
"""

import asyncio
import base64
import json
import time

from google.adk.events.event import Event
from google.adk.sessions import _pagination
from google.adk.sessions.base_session_service import ListSessionsResponse
from google.adk.sessions.session import Session
import pytest

from ._conformance import session_service  # noqa: F401

_APP = 'my_app'


async def _create(service, user_id, session_id, state=None):
  return await service.create_session(
      app_name=_APP, user_id=user_id, session_id=session_id, state=state
  )


async def _create_many(service, count, user_id='user'):
  """Creates sessions s00, s01, ... in order, each a little later than the last."""
  for i in range(count):
    await _create(service, user_id, f's{i:02d}')
    # Distinct update times; ties are covered by their own test.
    await asyncio.sleep(0.005)


async def _walk_from(service, page_token, page_size, **kwargs):
  """Follows next_page_token to the end; returns the pages."""
  pages = []
  while True:
    response = await service.list_sessions(
        app_name=_APP, page_size=page_size, page_token=page_token, **kwargs
    )
    pages.append(response)
    page_token = response.next_page_token
    if page_token is None:
      return pages
    assert len(pages) < 1000, 'paging did not terminate'


async def _walk(service, page_size, **kwargs):
  return await _walk_from(service, None, page_size, **kwargs)


def _ids(pages):
  return [s.id for page in pages for s in page.sessions]


def _keys(sessions):
  return [(s.user_id, s.id) for s in sessions]


@pytest.mark.asyncio
async def test_default_returns_everything_without_token(session_service):
  await _create_many(session_service, 7)

  response = await session_service.list_sessions(app_name=_APP)

  assert [s.id for s in response.sessions] == [f's{i:02d}' for i in range(7)]
  assert response.next_page_token is None


@pytest.mark.asyncio
@pytest.mark.parametrize('page_size', [1, 2, 3, 4, 7, 8, 100])
async def test_paging_yields_every_session_once_in_order(
    session_service, page_size
):
  await _create_many(session_service, 7)
  expected = [f's{i:02d}' for i in range(7)]

  pages = await _walk(session_service, page_size)

  assert _ids(pages) == expected
  assert all(len(p.sessions) <= page_size for p in pages)
  assert all(p.next_page_token is not None for p in pages[:-1])
  assert pages[-1].next_page_token is None
  assert len(pages) == -(-7 // page_size)


@pytest.mark.asyncio
async def test_page_size_larger_than_total_is_a_single_page(session_service):
  await _create_many(session_service, 3)

  response = await session_service.list_sessions(app_name=_APP, page_size=50)

  assert [s.id for s in response.sessions] == ['s00', 's01', 's02']
  assert response.next_page_token is None


@pytest.mark.asyncio
async def test_exact_multiple_has_no_trailing_empty_page(session_service):
  await _create_many(session_service, 6)

  pages = await _walk(session_service, 3)

  assert [len(p.sessions) for p in pages] == [3, 3]
  assert pages[-1].next_page_token is None


@pytest.mark.asyncio
async def test_page_size_one_returns_one_session_per_page(session_service):
  await _create_many(session_service, 4)

  pages = await _walk(session_service, 1)

  assert [len(p.sessions) for p in pages] == [1, 1, 1, 1]
  assert _ids(pages) == ['s00', 's01', 's02', 's03']


@pytest.mark.asyncio
async def test_empty_app_returns_empty_page_without_token(session_service):
  response = await session_service.list_sessions(app_name='nobody', page_size=5)

  assert response.sessions == []
  assert response.next_page_token is None


@pytest.mark.asyncio
async def test_user_filter_pages_only_that_users_sessions(session_service):
  for i in range(5):
    await _create(session_service, 'alice', f'a{i}')
    await asyncio.sleep(0.005)
    await _create(session_service, 'bob', f'b{i}')
    await asyncio.sleep(0.005)

  alice = await _walk(session_service, 2, user_id='alice')
  bob = await _walk(session_service, 4, user_id='bob')
  everyone = await _walk(session_service, 3)

  assert _ids(alice) == [f'a{i}' for i in range(5)]
  assert {s.user_id for p in alice for s in p.sessions} == {'alice'}
  assert _ids(bob) == [f'b{i}' for i in range(5)]
  full = await session_service.list_sessions(app_name=_APP)
  assert len(full.sessions) == 10
  assert _ids(everyone) == [s.id for s in full.sessions]


@pytest.mark.asyncio
async def test_unknown_user_returns_empty_page(session_service):
  await _create_many(session_service, 2)

  response = await session_service.list_sessions(
      app_name=_APP, user_id='stranger', page_size=5
  )

  assert response.sessions == []
  assert response.next_page_token is None


@pytest.mark.asyncio
async def test_page_token_without_page_size_returns_the_rest(session_service):
  await _create_many(session_service, 5)
  first = await session_service.list_sessions(app_name=_APP, page_size=2)

  rest = await session_service.list_sessions(
      app_name=_APP, page_token=first.next_page_token
  )

  assert [s.id for s in first.sessions] == ['s00', 's01']
  assert [s.id for s in rest.sessions] == ['s02', 's03', 's04']
  assert rest.next_page_token is None


@pytest.mark.asyncio
@pytest.mark.parametrize('page_size', [0, -1, -100, True, 1.5, '2'])
async def test_invalid_page_size_is_rejected(session_service, page_size):
  await _create_many(session_service, 2)

  with pytest.raises(ValueError, match='page_size'):
    await session_service.list_sessions(app_name=_APP, page_size=page_size)


@pytest.mark.asyncio
async def test_invalid_page_size_is_rejected_for_empty_app(session_service):
  with pytest.raises(ValueError, match='page_size'):
    await session_service.list_sessions(app_name='nobody', page_size=0)


def _token(payload) -> str:
  raw = json.dumps(payload).encode()
  return base64.urlsafe_b64encode(raw).decode().rstrip('=')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'bad_token',
    [
        '',
        'not a token',
        '!!!!',
        'abc',
        base64.urlsafe_b64encode(b'not json').decode(),
        _token({'a': 1}),
        _token([1, _APP, None, 1.0, 'user']),
        _token([2, _APP, None, 1.0, 'user', 's00']),
        _token([1, _APP, None, 'soon', 'user', 's00']),
        _token([1, _APP, None, True, 'user', 's00']),
        _token([1, 5, None, 1.0, 'user', 's00']),
        _token([1, _APP, None, 1.0, 'user', None]),
    ],
)
async def test_malformed_page_token_is_rejected(session_service, bad_token):
  await _create_many(session_service, 3)

  with pytest.raises(ValueError, match='page_token'):
    await session_service.list_sessions(
        app_name=_APP, page_size=2, page_token=bad_token
    )


@pytest.mark.asyncio
async def test_page_token_of_another_app_is_rejected(session_service):
  await _create_many(session_service, 4)
  await session_service.create_session(app_name='other_app', user_id='user')
  token = (
      await session_service.list_sessions(app_name=_APP, page_size=1)
  ).next_page_token

  with pytest.raises(ValueError, match='page_token'):
    await session_service.list_sessions(
        app_name='other_app', page_size=1, page_token=token
    )


@pytest.mark.asyncio
async def test_page_token_of_another_user_filter_is_rejected(session_service):
  await _create_many(session_service, 4)
  scoped = (
      await session_service.list_sessions(
          app_name=_APP, user_id='user', page_size=1
      )
  ).next_page_token
  unscoped = (
      await session_service.list_sessions(app_name=_APP, page_size=1)
  ).next_page_token

  with pytest.raises(ValueError, match='page_token'):
    await session_service.list_sessions(
        app_name=_APP, page_size=1, page_token=scoped
    )
  with pytest.raises(ValueError, match='page_token'):
    await session_service.list_sessions(
        app_name=_APP, user_id='user', page_size=1, page_token=unscoped
    )
  with pytest.raises(ValueError, match='page_token'):
    await session_service.list_sessions(
        app_name=_APP, user_id='someone_else', page_size=1, page_token=scoped
    )


@pytest.mark.asyncio
async def test_sessions_created_between_pages_are_listed_once(session_service):
  await _create_many(session_service, 4)
  first = await session_service.list_sessions(app_name=_APP, page_size=2)

  await _create(session_service, 'user', 'late1')
  await asyncio.sleep(0.005)
  await _create(session_service, 'user', 'late2')
  rest = await _walk_from(session_service, first.next_page_token, 2)

  assert [s.id for s in first.sessions] == ['s00', 's01']
  assert _ids(rest) == ['s02', 's03', 'late1', 'late2']


@pytest.mark.asyncio
async def test_sessions_deleted_between_pages_do_not_shift_the_rest(
    session_service,
):
  await _create_many(session_service, 6)
  first = await session_service.list_sessions(app_name=_APP, page_size=2)

  # Delete the session the token points at, one already returned, and one that
  # has not been reached yet. An offset would skip s04 here.
  for session_id in ('s01', 's00', 's03'):
    await session_service.delete_session(
        app_name=_APP, user_id='user', session_id=session_id
    )
  rest = await _walk_from(session_service, first.next_page_token, 2)

  assert _ids(rest) == ['s02', 's04', 's05']


@pytest.mark.asyncio
async def test_session_updated_between_pages_moves_to_the_end(session_service):
  await _create_many(session_service, 5)
  first = await session_service.list_sessions(app_name=_APP, page_size=2)
  returned = await session_service.get_session(
      app_name=_APP, user_id='user', session_id='s00'
  )
  await session_service.append_event(
      session=returned,
      event=Event(invocation_id='inv', author='user', timestamp=time.time()),
  )

  rest = await _walk_from(session_service, first.next_page_token, 2)

  assert [s.id for s in first.sessions] == ['s00', 's01']
  # Nothing is skipped; the updated session is listed again at its new position.
  assert _ids(rest) == ['s02', 's03', 's04', 's00']


@pytest.mark.asyncio
@pytest.mark.parametrize('page_size', [1, 2, 3, 4, 100])
async def test_sessions_with_equal_update_times_have_a_stable_order(
    session_service, page_size
):
  # Every session gets the same update time from an event stamped with the same
  # timestamp, so only the (user_id, id) tie-break orders them.
  # Exactly representable, and exact to the microsecond in every backend.
  timestamp = float(int(time.time()) + 1000) + 0.5
  sessions = []
  for user_id, session_id in [
      ('bob', 'a'),
      ('alice', 'z'),
      ('alice', 'b'),
      ('carol', 'a'),
      ('bob', 'b'),
      ('alice', 'a'),
  ]:
    sessions.append(await _create(session_service, user_id, session_id))
  for session in sessions:
    await session_service.append_event(
        session=session,
        event=Event(invocation_id='inv', author='user', timestamp=timestamp),
    )
  expected = [
      ('alice', 'a'),
      ('alice', 'b'),
      ('alice', 'z'),
      ('bob', 'a'),
      ('bob', 'b'),
      ('carol', 'a'),
  ]

  pages = await _walk(session_service, page_size)

  assert {s.last_update_time for p in pages for s in p.sessions} == {timestamp}
  assert _keys([s for p in pages for s in p.sessions]) == expected
  full = await session_service.list_sessions(app_name=_APP)
  assert _keys(full.sessions) == expected


@pytest.mark.asyncio
async def test_listed_sessions_keep_merged_app_and_user_state(session_service):
  await _create(
      session_service,
      'alice',
      'a1',
      state={'app:theme': 'dark', 'user:lang': 'en', 'k': 'a1'},
  )
  await asyncio.sleep(0.005)
  await _create(
      session_service, 'bob', 'b1', state={'user:lang': 'fr', 'k': 'b1'}
  )
  await asyncio.sleep(0.005)
  await _create(session_service, 'alice', 'a2', state={'k': 'a2'})
  await asyncio.sleep(0.005)
  await _create(session_service, 'carol', 'c1', state={'k': 'c1'})

  expected = {
      'a1': {'app:theme': 'dark', 'user:lang': 'en', 'k': 'a1'},
      'b1': {'app:theme': 'dark', 'user:lang': 'fr', 'k': 'b1'},
      'a2': {'app:theme': 'dark', 'user:lang': 'en', 'k': 'a2'},
      'c1': {'app:theme': 'dark', 'k': 'c1'},
  }
  for page_size in (1, 2, 4):
    pages = await _walk(session_service, page_size)
    assert {s.id: s.state for p in pages for s in p.sessions} == expected
  alice = await _walk(session_service, 1, user_id='alice')
  assert {s.id: s.state for p in alice for s in p.sessions} == {
      'a1': expected['a1'],
      'a2': expected['a2'],
  }


def test_list_sessions_response_defaults_to_no_next_page_token():
  assert ListSessionsResponse().next_page_token is None
  assert ListSessionsResponse(sessions=[]).next_page_token is None


# Unit tests of the shared cursor helpers.


def _session(user_id, session_id, last_update_time):
  return Session(
      app_name=_APP,
      user_id=user_id,
      id=session_id,
      last_update_time=last_update_time,
  )


def test_page_token_round_trips_the_cursor_exactly():
  last = _session('us:er/é', 's:1', 1767225600.123456)

  token = _pagination.encode_page_token(
      app_name=_APP, user_id='us:er/é', last_session=last
  )
  cursor = _pagination.decode_page_token(
      token, app_name=_APP, user_id='us:er/é'
  )

  assert cursor == _pagination.PageCursor(1767225600.123456, 'us:er/é', 's:1')
  assert token.isascii() and '=' not in token and '/' not in token


def test_decode_page_token_without_token_is_no_cursor():
  assert (
      _pagination.decode_page_token(None, app_name=_APP, user_id=None) is None
  )


def test_paginate_sorted_sessions_slices_after_the_cursor():
  sessions = [_session('u', f's{i}', float(i)) for i in range(5)]
  page, token = _pagination.paginate_sorted_sessions(
      sessions, app_name=_APP, user_id=None, page_size=2, page_token=None
  )
  assert [s.id for s in page] == ['s0', 's1']
  assert token is not None

  # The cursor is a position in the order, so it works on a changed list.
  changed = [s for s in sessions if s.id != 's1'] + [_session('u', 'new', 9.0)]
  page, token = _pagination.paginate_sorted_sessions(
      changed, app_name=_APP, user_id=None, page_size=10, page_token=token
  )
  assert [s.id for s in page] == ['s2', 's3', 's4', 'new']
  assert token is None
