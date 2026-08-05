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

"""Tests for the uvloop event-loop helpers."""

import asyncio

from google.adk.utils import event_loop
import pytest


@pytest.fixture(autouse=True)
def _reset_install_flag(monkeypatch):
  # Isolate each test from the process-wide install flag and restore the
  # global event-loop policy so installing uvloop here does not leak into
  # the rest of the test session.
  monkeypatch.setattr(event_loop, '_uvloop_installed', False)
  original_policy = asyncio.get_event_loop_policy()
  try:
    yield
  finally:
    asyncio.set_event_loop_policy(original_policy)


def test_enable_uvloop_installs_policy_when_available():
  if not event_loop.is_uvloop_available():
    pytest.skip('uvloop not installed in this environment.')

  assert event_loop.enable_uvloop() is True

  async def _loop_module() -> str:
    return type(asyncio.get_running_loop()).__module__

  assert asyncio.run(_loop_module()).startswith('uvloop')
  assert event_loop.is_uvloop_active() is True


def test_enable_uvloop_is_idempotent():
  if not event_loop.is_uvloop_available():
    pytest.skip('uvloop not installed in this environment.')

  assert event_loop.enable_uvloop() is True
  # Second call short-circuits on the install flag and stays True.
  assert event_loop.enable_uvloop() is True


def test_enable_uvloop_strict_raises_when_unavailable(monkeypatch):
  # Simulate uvloop being absent regardless of what is installed.
  import builtins

  real_import = builtins.__import__

  def _fake_import(name, *args, **kwargs):
    if name == 'uvloop':
      raise ImportError('no uvloop')
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, '__import__', _fake_import)

  assert event_loop.enable_uvloop() is False
  with pytest.raises(RuntimeError):
    event_loop.enable_uvloop(strict=True)


def test_maybe_enable_uvloop_from_env(monkeypatch):
  if not event_loop.is_uvloop_available():
    pytest.skip('uvloop not installed in this environment.')

  monkeypatch.delenv('ADK_UVLOOP', raising=False)
  assert event_loop.maybe_enable_uvloop_from_env() is False

  monkeypatch.setenv('ADK_UVLOOP', '1')
  assert event_loop.maybe_enable_uvloop_from_env() is True
