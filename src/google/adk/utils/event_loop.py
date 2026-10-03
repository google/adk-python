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

"""Event-loop selection helpers.

ADK never pins an event loop or installs a loop policy of its own; every
``asyncio.run`` inside the framework (the sync ``Runner.run`` shim, the CLI,
tests) honours whatever policy is active in the process. That makes swapping
in `uvloop <https://github.com/MagicStack/uvloop>`_ (a libuv-backed loop that
is meaningfully faster for network-bound workloads) a one-line change at your
program's entrypoint::

    import google.adk

    google.adk.enable_uvloop()

Call it once, before the first ``asyncio.run``/``Runner.run``. It sets the
process-wide event-loop policy, so any subsequently created loop — including
the one the sync ``Runner`` spins up on its background thread — uses libuv.

Caveat worth internalising: uvloop only accelerates code that actually awaits
on an asyncio loop. Work offloaded to ``ThreadPoolExecutor`` with a *sync*
client sees no benefit until it is moved onto the loop (async client +
``asyncio.gather``). libuv is the last 10%, not a 10x on its own.
"""

from __future__ import annotations

import asyncio
import logging

from .env_utils import is_env_enabled

logger = logging.getLogger('google_adk.' + __name__)

_UVLOOP_ENV_VAR = 'ADK_UVLOOP'
"""When truthy (``1``/``true``), the sync ``Runner.run`` path auto-enables uvloop."""

# Tracks whether we have already installed the uvloop policy so repeated calls
# (e.g. one at the entrypoint and one auto-triggered by the env var) are cheap
# no-ops instead of re-installing the policy on every invocation.
_uvloop_installed = False


def is_uvloop_available() -> bool:
  """Returns True if the ``uvloop`` package can be imported."""
  try:
    import uvloop  # noqa: F401  pylint: disable=g-import-not-at-top,unused-import

    return True
  except ImportError:
    return False


def enable_uvloop(*, strict: bool = False) -> bool:
  """Installs the uvloop (libuv) event-loop policy process-wide.

  Idempotent: calling it more than once is a cheap no-op after the first
  successful install.

  Args:
    strict: If True, raise ``RuntimeError`` when uvloop is not installed
      instead of silently falling back to the default asyncio loop. Use
      this when the speedup is a hard requirement and a silent fallback
      would mask a misconfigured deployment.

  Returns:
    True if uvloop is now the active policy, False if it was unavailable
    and ``strict`` is False.

  Raises:
    RuntimeError: If ``strict`` is True and uvloop cannot be imported.
  """
  global _uvloop_installed
  if _uvloop_installed:
    return True

  try:
    import uvloop  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    message = (
        'uvloop is not installed. Install it with `pip install'
        ' "google-adk[uvloop]"` (uvloop does not support Windows).'
    )
    if strict:
      raise RuntimeError(message) from e
    logger.info('%s Falling back to the default asyncio event loop.', message)
    return False

  uvloop.install()
  _uvloop_installed = True
  logger.info('uvloop (libuv) event-loop policy installed.')
  return True


def is_uvloop_active() -> bool:
  """Returns True if the running (or default-policy) loop is a uvloop loop.

  Checks the currently running loop when called from inside a coroutine, and
  otherwise inspects the active event-loop policy's loop factory.
  """
  try:
    loop = asyncio.get_running_loop()
    return type(loop).__module__.startswith('uvloop')
  except RuntimeError:
    # No running loop; fall back to inspecting the installed policy.
    policy = asyncio.get_event_loop_policy()
    return type(policy).__module__.startswith('uvloop')


def maybe_enable_uvloop_from_env() -> bool:
  """Enables uvloop iff the ``ADK_UVLOOP`` env var is truthy.

  Lets deployments opt into libuv without touching application code. Called
  internally by the sync ``Runner.run`` path. Returns True if uvloop is
  active after the call.
  """
  if is_env_enabled(_UVLOOP_ENV_VAR):
    return enable_uvloop()
  return _uvloop_installed
