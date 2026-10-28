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

"""Derived trust, lifecycle, and staleness signals (SPEC §5.1-§5.5).

**Trust tiers are advisory signals, not access control (SPEC §5.3).** Never gate a
privileged action — a file write, a network call, a tool execution — on a value
derived from a concept's frontmatter. A concept claiming ``verified: human:<id>``
is still untrusted input if you did not author the bundle.

OKF deliberately records *signals*, never a credibility score: "a score is
subjective, doesn't port across consumers, and goes stale the moment it's written."
So this module exposes the derived tier **and** :func:`verification_actors`, the raw
§7 actors it derived it from — a caller can always see which of the two it reasons from.
"""

from __future__ import annotations

from collections.abc import Mapping
import datetime as dt
from enum import Enum
from typing import Any

HUMAN_PREFIX = "human:"


class TrustTier(str, Enum):
  """Derived from ``verified`` events, never stored (SPEC §5.3)."""

  UNVERIFIED = "unverified"
  MACHINE_CONFIRMED = "machine-confirmed"
  HUMAN_REVIEWED = "human-reviewed"


class Lifecycle(str, Enum):
  """SPEC §5.4. Absent ``status`` means :attr:`STABLE`."""

  DRAFT = "draft"
  STABLE = "stable"
  DEPRECATED = "deprecated"


def _verification_events(fm: Mapping[str, Any]) -> list[Mapping[str, Any]]:
  raw = fm.get("verified")
  if raw is None:
    return []
  if isinstance(raw, Mapping):
    # SPEC §5.2 MUST: a bare {by, at} mapping is a one-element list.
    return [raw]
  if isinstance(raw, list):
    return [event for event in raw if isinstance(event, Mapping)]
  return []


def verification_actors(fm: Mapping[str, Any]) -> tuple[str, ...]:
  """Verifying actors in the §7 convention (``human:<id>``, ``<producer>/<version>``, ...).

  An event with no identifiable ``by`` contributes no signal and is dropped:
  an anonymous verification is not evidence of anything.
  """
  found: list[str] = []
  for event in _verification_events(fm):
    actor = str(event.get("by") or "").strip()
    if actor:
      found.append(actor)
  return tuple(found)


def trust_tier(fm: Mapping[str, Any]) -> TrustTier:
  """Derive the §5.3 tier. Advisory only — see the module docstring."""
  found = verification_actors(fm)
  if not found:
    return TrustTier.UNVERIFIED
  if any(actor.startswith(HUMAN_PREFIX) for actor in found):
    return TrustTier.HUMAN_REVIEWED
  return TrustTier.MACHINE_CONFIRMED


def lifecycle(fm: Mapping[str, Any]) -> Lifecycle:
  """Parse ``status`` (SPEC §5.4). Unknown values fall back to ``stable``.

  §11 forbids rejecting a concept for an unexpected value, so this never raises.
  """
  raw = fm.get("status")
  if isinstance(raw, str):
    try:
      return Lifecycle(raw.strip().lower())
    except ValueError:
      return Lifecycle.STABLE
  return Lifecycle.STABLE


def _parse_okf_datetime(value: Any) -> dt.datetime | None:
  """Parse an OKF timestamp, or return ``None`` if it is not one.

  SPEC §5 preamble: every timestamp is an ISO 8601 datetime with an explicit UTC
  offset. A date-only value such as ``2027-01-01`` is **not** an OKF datetime, and
  a timezone is never guessed for it — it is treated as absent.

  PyYAML already converts unquoted ISO timestamps to ``datetime`` objects, so
  those are accepted too — but only when they carry a timezone. A naive
  datetime is as ambiguous as a date-only string and is likewise ignored.
  """
  if isinstance(value, dt.datetime):
    return value if value.tzinfo is not None else None
  if not isinstance(value, str):
    return None
  candidate = value.strip().replace("Z", "+00:00")
  if "T" not in candidate:
    return None
  try:
    parsed = dt.datetime.fromisoformat(candidate)
  except ValueError:
    return None
  return parsed if parsed.tzinfo is not None else None


def is_stale(fm: Mapping[str, Any], *, now: dt.datetime) -> bool:
  """True when ``now >= stale_after`` (SPEC §5.5). Absolute instants only."""
  deadline = _parse_okf_datetime(fm.get("stale_after"))
  return deadline is not None and now >= deadline
