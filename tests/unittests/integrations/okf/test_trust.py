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

"""Tests for derived trust/lifecycle/staleness signals (SPEC §5)."""

from __future__ import annotations

import datetime as dt

from google.adk.integrations.okf._trust import is_stale
from google.adk.integrations.okf._trust import Lifecycle
from google.adk.integrations.okf._trust import lifecycle
from google.adk.integrations.okf._trust import trust_tier
from google.adk.integrations.okf._trust import TrustTier
from google.adk.integrations.okf._trust import verification_actors

NOW = dt.datetime(2026, 10, 5, 12, 0, tzinfo=dt.timezone.utc)


def test_no_verified_is_unverified():
  assert trust_tier({}) is TrustTier.UNVERIFIED


def test_machine_only_actor_is_machine_confirmed():
  fm = {
      "verified": [
          {"by": "reference_agent/gemini-2.5-pro", "at": "2026-08-01T00:00:00Z"}
      ]
  }
  assert trust_tier(fm) is TrustTier.MACHINE_CONFIRMED


def test_human_actor_is_human_reviewed():
  fm = {"verified": [{"by": "human:aldrin", "at": "2026-08-01T00:00:00Z"}]}
  assert trust_tier(fm) is TrustTier.HUMAN_REVIEWED


def test_bare_mapping_is_treated_as_one_element_list():
  """SPEC §5.2 MUST."""
  fm = {"verified": {"by": "human:aldrin", "at": "2026-08-01T00:00:00Z"}}
  assert trust_tier(fm) is TrustTier.HUMAN_REVIEWED


def test_process_actor_is_machine_confirmed():
  fm = {
      "verified": [
          {"by": "process:finance-nightly", "at": "2026-08-01T00:00:00Z"}
      ]
  }
  assert trust_tier(fm) is TrustTier.MACHINE_CONFIRMED


def test_mixed_actors_prefer_human_reviewed():
  fm = {
      "verified": [
          {
              "by": "reference_agent/gemini-2.5-pro",
              "at": "2026-08-01T00:00:00Z",
          },
          {"by": "human:ahormati", "at": "2026-08-02T00:00:00Z"},
      ]
  }
  assert trust_tier(fm) is TrustTier.HUMAN_REVIEWED


def test_absent_status_defaults_to_stable():
  assert lifecycle({}) is Lifecycle.STABLE


def test_status_values_round_trip():
  assert lifecycle({"status": "draft"}) is Lifecycle.DRAFT
  assert lifecycle({"status": "stable"}) is Lifecycle.STABLE
  assert lifecycle({"status": "deprecated"}) is Lifecycle.DEPRECATED


def test_unknown_status_is_not_rejected():
  """§11: consumers MUST NOT reject unknown values."""
  assert lifecycle({"status": "archived-ish"}) is Lifecycle.STABLE


def test_stale_after_absolute_instant():
  fm = {"stale_after": "2026-09-01T00:00:00Z"}
  assert is_stale(fm, now=NOW) is True


def test_future_stale_after_is_not_stale():
  assert is_stale({"stale_after": "2027-01-01T00:00:00Z"}, now=NOW) is False


def test_date_only_stale_after_is_not_a_valid_okf_datetime_and_is_ignored():
  """SPEC §5 preamble: timestamps need an explicit UTC offset. Do not guess a TZ."""
  assert is_stale({"stale_after": "2026-09-01"}, now=NOW) is False


def test_yaml_parsed_datetime_stale_after_is_honoured():
  """PyYAML turns unquoted ISO timestamps into datetime objects before we see them."""
  fm = {"stale_after": dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)}
  assert is_stale(fm, now=NOW) is True
  naive = {"stale_after": dt.datetime(2026, 9, 1)}
  assert is_stale(naive, now=NOW) is False


def test_missing_stale_after_never_stale():
  assert is_stale({}, now=NOW) is False


def test_anonymous_verification_event_is_not_evidence():
  """A `verified` entry with no `by` identifies nobody — it must not confer trust.

  SPEC §5.2 gives every verification event a `by`; an event missing one is
  malformed, and the conservative reading is that it proves nothing. Treating it
  as machine-confirmation would let a malformed bundle inflate its own trust.
  """
  fm = {"verified": [{"at": "2026-08-01T00:00:00Z"}]}
  assert verification_actors(fm) == ()
  assert trust_tier(fm) is TrustTier.UNVERIFIED


def test_verification_actors_exposes_the_raw_signals():
  """The spec stores signals, not a score — so the raw actors stay reachable."""
  fm = {"verified": [{"by": "human:ahormati", "at": "2026-08-01T00:00:00Z"}]}
  assert verification_actors(fm) == ("human:ahormati",)
