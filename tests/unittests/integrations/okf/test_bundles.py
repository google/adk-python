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

"""Golden tests against the four published OKF v0.2 sample bundles.

These pin facts measured from the real corpus, so a regression in parsing or
trust derivation shows up immediately. The bundles are vendored (see
``fixtures/bundles/PROVENANCE.md``) to keep the suite offline and deterministic.
"""

from __future__ import annotations

from pathlib import Path
import re

from google.adk.integrations.okf._bundle import Bundle
from google.adk.integrations.okf._concept import split_frontmatter
from google.adk.integrations.okf._trust import Lifecycle
from google.adk.integrations.okf._trust import lifecycle
from google.adk.integrations.okf._trust import trust_tier
from google.adk.integrations.okf._trust import TrustTier
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "bundles"
BUNDLES = ("crypto_bitcoin", "ga4", "stackoverflow", "acme_retail")

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_HEADING = re.compile(r"^#{1,6}\s+(?P<date>\S+)")


def _reserved_file_findings(bundle: Bundle) -> list[str]:
  """OKF §8/§9 violations among a bundle's reserved ``index.md``/``log.md``.

  §8: an ``index.md`` carries no frontmatter, except that the bundle-root one
  may declare ``okf_version``. §9: a date heading in ``log.md`` is ISO 8601.
  Only date-shaped headings are constrained, so a title heading is not one.
  """
  findings: list[str] = []
  for rel_path in bundle.reserved_files():
    path = bundle.root / rel_path
    name = rel_path.rsplit("/", 1)[-1]
    if name == "index.md":
      try:
        frontmatter, _ = split_frontmatter(path.read_text(encoding="utf-8"))
      except ValueError:
        continue  # no frontmatter block: the §8 default, not a violation
      keys = sorted(key for key in frontmatter if key != "okf_version")
      if rel_path != "index.md":
        findings.append(f"{rel_path}: non-root index.md has frontmatter (§8)")
      elif keys:
        findings.append(
            f"{rel_path}: root index.md declares {keys} beyond okf_version (§8)"
        )
    else:
      for line in path.read_text(encoding="utf-8").splitlines():
        match = _DATE_HEADING.match(line)
        if not match or not re.search(r"\d", match["date"]):
          continue
        if not _ISO_DATE.match(match["date"]):
          findings.append(
              f"{rel_path}: date heading {match['date']!r} is not YYYY-MM-DD"
              " (§9)"
          )
  return findings


@pytest.mark.parametrize(
    ("name", "concepts"),
    [
        ("crypto_bitcoin", 9),
        ("ga4", 9),
        ("stackoverflow", 26),
        ("acme_retail", 9),
    ],
)
def test_published_bundle_loads_cleanly_with_measured_counts(name, concepts):
  """Each bundle satisfies §11 rules 1-2, with these exact concept counts."""
  loaded = Bundle.open(FIXTURES / name)
  assert loaded.violations == []
  assert len(loaded.concepts()) == concepts


def test_total_concept_corpus_is_53():
  total = sum(len(Bundle.open(FIXTURES / name).concepts()) for name in BUNDLES)
  assert total == 53


@pytest.mark.parametrize("name", BUNDLES)
def test_reserved_files_follow_sections_8_and_9(name):
  assert _reserved_file_findings(Bundle.open(FIXTURES / name)) == []


def test_acme_retail_is_the_only_bundle_with_verified_concepts():
  """Measured: ``verified`` appears in acme_retail only (8 of 9 concepts)."""
  for name in ("crypto_bitcoin", "ga4", "stackoverflow"):
    tiers = {
        trust_tier(concept.frontmatter)
        for concept in Bundle.open(FIXTURES / name).concepts()
    }
    assert tiers == {TrustTier.UNVERIFIED}, name

  acme = Bundle.open(FIXTURES / "acme_retail").concepts()
  reviewed = [
      concept
      for concept in acme
      if trust_tier(concept.frontmatter) is TrustTier.HUMAN_REVIEWED
  ]
  assert len(reviewed) == 8


def test_acme_retail_is_the_only_bundle_with_attested_computations():
  acme = Bundle.open(FIXTURES / "acme_retail").concepts()
  computations = [c for c in acme if c.is_attested_computation]
  assert computations, "expected at least one attested computation concept"
  for concept in computations:
    assert isinstance(concept.frontmatter.get("runtime"), str)

  for name in ("crypto_bitcoin", "ga4", "stackoverflow"):
    assert not any(
        concept.is_attested_computation
        for concept in Bundle.open(FIXTURES / name).concepts()
    )


def test_producer_defined_keys_are_tolerated_and_preserved():
  """§11 forbids rejecting a bundle for an unknown frontmatter key."""
  loaded = Bundle.open(FIXTURES / "acme_retail")
  extra = [concept for concept in loaded.concepts() if concept.extra_keys]
  assert extra, "expected at least one producer-extension key"
  concept = extra[0]
  assert concept.frontmatter[concept.extra_keys[0]] is not None


@pytest.mark.parametrize("name", BUNDLES)
def test_absent_status_reads_as_stable(name):
  for concept in Bundle.open(FIXTURES / name).concepts():
    assert lifecycle(concept.frontmatter) in set(Lifecycle)
