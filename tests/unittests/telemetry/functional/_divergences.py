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

"""Where the two inference instrumentations disagree, and why.

The goldens are the native recording, and only that: what ADK's own
instrumentation emits is pinned in full, span for span. ``divergences``
says how the OTel instrumentor's recording of the same run differs --
every differing attribute, every span or metric only one of the two emits
-- and ``collate`` turns that into ``functional_divergences.json``: one
group per span / log / metric, one entry per slot within it, each owing a
``kind`` and a ``reason`` saying whose bug it is.

So the gaps live in one reviewable file rather than smeared through every
golden, at the cost of not being able to rebuild the OTel recording from
disk. What a test can still say is that no *new* gap opened.

The comparison is a plain recursive walk over the two recordings as JSON, so
it descends *into* JSON-valued attributes as well: two payloads that differ
in one field diverge in that field, not wholesale.
"""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
import os
from typing import Annotated
from typing import Literal
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import Strict
from pydantic import TypeAdapter
from typing_extensions import TypeAlias
from typing_extensions import TypeAliasType

from ._digests import json_key
from ._digests import TelemetryDigest

# Which library instruments inference for a recording:
#
# * ``native``: ADK's own inference instrumentation, driven by a ``MockModel``
#   that never touches ``google.genai``.
# * ``otel``: opentelemetry-instrumentation-google-genai, driven by a real
#   ``Gemini`` model over a mocked-out ``google.genai``. ADK detects the
#   instrumentor and stands down, so what is recorded is entirely OTel's.
#
# Every scenario is recorded under both, and the two have to agree except
# where a divergence says otherwise.
InferenceInstrumentation = Literal["native", "otel"]
INFERENCE_INSTRUMENTATIONS: tuple[InferenceInstrumentation, ...] = (
    "native",
    "otel",
)

# Who has to change for a divergence to go away: ADK, the OTel instrumentor,
# or nobody -- the two are meant to differ here.
DivergenceKind = Literal["adk_bug", "otel_bug", "desired_behavior"]

# Which kind of thing a divergence sits on.
OwnerKind = Literal["span", "log", "metric"]

# The scalars of a recording. ``Strict``, so that reading an example value
# back gives the one that was written: a lax ``bool`` accepts the integer 1,
# and one instrumentation's ``1`` would come back as the other's ``True``.
Scalar: TypeAlias = Union[
    None,
    Annotated[bool, Strict()],
    Annotated[int, Strict()],
    Annotated[float, Strict()],
    Annotated[str, Strict()],
]

# A recording, as JSON. ``TypeAliasType`` rather than a plain alias: pydantic
# needs a named type to tie the recursion off.
Json = TypeAliasType("Json", Union[Scalar, list["Json"], dict[str, "Json"]])


class Missing(BaseModel):
  """One side of a divergence, where that instrumentation recorded nothing.

  Emitting something the other doesn't is itself a way to disagree, so this
  is a value like any other rather than an absence.
  """

  # ``forbid`` and no default: only exactly ``{"missing_on_this_side": true}``
  # reads back as this, never a recorded object that ignores the extra keys.
  model_config = ConfigDict(frozen=True, extra="forbid")

  missing_on_this_side: Literal[True]


MISSING = Missing(missing_on_this_side=True)

# One instrumentation's side of a slot: what it recorded there, or nothing.
Recorded: TypeAlias = Union[Missing, Json]


@dataclass(frozen=True)
class Divergent:
  """What each inference instrumentation recorded, where they disagree.

  Walked, never stored: what reaches ``functional_divergences.json`` is one
  of these as the example for its slot.
  """

  native_value: Recorded
  otel_value: Recorded

  def recorded(self) -> Recorded:
    """Whichever side actually recorded something, to name the slot after."""
    if isinstance(self.native_value, Missing):
      return self.otel_value
    return self.native_value


# Two recordings overlaid: the same JSON, with a ``Divergent`` wherever they
# disagreed. Built to be walked, never stored -- the goldens hold the native
# recording, and the divergences their own file.
Merged: TypeAlias = Union[Divergent, dict, list, Scalar]


@dataclass(frozen=True)
class DivergenceId:
  """Identifies one slot the two instrumentations disagree on.

  Owner-relative, so a slot is named the same wherever it turns up: the same
  attribute on the same span is one divergence whether that span is nested
  two deep or five, appears twice in a run, or appears in all the goldens.

  ``path`` is the route to the slot *within* its owner, by field name --
  ``("attributes", "gen_ai.input.messages", "parts", "text")``. List indices
  are deliberately left out: a divergence in the second of two messages is
  the same divergence as in the first, and one explanation covers both. An
  empty path means the owner itself: a span, log or metric only one of the
  two records at all.

  Not what is stored: the file groups by owner, and an entry holds the path.
  """

  owner: OwnerKind
  owner_name: str
  path: tuple[str, ...] = ()

  def at(self, key: str) -> DivergenceId:
    return DivergenceId(
        owner=self.owner, owner_name=self.owner_name, path=(*self.path, key)
    )


class Divergence(BaseModel):
  """One slot within an owner, and the explanation a developer owes for it.

  The two example values are from the first recording that produced the
  divergence: enough to see what the gap is, not a record of every case.

  ``kind`` and ``reason`` are null when ``regenerate`` first records the
  divergence; the tests fail until both are filled in.
  """

  path: tuple[str, ...]
  example_native_instrumentation_value: Recorded = Field(
      union_mode="left_to_right"
  )
  example_otel_instrumentation_value: Recorded = Field(
      union_mode="left_to_right"
  )
  kind: Union[DivergenceKind, None] = None
  reason: Union[str, None] = None

  @property
  def explained(self) -> bool:
    return self.kind is not None and bool(self.reason)


class DivergenceGroup(BaseModel):
  """Every divergence on one span, log or metric.

  ``affected_tests`` is the common prefix of the ``<scenario>/<test_id>`` of
  every case that produced one of them, followed by ``*`` -- ``*`` when they
  span every scenario. It is a summary, not a filter: it over-approximates,
  and a group with no affected test is dropped rather than written.
  """

  owner_type: OwnerKind
  owner_name: str
  affected_tests: str
  divergences: list[Divergence]

  @staticmethod
  def by_id(
      groups: list[DivergenceGroup],
  ) -> dict[DivergenceId, Divergence]:
    """The recorded divergences, keyed by the slot each one sits on."""
    return {
        DivergenceId(
            owner=group.owner_type,
            owner_name=group.owner_name,
            path=entry.path,
        ): entry
        for group in groups
        for entry in group.divergences
    }


_DIGEST: TypeAdapter[TelemetryDigest] = TypeAdapter(TelemetryDigest)

# Where the walk starts. The owner is a placeholder: the first thing the walk
# descends into is ``root_span``, which names one.
_ROOT = DivergenceId(owner="span", owner_name="", path=())


@dataclass(frozen=True)
class _Owns:
  """What lies below one field of a digest, for naming divergences after."""

  kind: OwnerKind
  # The field that names the owner. ``None`` where the owner is named by the
  # key that reaches it, rather than by anything on it.
  named_by: Union[str, None]


# The fields at which a new owner begins. Everything below one is named after
# it rather than after the route to it.
_OWNERS: dict[str, _Owns] = {
    "root_span": _Owns(kind="span", named_by="name"),
    "children": _Owns(kind="span", named_by="name"),
    "logs": _Owns(kind="log", named_by="event_name"),
    "metric_points": _Owns(kind="metric", named_by=None),
}

# What lines two lists up, so that only genuinely absent elements come out as
# missing: a span by its name, a log by its event name, a metric point by its
# dimensions. Anything else stands for itself. Ordered by precedence -- a
# span carries attributes too.
_IDENTITY_FIELDS = ("name", "event_name", "attributes")


# ---------------------------------------------------------------------------
# Comparing: two recordings in, the divergences between them out.
# ---------------------------------------------------------------------------


def divergences(
    native: TelemetryDigest, otel: TelemetryDigest
) -> dict[DivergenceId, Divergent]:
  """Every slot the two recordings of one case disagree on.

  In the order the recording emitted them, so the divergence file reads down
  the run rather than down the alphabet. Deduplicated: a slot named the same
  is the same divergence, and owes one explanation -- the first occurrence is
  the one whose values are kept, as the example.
  """
  merged = _merge(
      _DIGEST.dump_python(native, mode="json"),
      _DIGEST.dump_python(otel, mode="json"),
  )
  found: dict[DivergenceId, Divergent] = {}
  for divergence_id, node in _locate(merged, _ROOT):
    found.setdefault(divergence_id, node)
  return found


def collate(
    found: Mapping[str, Mapping[DivergenceId, Divergent]],
    explained: Union[Mapping[DivergenceId, Divergence], None] = None,
) -> list[DivergenceGroup]:
  """Groups what every case found into what ``functional_divergences.json`` holds.

  ``found`` is keyed by ``<scenario>/<test_id>``. Explanations already given
  are carried over by id, so re-recording only ever adds unexplained entries;
  a divergence no case produces any more is simply not in ``found``, and is
  gone from the result.
  """
  explained = explained or {}
  tests: dict[DivergenceId, list[str]] = {}
  example: dict[DivergenceId, Divergent] = {}
  for test, in_test in found.items():
    for divergence_id, node in in_test.items():
      tests.setdefault(divergence_id, []).append(test)
      example.setdefault(divergence_id, node)

  grouped: dict[tuple[OwnerKind, str], list[DivergenceId]] = {}
  for divergence_id in example:
    key = (divergence_id.owner, divergence_id.owner_name)
    grouped.setdefault(key, []).append(divergence_id)

  groups: list[DivergenceGroup] = []
  for (owner, owner_name), ids in grouped.items():
    entries: list[Divergence] = []
    for divergence_id in ids:
      previous = explained.get(divergence_id)
      entries.append(
          Divergence(
              path=divergence_id.path,
              example_native_instrumentation_value=example[
                  divergence_id
              ].native_value,
              example_otel_instrumentation_value=example[
                  divergence_id
              ].otel_value,
              kind=previous.kind if previous is not None else None,
              reason=previous.reason if previous is not None else None,
          )
      )
    affected = sorted({test for i in ids for test in tests[i]})
    groups.append(
        DivergenceGroup(
            owner_type=owner,
            owner_name=owner_name,
            affected_tests=os.path.commonprefix(affected) + "*",
            divergences=entries,
        )
    )
  return groups


def _merge(native: Recorded, otel: Recorded) -> Merged:
  # Compared by canonical JSON rather than by ``==``, which calls True and 1
  # equal and would lose a side. Quadratic on depth; these trees are small.
  if json_key(native) == json_key(otel):
    return native
  if isinstance(native, dict) and isinstance(otel, dict):
    # Native's key order first, then whatever only otel recorded: a golden
    # reads in the order the recording wrote it, not down the alphabet.
    return {
        key: _merge(native.get(key, MISSING), otel.get(key, MISSING))
        for key in dict.fromkeys((*native, *otel))
    }
  if isinstance(native, list) and isinstance(otel, list):
    return _merge_lists(native, otel)
  return Divergent(native_value=native, otel_value=otel)


def _merge_lists(native: list[Json], otel: list[Json]) -> list[Merged]:
  """Overlays two lists, lining their elements up by what identifies them.

  Elements only one side has stay put, as a divergence missing on the other,
  so an extra span or an extra metric point shifts nothing after it.
  """
  opcodes = SequenceMatcher(
      a=[json_key(_identity(element)) for element in native],
      b=[json_key(_identity(element)) for element in otel],
      autojunk=False,
  ).get_opcodes()
  merged: list[Merged] = []
  for _, i1, i2, j1, j2 in opcodes:
    paired = min(i2 - i1, j2 - j1)
    merged += [_merge(native[i1 + n], otel[j1 + n]) for n in range(paired)]
    merged += [_merge(element, MISSING) for element in native[i1 + paired : i2]]
    merged += [_merge(MISSING, element) for element in otel[j1 + paired : j2]]
  return merged


def _identity(element: Json) -> Json:
  if isinstance(element, dict):
    for field in _IDENTITY_FIELDS:
      if field in element:
        return {field: element[field]}
  return element


# ---------------------------------------------------------------------------
# Naming: the overlay in, the id of every divergence in it out.
# ---------------------------------------------------------------------------


def _locate(
    node: Merged, at: DivergenceId
) -> Iterator[tuple[DivergenceId, Divergent]]:
  """Yields the id of every divergence in an overlaid tree, and the divergence."""
  if isinstance(node, Divergent):
    # Keyed by whichever side recorded something: a span only one of the two
    # emits is still that span's divergence, not its parent's.
    yield _owner(at, node.recorded()), node
  elif isinstance(node, dict):
    for key, child in node.items():
      yield from _locate(child, _owner(_descend(at, key), child))
  elif isinstance(node, list):
    for element in node:
      yield from _locate(element, _owner(at, element))


def _descend(at: DivergenceId, key: str) -> DivergenceId:
  """Steps into field ``key``, which may itself name a new owner."""
  owns = _owns(at)
  if owns is not None and owns.named_by is None:
    return DivergenceId(owner=owns.kind, owner_name=key)
  return at.at(key)


def _owner(at: DivergenceId, node: Union[Merged, Missing]) -> DivergenceId:
  """Renames the location after ``node``, where ``node`` names a new owner."""
  owns = _owns(at)
  if owns is None or owns.named_by is None or not isinstance(node, dict):
    return at
  name = node.get(owns.named_by)
  return DivergenceId(
      owner=owns.kind, owner_name=name if isinstance(name, str) else "?"
  )


def _owns(at: DivergenceId) -> Union[_Owns, None]:
  return _OWNERS.get(at.path[-1]) if at.path else None
