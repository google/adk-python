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

"""What a model-supplied value may become, and where each form is allowed."""

import dataclasses

from google.adk.telemetry import _hallucination
import pytest


def test_a_value_is_unconfirmed_until_something_resolves_it():
  """The default is the safe one: nothing the model wrote is taken as real."""
  value = _hallucination.MaybeHallucinated("some-skill")

  assert value.maybe_hallucinated_value == "some-skill"
  assert not isinstance(value, _hallucination.ConfirmedNotHallucinated)


def test_a_value_cannot_be_changed_in_place():
  """Immutable, so a value cannot shift under a holder that already has it."""
  value = _hallucination.MaybeHallucinated("some-skill")

  with pytest.raises(dataclasses.FrozenInstanceError):
    value.maybe_hallucinated_value = "another-skill"


def test_confirming_a_value_does_not_confirm_the_one_it_came_from():
  """A `Confirmed` is a new value, so nothing upgrades by reference."""
  unconfirmed = _hallucination.MaybeHallucinated("some-skill")

  _hallucination.ConfirmedNotHallucinated(unconfirmed.maybe_hallucinated_value)

  assert not isinstance(unconfirmed, _hallucination.ConfirmedNotHallucinated)
  assert unconfirmed.bounded() == "<hallucinated>"


def test_only_a_confirmed_value_reaches_a_metric():
  """The whole point: an unbounded name cannot become a metric label."""
  assert (
      _hallucination.ConfirmedNotHallucinated("some-skill").bounded()
      == "some-skill"
  )
  assert (
      _hallucination.MaybeHallucinated("some-skill").bounded()
      == "<hallucinated>"
  )


def test_standing_is_part_of_a_value_s_identity():
  """Two values that read alike are not interchangeable if one is unresolved."""
  assert _hallucination.MaybeHallucinated(
      "x"
  ) == _hallucination.MaybeHallucinated("x")
  assert _hallucination.ConfirmedNotHallucinated(
      "x"
  ) == _hallucination.ConfirmedNotHallucinated("x")
  assert _hallucination.MaybeHallucinated(
      "x"
  ) != _hallucination.ConfirmedNotHallucinated("x")
