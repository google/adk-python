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

"""Tests for the `CascadeLive` model."""

from __future__ import annotations

from google.adk.live import CascadeLive
from google.adk.models.base_llm import BaseLlm
from google.adk.utils import model_name_utils
import pytest

from .test_cascade_live_connection import FakeStt
from .test_cascade_live_connection import FakeTts
from .test_cascade_live_connection import ScriptedReasoner


def test_model_is_the_reasoners_own_name_for_an_instance():
  """An instance passed as `model` normalizes to its name and is stashed."""
  reasoner = ScriptedReasoner()
  cascade = CascadeLive(model=reasoner, stt=FakeStt(), tts=FakeTts())

  # `type is str`, not `isinstance`: the failure this guards against is
  # `model` staying the instance rather than its name, which an `==`
  # assertion alone would not catch.
  assert type(cascade.model) is str
  assert cascade.model == 'scripted'

  # The instance itself is kept for reuse, so no name lookup is needed later.
  assert cascade._llm is reasoner


def test_model_is_the_reasoners_own_name_for_a_string():
  """A reasoner named rather than instantiated is kept verbatim."""
  cascade = CascadeLive(model='gemini-2.0-flash', stt=FakeStt(), tts=FakeTts())

  assert type(cascade.model) is str
  assert cascade.model == 'gemini-2.0-flash'
  # A name defers instantiation: nothing is stashed until `_resolve_llm`.
  assert cascade._llm is None


def test_model_stays_matchable_by_the_frameworks_name_regexes():
  """The stored name is undecorated, so name-based routing still works.

  `model` is fed to anchored patterns (`^gemini-`), not just displayed. A
  prefixed value such as `cascade-gemini-2.0-flash` would leave a cascaded
  Gemini agent silently classified as non-Gemini.
  """
  cascade = CascadeLive(model='gemini-2.0-flash', stt=FakeStt(), tts=FakeTts())

  assert model_name_utils.is_gemini_model(cascade.model)


def test_model_is_required():
  """A missing or empty reasoner is rejected at construction."""
  with pytest.raises(TypeError):
    CascadeLive(  # pytype: disable=missing-parameter
        stt=FakeStt(), tts=FakeTts()
    )
  with pytest.raises(ValueError):
    CascadeLive(model='', stt=FakeStt(), tts=FakeTts())


def test_model_dump_carries_a_name_not_the_reasoner_instance():
  """Serialization records `model` as a name; the stashed instance never leaks.

  A `BaseLlm` passed as `model` is stashed on the private `_llm`, so the
  pydantic dump must hold only its name string - a leaked object here would
  break serialization and defeat the point of normalizing to a name.
  """
  cascade = CascadeLive(model=ScriptedReasoner(), stt=FakeStt(), tts=FakeTts())

  dumped = cascade.model_dump()

  assert type(dumped['model']) is str
  assert dumped['model'] == 'scripted'
  assert 'reasoner' not in dumped
  assert not any(isinstance(value, BaseLlm) for value in dumped.values())
