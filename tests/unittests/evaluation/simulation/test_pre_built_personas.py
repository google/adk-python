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

from google.adk.evaluation.simulation.pre_built_personas import get_default_persona_registry
from google.adk.evaluation.simulation.pre_built_personas import PreBuiltBehaviors
import pytest


def test_get_default_persona_registry():
  """Tests that the default persona registry can be loaded."""
  assert get_default_persona_registry() is not None


@pytest.mark.parametrize(
    'behavior', list(PreBuiltBehaviors), ids=lambda b: b.name
)
def test_pre_built_behavior_renders_instructions_and_rubrics(behavior):
  """Every behavior contributes text to the simulator prompt and its rubrics.

  Both strings are interpolated into the user-simulator instructions and into
  the verifier rubrics, so an empty list here silently produces an empty
  prompt section rather than a visible failure.
  """
  user_behavior = behavior.value
  assert user_behavior.get_behavior_instructions_str().strip()
  assert user_behavior.get_violation_rubrics_str().strip()


def test_pre_built_behaviors_have_no_enum_aliases():
  """Two behaviors with identical contents would collapse into one member.

  `UserBehavior` compares by field value, so an accidentally duplicated
  behavior becomes an `enum` alias: it stays in `__members__` but disappears
  from iteration, and any persona referencing it silently gets the other one.
  """
  assert len(list(PreBuiltBehaviors)) == len(PreBuiltBehaviors.__members__)


@pytest.mark.parametrize('persona_id', ['EXPERT', 'NOVICE', 'EVALUATOR'])
def test_default_personas_compose_distinct_pre_built_behaviors(persona_id):
  """Default personas are built only from distinct `PreBuiltBehaviors`."""
  persona = get_default_persona_registry().get_persona(persona_id)
  known_behaviors = [b.value for b in PreBuiltBehaviors]

  assert persona.behaviors, f'{persona_id} has no behaviors'
  for behavior in persona.behaviors:
    assert behavior in known_behaviors, (
        f'{persona_id} uses a behavior that is not in PreBuiltBehaviors:'
        f' {behavior.name}'
    )
  behavior_names = [b.name for b in persona.behaviors]
  assert len(behavior_names) == len(
      set(behavior_names)
  ), f'{persona_id} lists a behavior more than once: {behavior_names}'
